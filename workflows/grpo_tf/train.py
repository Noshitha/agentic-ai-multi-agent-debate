# grpo_tf/train.py
import pandas as pd
import argparse, asyncio, os, json, random

from .dataset import load_data
from .verify import verify as verify_func
from .prompts import PROBLEM_WITH_EXPERIENCE
from .rollout import rollout_dataset
from .memory import RAGMemory
from .controller import Controller
from .metrics import compute_metrics
from .experience_updater import ExperienceUpdater
from .llm_vllm import ZeroShotPolicyVLLM

random.seed(42)

def format_with_experiences(sample, experiences):
    exp_txt = "\n".join([f"[{i}] {v}" for i, v in experiences.items()]) or "None"

    instruction = sample.get("instruction") or ""
    text = sample.get("text") or ""

    return PROBLEM_WITH_EXPERIENCE.format(
        experiences=exp_txt,
        instruction=instruction,
        text=text
    )


async def main(args):
    exp_dir = os.path.join("runs", args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    stats_path = os.path.join(exp_dir, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

    data = load_data(args.dataset)
    if args.dataset_truncate:
        data = data[:args.dataset_truncate]

    # clean batching
    if len(data) % args.batchsize != 0:
        new_len = (len(data) // args.batchsize) * args.batchsize
        print(f"Truncating dataset from {len(data)} to {new_len} for clean batching.")
        data = data[:new_len]

    # Keep embedding model off GPU so vLLM can start
    memory = RAGMemory(device="cpu") if "device" in RAGMemory.__init__.__code__.co_varnames else RAGMemory()
    controller = Controller()
    updater = ExperienceUpdater()

    # Build vLLM policy ONCE
    policy = ZeroShotPolicyVLLM(model_path=args.model_path)

    num_batches = len(data) // args.batchsize
    csv_path = os.path.join(exp_dir, "metrics.csv")

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch} ===")
        random.shuffle(data)

        for b in range(num_batches):
            step = epoch * num_batches + b
            step_dir = os.path.join(exp_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)

            batch = data[b*args.batchsize:(b+1)*args.batchsize]

            # build prompts with RAG experiences (per sample)
            formatted_batch = [] 
            for sample in batch:
                use_rag = len(getattr(memory, "memory", [])) > 5
                instruction = sample.get("instruction") or ""
                text = sample.get("text") or ""
                query = f"{instruction} {text}".strip()
                retrieved = memory.retrieve(query=query, k=3) if use_rag else []
                experiences = {f"exp_{i}": r for i, r in enumerate(retrieved)}
                formatted_batch.append(
                    {"prompt": format_with_experiences(sample, experiences), 
                    **sample,
                    })

            # Prompt snapshot 
            if step % 200 == 0:
                prompt_snapshots = []

                # log a few representative prompts (before GRPO duplication)
                for i in range(min(3, len(formatted_batch))):
                    prompt_snapshots.append({
                        "instruction": formatted_batch[i].get("instruction", ""),
                        "text": formatted_batch[i].get("text", ""),
                        "prompt": formatted_batch[i]["prompt"],
                    })

                prompt_path = os.path.join(step_dir, f"prompt_snapshot_step_{step}.json")
                with open(prompt_path, "w") as f:
                    json.dump(prompt_snapshots, f, indent=2)


            # GRPO duplication
            formatted_batch = formatted_batch * args.grpo_n
            rollout_path = os.path.join(step_dir, "rollout.jsonl")

            # Rollout (reuse policy)
            rollouts, stats_step = await rollout_dataset(
                data=formatted_batch,
                rollouts=[],
                verify_func=verify_func,
                rollout_filename=rollout_path,
                policy=policy,
                rollout_concurrency=args.rollout_concurrency,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                task_timeout=args.task_timeout,
                save_every=0,  # save only once per batch
            )

            # Metrics
            stats_step.update(compute_metrics(rollouts))
            
            # GRPO-style summarized experiences (deduplicated+ experiences per sample stats)
            # new_experiences = updater.run(rollouts, grpo_n=args.grpo_n)

            new_experiences, debug_by_sample, experiences_stats = updater.run(rollouts, grpo_n=args.grpo_n, return_debug=True)
            added = 0
            if new_experiences:
                added = memory.add_experiences(new_experiences, reward=stats_step["avg_reward"])

            #Log experience/memory stats BEFORE writing stats/csv

            if isinstance(experiences_stats,dict):
                stats_step["raw_experiences"] = experiences_stats.get("raw_experiences",0)
                stats_step["deduplicated_experiences"] = experiences_stats.get("deduplicated_experiences",0)
            else:
                stats_step["raw_experiences"] = 0
                stats_step["deduplicated_experiences"] = 0

            stats_step["memory_size"] = len(getattr(memory, "memory", []))  #len(memory.memory)
            stats_step["memory_added"] = added 


            #save experiences list to json file
            experiences_path = os.path.join(step_dir, "experiences.json")
            with open(experiences_path, "w") as f:
                json.dump(new_experiences, f, indent=2)

            #save debug_by_sample to json file
            debug_path = os.path.join(step_dir, "debug.json")
            with open(debug_path, "w") as f:
                json.dump(debug_by_sample, f, indent=2)
            
            # Persist stats.jso after all fields are added
            stats[f"step_{step}"] = stats_step
            json.dump(stats, open(stats_path, "w"), indent=2)

            print(
                f"→ Step {step}: avg_reward={stats_step['avg_reward']:.3f} | "
                f"f1={stats_step.get('f1_macro', 0):.3f} | "
                f"raw_exp={stats_step.get('raw_experiences', 0)} | "
                f"dedup_exp={stats_step.get('deduplicated_experiences', 0)} | "
                f"mem={stats_step.get('memory_size', 0)} (+{stats_step.get('memory_added', 0)})"
            )

            # CSV logging (includes new fields)
            row = {"step": step, **stats_step}
            row = {k: (",".join(v) if isinstance(v, list) else v) for k, v in row.items()}
            df = pd.DataFrame([row])
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode="a", header=False, index=False)

            # Controller update
            decision, args.temperature = controller.update(stats_step["avg_reward"], new_experiences)

            if decision == "refresh_experiences":
                print("Controller triggered experience refresh.")
                if len(getattr(memory, "memory", [])) > 10:
                    memory.memory = random.sample(memory.memory, k=max(5, len(memory.memory)//2))

if __name__ == "__main__":
    p = argparse.ArgumentParser("Training-free GRPO + RAG Memory")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batchsize", type=int, default=10)
    p.add_argument("--grpo_n", type=int, default=5)
    p.add_argument("--dataset_truncate", type=int, default=None)
    p.add_argument("--rollout_concurrency", type=int, default=1)  
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--task_timeout", type=float, default=600)
    args = p.parse_args()
    asyncio.run(main(args))