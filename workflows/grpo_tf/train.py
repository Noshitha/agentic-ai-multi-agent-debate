import pandas as pd
import argparse, asyncio, os, json, copy, random
from .dataset import load_data
from .verify import verify as verify_func
from .prompts import PROBLEM_WITH_EXPERIENCE
from .rollout import rollout_dataset
from .memory import RAGMemory
from .controller import Controller
from .metrics import compute_metrics

random.seed(42)

def format_with_experiences(sample, experiences):
    exp_txt = "\n".join([f"[{i}] {v}" for i, v in experiences.items()]) or "None"
    return PROBLEM_WITH_EXPERIENCE.format(
        experiences=exp_txt,
        instruction=sample["instruction"],
        text=sample["text"]
    )

async def main(args):
    exp_dir = os.path.join("runs", args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    stats_path = os.path.join(exp_dir, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

    data = load_data(args.dataset)
    if args.dataset_truncate:
        data = data[:args.dataset_truncate]
    if len(data) % args.batchsize != 0:
        print(f"Truncating dataset from {len(data)} to {len(data)//args.batchsize * args.batchsize} for clean batching.")
        data = data[: len(data)//args.batchsize * args.batchsize]

    # === Initialize RAG memory and controller ===
    memory = RAGMemory()
    controller = Controller()

    # === Training loop ===
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch} ===")
        random.shuffle(data)
        # inside main(args):
        for b in range(len(data)//args.batchsize):
            step = epoch*(len(data)//args.batchsize)+b
            step_dir = os.path.join(exp_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            batch = data[b*args.batchsize:(b+1)*args.batchsize]

            # Retrieve experiences only if memory has good samples
            formatted_batch = []
            for sample in batch:
                use_rag = len(memory.memory) > 5
                retrieved = memory.retrieve(query=sample["instruction"] + " " + sample["text"], k=3) if use_rag else []
                experiences = {f"exp_{i}": r for i, r in enumerate(retrieved)}
                formatted_batch.append({
                    "prompt": format_with_experiences(sample, experiences),
                    **sample,
                })

            # Duplicate for multiple rollouts (GRPO)
            formatted_batch *= args.grpo_n
            rollout_path = os.path.join(step_dir, "rollout.jsonl")

            # Run rollouts for this batch
            rollouts, stats_step = await rollout_dataset(
                formatted_batch,
                [],
                verify_func,
                rollout_filename=rollout_path,
                model_path=args.model_path,
                rollout_concurrency=args.rollout_concurrency,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            metrics = compute_metrics(rollouts)
            stats_step.update(metrics)

            # Compute step-level stats
            stats[f"step_{step}"] = stats_step
            json.dump(stats, open(stats_path, "w"), indent=2)
            print(f"→ Step {step}: avg_reward={stats_step['avg_reward']:.3f}")

            # after writing stats.json:
            csv_path = os.path.join(exp_dir, "metrics.csv")
            row = {"step": step, **stats_step}
            # flatten non-scalar entries for CSV (like label lists)
            row = {k: (",".join(v) if isinstance(v, list) else v) for k, v in row.items()}
            df = pd.DataFrame([row])

            # append new step to CSV (create if missing)
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode="a", header=False, index=False)

            # Pick best candidate across rollouts
            best_sample = max(rollouts, key=lambda r: r.get("reward", 0))
            best_text = best_sample.get("response", "")
            best_reward = best_sample.get("reward", 0)

            # Add high-reward sample to RAG memory
            if best_reward > 0:
                memory.add(best_text, best_reward)

            # Controller update
            decision, args.temperature = controller.update(
                stats_step['avg_reward'], [best_text]
            )
            if decision == "refresh_experiences":
                print("Controller triggered experience refresh.")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Training-free GRPO + RAG Memory")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--grpo_n", type=int, default=5)
    p.add_argument("--dataset_truncate", type=int, default=None)
    p.add_argument("--rollout_concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=256)
    args = p.parse_args()
    asyncio.run(main(args))