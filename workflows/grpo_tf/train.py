import argparse, asyncio, os, json, copy, random
from .dataset import load_data
from .verify import verify as verify_func
from .prompts import PROBLEM_WITH_EXPERIENCE
from .rollout import rollout_dataset
from .memory import RAGMemory
from .controller import Controller


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


    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch} ===")
        random.shuffle(data)
        for b in range(len(data)//args.batchsize):
            step = epoch*(len(data)//args.batchsize)+b
            step_dir = os.path.join(exp_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            batch = data[b*args.batchsize:(b+1)*args.batchsize]
            experiences = {}
            # formatted = [
            #     {"prompt": format_with_experiences(x["problem"], experiences), **x} for x in batch
            # ]
            formatted = [
                {"prompt": format_with_experiences(x, experiences), **x}
                for x in batch
            ]
            formatted *= args.grpo_n

            rollout_path = os.path.join(step_dir, "rollout.jsonl")
            rollouts = []
            rollouts, stats_step = await rollout_dataset(
                formatted, rollouts, verify_func,
                rollout_filename=rollout_path,
                model_path=args.model_path,
                rollout_concurrency=args.rollout_concurrency,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            stats[f"step_{step}"] = stats_step
            json.dump(stats, open(stats_path, "w"), indent=2)
            print(f"→ Step {step}: avg_reward={stats_step['avg_reward']:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Training-free GRPO")
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
