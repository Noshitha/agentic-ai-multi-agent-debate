# grpo_tf/rollout.py
import asyncio
import json
import os
import time
from tqdm import tqdm


def load_rollouts(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_rollouts(rows, path: str):
    """
    Writes a full snapshot (simple + safe).
    This overwrites the file each time, so call sparingly (e.g., every N samples + at end).
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "runid": r.get("runid"),
                        "response": r.get("response", ""),
                        "reward": r.get("reward", 0.0),
                        "predicted_label": r.get("predicted_label", ""),
                        "groundtruth": r.get("groundtruth", ""),
                        "rollout_time": r.get("rollout_time", 0.0),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


async def rollout_dataset(
    data,
    rollouts,
    verify_func,
    rollout_filename: str,
    policy,                      # ✅ pre-built (vLLM) policy object
    rollout_concurrency: int = 1, # ✅ force 1 for vLLM stability
    temperature: float = 0.7,
    max_tokens: int = 512,
    task_timeout: float = 600.0,  # kept for API compatibility; not used in sync vLLM call
    save_every: int = 1,          # ✅ for debugging; set to 0 or 5 later for speed
):
    """
    Run model rollouts and record rewards.

    Key design choice:
    - For vLLM, do NOT run multiple concurrent generate() calls on the same engine.
      So we force rollout_concurrency=1 and run generation synchronously.
    """

    # --- initialize rollouts (creates initial JSONL with empty responses) ---
    if not rollouts:
        rollouts = [{"runid": i, **d} for i, d in enumerate(data)]
        save_rollouts(rollouts, rollout_filename)

    # --- queue only unfinished ---
    q = asyncio.Queue()
    for r in rollouts:
        if not r.get("response"):
            await q.put(r)

    pending = q.qsize()
    pbar = tqdm(total=pending, desc="Rollouts")

    # vLLM safety: keep single worker
    rollout_concurrency = 1

    completed = 0

    async def worker(_wid: int):
        nonlocal completed
        while True:
            try:
                sample = q.get_nowait()
            except asyncio.QueueEmpty:
                break

            start = time.time()
            try:
                prompt = sample["prompt"]

                # vLLM call (sync)
                output = policy.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )

                # fill sample fields
                sample["response"] = output
                sample["reward"] = verify_func(sample, sample["groundtruth"])
                sample["rollout_time"] = time.time() - start

                # write back
                rollouts[sample["runid"]] = {
                    "runid": sample["runid"],
                    "response": sample["response"],
                    "reward": sample["reward"],
                    "predicted_label": sample.get("predicted_label", ""),
                    "groundtruth": sample.get("groundtruth", ""),
                    "rollout_time": sample["rollout_time"],
                }

            except Exception as e:
                rollouts[sample["runid"]] = {
                    "runid": sample["runid"],
                    "response": f"Error: {e}",
                    "reward": 0.0,
                    "predicted_label": "",
                    "groundtruth": sample.get("groundtruth", ""),
                    "rollout_time": time.time() - start,
                }
            finally:
                q.task_done()
                pbar.update(1)
                completed += 1

                # periodic checkpoint (super useful while debugging)
                if save_every and (completed % save_every == 0):
                    save_rollouts(rollouts, rollout_filename)

    # run
    worker_task = asyncio.create_task(worker(0))
    await q.join()
    worker_task.cancel()
    await asyncio.gather(worker_task, return_exceptions=True)
    pbar.close()

    # final checkpoint
    save_rollouts(rollouts, rollout_filename)

    rewards = [r.get("reward", 0.0) for r in rollouts]
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    return rollouts, {"avg_reward": avg}
