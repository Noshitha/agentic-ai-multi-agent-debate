import asyncio, json, os, time
from tqdm import tqdm
from .llm import ZeroShotPolicy


def load_rollouts(path):
    """Load saved rollouts if they exist."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_rollouts(rows, path):
    """Save minimal rollout info to disk (compact JSONL)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # only keep the essential info
            f.write(json.dumps({
                "runid": r.get("runid"),
                "response": r.get("response", ""),
                "reward": r.get("reward", 0.0),
                "predicted_label": r.get("predicted_label", ""),
                "groundtruth": r.get("groundtruth", ""),
                "rollout_time": r.get("rollout_time", 0.0)
            }, ensure_ascii=False) + "\n")


async def rollout_dataset(
    data,
    rollouts,
    verify_func,
    rollout_filename,
    model_path,
    rollout_concurrency=4,
    temperature=0.7,
    max_tokens=512
):
    """Run model rollouts asynchronously and record rewards."""

    #  Remove unused `problem` field check
    if not rollouts:
        rollouts = [{"runid": i, **d} for i, d in enumerate(data)]
        save_rollouts(rollouts, rollout_filename)

    # queue up samples that need responses
    q = asyncio.Queue()
    for r in rollouts:
        if "response" not in r or not r["response"]:
            await q.put(r)

    pbar = tqdm(total=q.qsize(), desc="Rollouts")
    policy = ZeroShotPolicy(model_path=model_path)

    async def worker(wid):
        while not q.empty():
            sample = await q.get()
            start = time.time()
            try:
                prompt = sample["prompt"]
                # run generation in thread to avoid blocking
                coro = asyncio.to_thread(
                    policy.generate,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                output = await asyncio.wait_for(coro, timeout=600)

                # compute reward
                sample["response"] = output
                sample["reward"] = verify_func(sample, sample["groundtruth"])
                sample["rollout_time"] = time.time() - start
                
                pred_label = sample.get("predicted_label", "")
                # keep memory light — only store small dicts
                rollouts[sample["runid"]] = {
                    "runid": sample["runid"],
                    "response": sample["response"],
                    "reward": sample["reward"],
                    "predicted_label": pred_label,
                    "groundtruth": sample.get("groundtruth", ""),
                    "rollout_time": sample["rollout_time"]
                }

                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)

            except Exception as e:
                rollouts[sample["runid"]] = {
                    "runid": sample["runid"],
                    "response": f"Error: {e}",
                    "reward": 0.0,
                    "predicted_label": "",
                    "groundtruth": sample.get("groundtruth", ""),
                    "rollout_time": time.time() - start
                }
                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)

            finally:
                q.task_done()

    # run async workers
    workers = [asyncio.create_task(worker(i)) for i in range(rollout_concurrency)]
    await q.join()
    for w in workers:
        w.cancel()
    pbar.close()

    rewards = [r.get("reward", 0.0) for r in rollouts]
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    return rollouts, {"avg_reward": avg}
