import asyncio, copy, json, os, time
from tqdm import tqdm
from .llm import ZeroShotPolicy

def load_rollouts(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

def save_rollouts(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

async def rollout_dataset(data, rollouts, verify_func, rollout_filename,
                          model_path, rollout_concurrency=4, temperature=0.7, max_tokens=512):
    """Run model rollouts asynchronously and record rewards."""
    if rollouts:
        assert [d["problem"] for d in data] == [r["problem"] for r in rollouts]
    else:
        rollouts = [{"runid": i, **d} for i, d in enumerate(data)]
        save_rollouts(rollouts, rollout_filename)

    q = asyncio.Queue()
    for r in rollouts:
        if "response" not in r:
            await q.put(r)

    pbar = tqdm(total=q.qsize(), desc="Rollouts")
    policy = ZeroShotPolicy(model_path=model_path)

    async def worker(wid):
        while not q.empty():
            sample = await q.get()
            start = time.time()
            try:
                prompt = sample["prompt"]
                coro = asyncio.to_thread(policy.generate, prompt, max_new_tokens=max_tokens, temperature=temperature)
                output = await asyncio.wait_for(coro, timeout=600)
                sample["response"] = output
                sample["reward"] = verify_func(sample, sample["groundtruth"])
                sample["rollout_time"] = time.time() - start
                rollouts[sample["runid"]] = sample
                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)
            except Exception as e:
                sample["response"] = f"Error: {e}"
                sample["reward"] = 0.0
                rollouts[sample["runid"]] = sample
                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)
            finally:
                q.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(rollout_concurrency)]
    await q.join()
    for w in workers:
        w.cancel()
    pbar.close()

    rewards = [r.get("reward", 0.0) for r in rollouts]
    avg = sum(rewards)/len(rewards) if rewards else 0.0
    return rollouts, {"avg_reward": avg}
