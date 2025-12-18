import os
import pandas as pd
import matplotlib.pyplot as plt

RUNS = {
    "vllm_1gpu_debug": "/project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows/runs/alcohol_grpo_vllm_1gpu_debug/metrics.csv",
}

METRICS_TO_PLOT = ["avg_reward", "f1_macro"]
OUTPUT = "/project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows/runs/metrics_binned_50.png"

BIN_SIZE = 50
X_LIMIT = None


def read_metrics_loose(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        # expect: step,avg_reward,accuracy,precision,recall,f1_macro,labels
        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split(",", 6)  # <-- key trick
            if len(parts) < 7:
                print(f"[WARN] skipping short line {lineno}: {line[:120]}")
                continue

            step, avg_reward, accuracy, precision, recall, f1_macro, labels = parts

            # clean labels: remove surrounding quotes and any weird leading quote
            labels = labels.strip().strip('"')
            if labels.startswith('",'):
                labels = labels[2:].strip().strip('"')

            rows.append({
                "step": step,
                "avg_reward": avg_reward,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_macro": f1_macro,
                "labels": labels,
            })

    df = pd.DataFrame(rows)

    # coerce numeric columns
    for c in ["step", "avg_reward", "accuracy", "precision", "recall", "f1_macro"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["step"]).sort_values("step")
    return df


def bin_and_average(df: pd.DataFrame, bin_size: int, metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    df["step_bin"] = (df["step"] // bin_size) * bin_size

    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    agg = df.groupby("step_bin", as_index=False)[metrics].mean()
    agg["step_mid"] = agg["step_bin"] + (bin_size / 2)
    return agg


n = len(METRICS_TO_PLOT)
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(11, 3.5 * n), sharex=True)
if n == 1:
    axes = [axes]

for run_name, csv_path in RUNS.items():
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing: {csv_path}")
        continue

    df = read_metrics_loose(csv_path)

    if X_LIMIT is not None:
        df = df[df["step"] <= X_LIMIT]

    binned = bin_and_average(df, BIN_SIZE, METRICS_TO_PLOT)

    for ax, metric in zip(axes, METRICS_TO_PLOT):
        if metric not in binned.columns:
            print(f"[WARN] {run_name} missing metric: {metric}")
            continue

        ax.plot(binned["step_mid"], binned[metric], marker="o", label=run_name)
        ax.set_title(f"{metric} (avg per {BIN_SIZE} steps)")
        ax.grid(True, alpha=0.3)

        # Only clamp to [0,1] for classification metrics
        if metric != "avg_reward":
            ax.set_ylim(0, 1)

axes[-1].set_xlabel("Step (bin midpoint)")
axes[0].legend(loc="best")
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300)
plt.show()
print(f"Saved: {OUTPUT}")
