import pandas as pd
import matplotlib.pyplot as plt

# Load your metrics file
df = pd.read_csv("/project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows/runs/alcohol_grpo/metrics.csv")

# Set figure style
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10,6))

# Plot key metrics
plt.plot(df["step"], df["avg_reward"], label="Avg Reward", marker="o")
plt.plot(df["step"], df["accuracy"], label="Accuracy", marker="s")
plt.plot(df["step"], df["f1_macro"], label="F1 (macro)", marker="^")

# Optional: add precision/recall if you want
plt.plot(df["step"], df["precision"], linestyle="--", alpha=0.5, label="Precision")
plt.plot(df["step"], df["recall"], linestyle="--", alpha=0.5, label="Recall")

# Labels and formatting
plt.title("Training-Free GRPO + RAG Performance (Alcohol Use Detection)")
plt.xlabel("Step")
plt.ylabel("Score")
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("/project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows/runs/alcohol_grpo/metrics_curve.png", dpi=300)
