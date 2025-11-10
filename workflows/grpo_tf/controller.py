# grpo_tf/controller.py
import random

class Controller:
    """
    Lightweight controller for Training-Free GRPO + RAG.
    - Tracks reward improvements
    - Decides when to refresh experiences
    - Adjusts exploration temperature adaptively
    """

    def __init__(self, patience=5, reward_threshold=0.55):
        self.best_reward = 0.0
        self.no_improve_count = 0
        self.patience = patience
        self.reward_threshold = reward_threshold
        self.experience_bank = []

    def update(self, avg_reward, new_experiences):
        decision = "continue"

        # Improvement check
        if avg_reward > self.best_reward + 0.01:  # slight tolerance
            self.best_reward = avg_reward
            self.no_improve_count = 0
            self.experience_bank.extend(new_experiences)
        else:
            self.no_improve_count += 1

        # Refresh trigger after stagnation
        if self.no_improve_count >= self.patience:
            if len(self.experience_bank) > 4:
                keep = random.sample(
                    self.experience_bank,
                    k=max(2, len(self.experience_bank) // 2)
                )
                self.experience_bank = keep
            self.no_improve_count = 0
            decision = "refresh_experiences"
            print(f"[Controller] Refreshing experiences. Best reward so far: {self.best_reward:.2f}")


        # Adaptive temperature control
        new_temp = 0.7 if avg_reward < self.reward_threshold else 0.5

        return decision, new_temp
