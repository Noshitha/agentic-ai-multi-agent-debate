# grpo_tf/controller.py

import random

class Controller:
    """
    Lightweight controller for Training-Free GRPO.
    Monitors reward trends and adapts experience and temperature.
    """

    def __init__(self, patience=3, reward_threshold=0.55):
        self.best_reward = 0.0
        self.no_improve_count = 0
        self.patience = patience
        self.reward_threshold = reward_threshold
        self.experience_bank = []

    def update(self, avg_reward, new_experiences):
        """
        Decide whether to refresh experience memory or adjust exploration parameters.
        """
        decision = "continue"

        # Track best reward
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.no_improve_count = 0
            self.experience_bank.extend(new_experiences)
        else:
            self.no_improve_count += 1

        # Forget half the memory if stuck for too long
        if self.no_improve_count > self.patience:
            keep = random.sample(self.experience_bank, k=max(1, len(self.experience_bank)//2))
            self.experience_bank = keep
            self.no_improve_count = 0
            decision = "refresh_experiences"

        # Adjust exploration (temperature)
        new_temp = 0.7 if avg_reward < self.reward_threshold else 0.5

        return decision, new_temp
