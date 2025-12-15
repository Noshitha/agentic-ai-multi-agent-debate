# grpo_tf/experience_updater.py
from collections import defaultdict
import re

class ExperienceUpdater:
    """
    Summarizes GRPO rollouts into reusable SDOH experiences.
    Input: grouped rollouts (K per sample)
    Output: short, generalizable textual lessons
    """

    def summarize_group(self, rollouts):
        """
        rollouts: list of rollout dicts for ONE original sample
        """
        correct = [r for r in rollouts if r["reward"] >= 1.0]
        partial = [r for r in rollouts if 0.4 <= r["reward"] < 1.0]
        wrong   = [r for r in rollouts if r["reward"] < 0.4]

        lessons = []

        if correct:
            lessons.append(
                "Correct classifications tend to explicitly align temporal language "
                "(e.g., 'used to', 'previously') with the correct SDOH label."
            )

        if partial:
            lessons.append(
                "Partial credit answers often identify the right concept but miss temporal cues; "
                "carefully distinguish past vs current usage."
            )

        if wrong:
            lessons.append(
                "Incorrect answers frequently ignore negation or temporal qualifiers "
                "such as 'denies', 'no history of', or 'previous'."
            )

        return lessons

    def run(self, rollouts, grpo_n):
        """
        rollouts: flat list (duplicated by GRPO)
        grpo_n: number of rollouts per original sample
        """
        grouped = defaultdict(list)
        for r in rollouts:
            base_id = r["runid"] // grpo_n
            grouped[base_id].append(r)

        experiences = []
        for _, group in grouped.items():
            experiences.extend(self.summarize_group(group))

        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for e in experiences:
            if e not in seen:
                uniq.append(e)
                seen.add(e)

        return uniq
