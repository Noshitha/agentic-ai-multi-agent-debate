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
            lessons.append("Correct classifications tend to explicitly align temporal language "
                "(e.g., 'used to', 'previously') with the correct SDOH label.")
        if partial:
            lessons.append("Partial credit answers often identify the right concept but miss temporal cues;" 
            "carefully distinguish past vs current usage.")
        if wrong:
            lessons.append("Incorrect answers frequently ignore negation or temporal qualifiers "
                "such as 'denies', 'no history of', or 'previous'.")
        debug = {
            "n_rollouts": len(rollouts),
            "n_correct": len(correct),
            "n_partial": len(partial),
            "n_wrong": len(wrong),
        }
        return lessons, debug

    def run(self, rollouts, grpo_n, return_debug=False):
        """
        rollouts: flat list (duplicated by GRPO)
        grpo_n: number of rollouts per original sample
        """
        grouped = defaultdict(list)
        for r in rollouts:
            base_id = r["runid"] // grpo_n
            grouped[base_id].append(r)

        raw_experiences = []
        debug_by_sample = {}
        for base_id, group in grouped.items():
            lessons, debug = self.summarize_group(group)
            raw_experiences.extend(lessons)
            debug_by_sample[base_id] = debug

        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for e in raw_experiences:
            if e not in seen:
                uniq.append(e)
                seen.add(e)

        stats = {
            "raw_experiences": len(raw_experiences),
            "deduplicated_experiences": len(uniq), 
            "debug_by_sample": debug_by_sample,
        }
        
        if return_debug:
            return uniq, debug_by_sample, stats
        else:
            return uniq
