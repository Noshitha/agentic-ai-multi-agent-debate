# grpo_tf/dataset.py
import json

def load_data(path):
    return [
        {
            "instruction": d["instruction"],
            "text": d["input"],
            "problem": "",  # optional placeholder
            "groundtruth": d["output"],
        }
        for d in map(json.loads, open(path))
    ]


