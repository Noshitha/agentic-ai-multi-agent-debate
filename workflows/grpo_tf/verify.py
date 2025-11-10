# grpo_tf/verify.py
import re

def extract_label(output_text: str):
    match = re.search(r"Label\s*[:\-]*\s*(Present|Past|None)", output_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "UNKNOWN"

def verify(sample, groundtruth):
    pred = extract_label(sample["response"])
    gt = groundtruth.strip().capitalize()
    sample["predicted_label"] = pred
    return 1.0 if pred == gt else 0.0
