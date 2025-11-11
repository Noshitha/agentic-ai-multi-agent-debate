# grpo_tf/verify.py
import re

# Accept the strict format and be robust to casing/whitespace
_LABEL_RE = re.compile(r"^\s*Label:\s*(Present|Past|None)\s*$",re.IGNORECASE | re.MULTILINE)

def extract_label(text: str) -> str:
    """
    Parse the model response and return one of: 'Present', 'Past', 'None'.
    Raises ValueError if no label is found.
    """
    m = _LABEL_RE.search(text or "")
    if m:
        return m.group(1).strip().capitalize()

    # Fallback: keyword sniffing to avoid hard zero during cold start
    t = (text or "").lower()
    if "present" in t: return "Present"
    if "past"    in t: return "Past"
    if "none"    in t: return "None"
    raise ValueError("label_not_found")

def verify(sample, groundtruth):
    pred = extract_label(sample["response"])
    gt = groundtruth.strip().capitalize()
    sample["predicted_label"] = pred
    if pred == gt:
        return 1.0
    if gt == "None" and "no alcohol" in sample["response"].lower():
        return 0.5
    if gt == "Past" and "used to" in sample["response"].lower():
        return 0.5
    return 0.2
 