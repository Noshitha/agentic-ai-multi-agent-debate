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
