from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(samples):
    y_true = [s["groundtruth"].strip().capitalize() for s in samples if "groundtruth" in s]
    y_pred = [s.get("predicted_label", "").strip().capitalize() for s in samples]
    if not y_true or not y_pred:
        return {}

    labels = sorted(set(y_true + y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "labels": labels
    }
