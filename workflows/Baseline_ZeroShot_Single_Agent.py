import os, json, argparse, random, re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        #torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
        device_map=None,   
        trust_remote_code=True
    )
    model.to("cuda")  
    print(">>> Model loaded on:", next(model.parameters()).device)
    return tokenizer, model

# ----------------------------
# QUERY MODEL
# ----------------------------
def query_single_agent(tokenizer, model, instruction, text, max_new_tokens=256):
    """Ask model to reason step-by-step and produce a single label."""
    seed = 42
    random.seed(seed)

    agent_prompt = (
        f"{instruction}\n\n"
        f"Text:\n{text}\n\n"
        "You are an expert clinician. Follow this strict format:\n"
        "Reasoning: <one or two sentences explaining your decision>\n"
        "Label: <Present | Past | None>\n"
    )

    # inputs = tokenizer(agent_prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(agent_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=max_new_tokens
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


# ----------------------------
# EXTRACT LABEL
# ----------------------------
def extract_label(output_text):
    match = re.search(r"Label\s*[:\-]*\s*(Present|Past|None)", output_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "UNKNOWN"


# ----------------------------
# COMPUTE METRICS
# ----------------------------
def compute_metrics(results):
    labels = ["Present", "Past", "None", "UNKNOWN"]
    confusion = {g: {p: 0 for p in labels} for g in labels}

    # Build confusion matrix
    for r in results:
        g, p = r["gold"], r["predicted"]
        if g not in labels:
            g = "UNKNOWN"
        if p not in labels:
            p = "UNKNOWN"
        confusion[g][p] += 1

    total_correct = sum(confusion[g][g] for g in labels)
    total = sum(sum(confusion[g].values()) for g in labels)
    accuracy = total_correct / total if total else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for lbl in labels:
        tp = confusion[lbl][lbl]
        fp = sum(confusion[g][lbl] for g in labels if g != lbl)
        fn = sum(confusion[lbl][p] for p in labels if p != lbl)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[lbl].values())
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(labels)
    macro_recall    = sum(recalls) / len(labels)
    macro_f1        = sum(f1s) / len(labels)

    metrics = {
        "overall": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4)
        },
        "per_class": per_class,
        "confusion_matrix": confusion
    }
    return metrics


# ----------------------------
# EVALUATE MODEL ON TEST DATA
# ----------------------------
def evaluate_model(model_path, train_path, test_path, results_dir="outputs/alcohol_eval"):
    os.makedirs(results_dir, exist_ok=True)
    tokenizer, model = load_model(model_path)

    print(f"\n Loading test data from {test_path}")
    test_data = [json.loads(line) for line in open(test_path)]

    results = []
    for sample in tqdm(test_data, desc="Evaluating test set"):
        instruction = sample["instruction"]
        text = sample["input"]
        gold = sample["output"]

        output = query_single_agent(tokenizer, model, instruction, text)
        pred = extract_label(output)

        results.append({
            "input": text,
            "gold": gold,
            "predicted": pred,
            "match": pred.lower() == gold.lower(),
            "output_raw": output
        })

    metrics = compute_metrics(results)
    out_path = os.path.join(results_dir, "eval_metrics.json")

    with open(out_path, "w") as f:
        json.dump({
            "model": model_path,
            "metrics": metrics,
            "results": results
        }, f, indent=2)

    print("\n==== Evaluation Complete ====")
    print(f"Overall Accuracy       : {metrics['overall']['accuracy']*100:.2f}%")
    print(f"Macro Precision / Recall / F1 : "
        f"{metrics['overall']['macro_precision']:.2f} / "
        f"{metrics['overall']['macro_recall']:.2f} / "
        f"{metrics['overall']['macro_f1']:.2f}")

    print("\nConfusion Matrix:")
    labels = ["Present", "Past", "None", "UNKNOWN"]
    print(" " * 10 + "\t".join(labels))
    for g in labels:
        row = "\t".join(str(metrics["confusion_matrix"][g][p]) for p in labels)
        print(f"{g:10s}\t{row}")

    print("\nPer-Class Metrics:")
    for lbl, vals in metrics["per_class"].items():
        print(f"{lbl:10s} | "
            f"P={vals['precision']:.2f}  "
            f"R={vals['recall']:.2f}  "
            f"F1={vals['f1']:.2f}  "
            f"Support={vals['support']}")

    print(f"\nResults saved to {out_path}")



# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="outputs/alcohol_eval")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.train_path, args.test_path, args.results_dir)
