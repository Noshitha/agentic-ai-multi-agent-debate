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
        torch_dtype=torch.bfloat16,
        device_map=None,   
        trust_remote_code=True
    )
    model.to("cuda").eval()
    print(">>> Model loaded on:", next(model.parameters()).device)
    return tokenizer, model

# ----------------------------
# QUERY MODEL TRAINING FREE GRPO - Multiple candidate outputs and experience
# ----------------------------
def query_single_agent(tokenizer, model, instruction, text, experience =None, g=4, max_new_tokens=256, top_p=0.9, temperature=0.7):
    """Generate G candidate answers fro the same query (training free GRPO.)"""
    seed = 42
    random.seed(seed)
    memory_block = ""
    if experience:
        memory_block = "Prior Learning:\n" + "\n".join("- " + str(e) for e in experience) + "\n\n"

    agent_prompt = (
        f"{memory_block}"
        f"{instruction}\n\n"
        f"Text:\n{text}\n\n"
        "You are an expert clinician. Follow this strict format:\n"
        "Reasoning: <one or two sentences explaining your decision>\n"
        "Label: <Present | Past | None>\n"
    )

    # inputs = tokenizer(agent_prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(agent_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        do_sample=True, 
        top_p=top_p, 
        temperature=temperature, 
        max_new_tokens=max_new_tokens,
        num_return_sequences=g,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return decoded, agent_prompt


# ----------------------------
# EXTRACT LABEL
# ----------------------------
def extract_label(output_text):
    match = re.search(r"Label\s*[:\-]*\s*(Present|Past|None)", output_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "UNKNOWN"

# ----------------------------
# Supervised reward function
# ----------------------------
def reward_from_gold(pred_label, gold_label, unknown_reward=0.2):
    pl, gl = pred_label.lower(), gold_label.lower()
    if pl == gl:        return 1.0
    if pl == "unknown": return unknown_reward
    return 0.0

def compute_rewards(candidate_texts, gold_label):
    labels = [extract_label(t) for t in candidate_texts]
    rewards = [reward_from_gold(lbl, gold_label) for lbl in labels]
    return labels, rewards

# ----------------------------
# Compute group relative advantages
# ----------------------------
def compute_advantages(rewards):
    mean_r = sum(rewards)/len(rewards) if rewards else 0.0
    return [r - mean_r for r in rewards]

# ----------------------------
# Extract experience
# ----------------------------
def extract_experience(tokenizer, model, best_output, other_outputs, max_new_tokens=128):
    prompt = (
        "You coach a medical classifier (alcohol use: Present/Past/None).\n"
        "Given the BEST and OTHER answers, write 1–2 short rules explaining "
        "why the best is correct (negations like 'denies', present-tense cues like 'drinks', "
        "explicit absence statements). Keep it concise and reusable.\n\n"
        f"BEST:\n{best_output}\n\n"
        "OTHERS:\n" + "\n---\n".join(other_outputs) + "\n\nRules:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("Rules:",1)[-1].strip()



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
# EVALUATE MODEL - TRAINING FREE GRPO
# ----------------------------
def evaluate_grpo_single_agent(model_path, test_path, results_dir="outputs/grpo_eval/MediPhi-Instruct_eval",
                               G=4, memory_size=3, unknown_reward=0.2,
                               temperature=0.7, top_p=0.9):
    """
    Training-free GRPO-style evaluation:
    Generates G candidates per sample, computes supervised rewards,
    calculates group-relative advantages, extracts experience rules,
    and uses rolling memory for contextual adaptation.
    """
    tokenizer, model = load_model(model_path)

    print(f"\nLoading test data from {test_path}")
    test_data = [json.loads(line) for line in open(test_path)]
    memory = deque(maxlen=memory_size)
    results, audit = [], []

    for sample in tqdm(test_data, desc="Training-free GRPO evaluation"):
        instruction = sample["instruction"]
        text = sample["input"]
        gold = sample["output"]

        # 1) generate multiple candidates
        candidates, prompt = query_single_agent(tokenizer, model, instruction, text,
                                                experience=list(memory),
                                                g=G, top_p=top_p, temperature=temperature)
        # 2) compute labels and rewards
        labels, rewards = compute_rewards(candidates, gold)
        advs = compute_advantages(rewards)

        # 3) select best candidate
        best_idx = max(range(len(rewards)), key=lambda i: (rewards[i], advs[i]))
        best_label = labels[best_idx]
        best_output = candidates[best_idx]

        # 4) extract experience and update memory
        other_outputs = [c for i, c in enumerate(candidates) if i != best_idx]
        exp = None
        if other_outputs:
            exp = extract_experience(tokenizer, model, best_output, other_outputs)
            if exp:
                memory.append(exp)

        # 5) log results
        results.append({
            "input": text,
            "gold": gold,
            "predicted": best_label,
            "match": best_label.lower() == gold.lower()
        })
        audit.append({
            "input": text,
            "gold": gold,
            "candidates": [
                {"output": c, "label": labels[i], "reward": rewards[i], "adv": advs[i]}
                for i, c in enumerate(candidates)
            ],
            "best_idx": best_idx,
            "experience_added": exp
        })

    # 6) compute metrics + save
    metrics = compute_metrics(results)
    with open(os.path.join(results_dir, "eval_metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    with open(os.path.join(results_dir, "audit_traces.jsonl"), "w") as f:
        for row in audit:
            f.write(json.dumps(row) + "\n")

    print("\n==== GRPO Evaluation Complete ====")
    print(f"Accuracy : {metrics['overall']['accuracy']*100:.2f}%")
    print(f"Macro F1 : {metrics['overall']['macro_f1']:.3f}")
    print(f"Results saved to {results_dir}")




# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="outputs/grpo_eval/MediPhi-Instruct_eval")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--memory_size", type=int, default=3)
    parser.add_argument("--unknown_reward", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--mode", type=str, default="grpo", choices=["zero", "grpo"])

    args = parser.parse_args()

    evaluate_grpo_single_agent(
            model_path=args.model_path,
            test_path=args.test_path,
            results_dir=args.results_dir,
            G=args.num_candidates,
            memory_size=args.memory_size,
            unknown_reward=args.unknown_reward,
            temperature=args.temperature,
            top_p=args.top_p
        )
