import os
import json
import argparse
import random
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re


# ----------------------------
# MODEL LOADING
# ----------------------------
def load_model(model_id):
    """Load tokenizer and model for a given model_id."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return tokenizer, model


# ----------------------------
# AGENTIC QUERY
# ----------------------------
def query_agent(tokenizer, model, agent_id, prompt, max_new_tokens=256):
    seed = 42 + agent_id
    random.seed(seed)

    agent_prompt = (
        f"[Agent {agent_id+1} — Reasoning Mode]\n"
        f"Think step-by-step before deciding the label.\n"
        f"{prompt}\n\nOutput your reasoning and then final label as:\n"
        f"Reasoning: ...\nLabel: ..."
    )

    inputs = tokenizer(agent_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=max_new_tokens
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# ----------------------------
# LABEL EXTRACTION
# ----------------------------
def extract_label(text):
    """
    Extract the label (Present, Past, None) from model output.
    Works for both plain and JSON-formatted responses.
    """
    # Match "Label: Present" or "\"Label\": \"Present\""
    match = re.search(r'["\s]*Label["\s]*[:\-]*\s*["]*(Present|Past|None)', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # fallback to last 'Label:' line
    lines = [l for l in text.split("\n") if "Label" in l]
    if lines:
        return lines[-1].split(":")[-1].strip().capitalize()

    return "UNKNOWN"


# ----------------------------
# MAJORITY VOTE
# ----------------------------
def majority_vote(predictions):
    """Aggregate predictions using majority vote."""
    counts = Counter(predictions)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    elif len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "UNCERTAIN"  # tie
    else:
        return most_common[0][0]



def run_agentic(model_path, prompt, num_agents=3, results_dir="outputs"):
    tokenizer, model = load_model(model_path)
    os.makedirs(results_dir, exist_ok=True)

    agent_outputs, labels = [], []
    for i in range(num_agents):
        out = query_agent(tokenizer, model, i, prompt)
        lbl = extract_label(out)
        agent_outputs.append({"agent": i+1, "output": out, "label": lbl})
        labels.append(lbl)

    final_label = majority_vote(labels)

    results = {
        "model": model_path,
        "final_label": final_label,
        "agents": agent_outputs,
    }
    with open(os.path.join(results_dir, "qwen3_0.6B_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
# ----------------------------
# MAIN SCRIPT
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="Local or Hugging Face model ID (e.g., /path/to/Qwen3-0.6B)")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to the prompt text file")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of agents to simulate")
    args = parser.parse_args()

    # ----------------------------
    # Load prompt
    # ----------------------------
    with open(args.prompt_file, "r") as f:
        base_prompt = f.read().strip()

    # ----------------------------
    # Load model
    # ----------------------------
    tokenizer, model = load_model(args.model_id)
    os.makedirs(args.results_dir, exist_ok=True)

    # ----------------------------
    # SINGLE AGENT BASELINE
    # ----------------------------
    print("\n=== Running Single-Agent Baseline ===")
    single_output = query_agent(tokenizer, model, 0, base_prompt)
    single_label = extract_label(single_output)

    print(f"\nSingle-Agent Prediction: {single_label}")
    print(f"Response (first 300 chars):\n{single_output[:300]}...\n")

    # ----------------------------
    # MULTI-AGENT BASELINE (MAD)
    # ----------------------------
    print("=== Running Multi-Agent (3-Agent) Baseline ===")
    agent_outputs, labels = [], []
    for i in range(args.num_agents):
        out = query_agent(tokenizer, model, i, base_prompt)
        lbl = extract_label(out)
        agent_outputs.append({"agent": i+1, "output": out, "label": lbl})
        labels.append(lbl)

    final_label = majority_vote(labels)

    print(f"\nFinal MAD Label: {final_label}")
    for a in agent_outputs:
        print(f"Agent {a['agent']} → {a['label']}")

    # ----------------------------
    # SAVE RESULTS
    # ----------------------------
    results = {
        "model": args.model_id,
        "single_agent": {
            "label": single_label,
            "output": single_output
        },
        "mad_baseline": {
            "final_label": final_label,
            "agents": agent_outputs
        }
    }

    results_path = os.path.join(args.results_dir, "qwen3_0.6B_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
