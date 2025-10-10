import os, json, random, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import torch


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return tokenizer, model


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


def extract_label(output_text):
    for line in output_text.split("\n"):
        if "Label" in line:
            return line.split(":")[-1].strip()
    return "UNKNOWN"


def majority_vote(preds):
    counts = Counter(preds)
    most_common = counts.most_common()
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    return "UNCERTAIN"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="outputs")
    args = parser.parse_args()

    prompt_text = open(args.prompt_file).read()
    run_agentic(args.model_path, prompt_text, args.num_agents, args.results_dir)
