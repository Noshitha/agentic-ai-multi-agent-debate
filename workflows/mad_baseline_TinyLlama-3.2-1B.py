import os
import json
import argparse
import random
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_id):
    """Load tokenizer and model for given model_id."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model


def query_model(tokenizer, model, prompt, seed=0, max_new_tokens=128):
    """Query the model with a given prompt and random seed."""
    random.seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output = model.generate(
        input_ids,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        max_new_tokens=max_new_tokens
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


def extract_label(text):
    """
    Extract label from model output.
    Assumes the label is in the first line or explicitly marked in the prompt format.
    """
    first_line = text.strip().split("\n")[0]
    return first_line

import re

def extract_label(text):
    match = re.search(r"Label:\s*(Present|Past|None)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "UNKNOWN"

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


def run_single_agent(model_id, prompt):
    """Run a single agent baseline."""
    tokenizer, model = load_model(model_id)
    response = query_model(tokenizer, model, prompt, seed=0)
    label = extract_label(response)
    return label, response


def run_mad(model_id, prompt, num_agents=3):
    """Run multiple agents (same model) and aggregate with majority vote."""
    tokenizer, model = load_model(model_id)
    predictions = []

    for agent_id in range(num_agents):
        response = query_model(
            tokenizer, model,
            f"Agent {agent_id+1}:\n{prompt}",
            seed=agent_id
        )
        label = extract_label(response)
        predictions.append(label)

    final_label = majority_vote(predictions)
    return final_label, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model ID (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to the initial step prompt file")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of agents to simulate")
    args = parser.parse_args()

    # Load prompt
    with open(args.prompt_file, "r") as f:
        base_prompt = f.read()

    # Run single agent
    single_label, single_response = run_single_agent(args.model_id, base_prompt)

    # Run MAD baseline
    mad_label, mad_predictions = run_mad(args.model_id, base_prompt, args.num_agents)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "mad_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "single_agent": {
                "label": single_label,
                "response": single_response
            },
            "mad_baseline": {
                "final_label": mad_label,
                "agent_predictions": mad_predictions
            }
        }, f, indent=2)

    print("=== Single Agent Baseline ===")
    print("Prediction:", single_label)
    print("Full Response:", single_response[:300], "...\n")

    print("=== MAD Baseline ===")
    print("Final Prediction:", mad_label)
    print("Agent Predictions:", mad_predictions)
    print(f"\nResults saved to {results_path}")