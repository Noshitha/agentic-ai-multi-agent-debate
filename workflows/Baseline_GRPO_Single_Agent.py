import os, json, argparse, random, re
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, torch
from tqdm import tqdm


# =====================================================
# LOAD MODEL
# =====================================================
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda").eval()
    model.config.use_cache = False
    print(">>> Model loaded on:", next(model.parameters()).device)
    return tokenizer, model


# =====================================================
# SIMPLE REWARD FUNCTIONS
# =====================================================
def extract_label(output_text):
    match = re.search(r"Label\s*[:\-]*\s*(Present|Past|None)", output_text, re.IGNORECASE)
    return match.group(1).capitalize() if match else "UNKNOWN"

def reward_from_gold(pred_label, gold_label, unknown_reward=0.2):
    pl, gl = pred_label.lower(), gold_label.lower()
    if pl == gl: return 1.0
    if pl == "unknown": return unknown_reward
    return 0.0

def compute_rewards(candidate_texts, gold_label):
    labels = [extract_label(t) for t in candidate_texts]
    rewards = [reward_from_gold(lbl, gold_label) for lbl in labels]
    return labels, rewards

def compute_advantages(rewards):
    mean_r = sum(rewards)/len(rewards) if rewards else 0.0
    return [r - mean_r for r in rewards]


# =====================================================
# RETRIEVAL-AUGMENTED MEMORY
# =====================================================
class RAGMemory:
    def __init__(self, embedder_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_name)
        test_vec = self.embedder.encode(["probe"], normalize_embeddings=True)
        dim = test_vec.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.memory = []
        self.count = 0

    def add(self, text):
        emb = self.embedder.encode([text], normalize_embeddings=True).astype(np.float32)
        self.index.add(emb)
        self.memory.append(text)
        self.count += 1

    def retrieve(self, query, k=3):
        if self.count == 0:
            return []
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(q_emb, k)
        return [self.memory[i] for i in I[0] if 0 <= i < self.count]


# =====================================================
# POLICY QUERY
# =====================================================
def query_single_agent(tokenizer, model, instruction, text, retrieved_knowledge=None,
                       g=4, max_new_tokens=96, top_p=0.9, temperature=0.7):
    """Generate G candidate answers for the same query with retrieved context."""
    seed = 42
    random.seed(seed)

    retrieval_block = ""
    if retrieved_knowledge:
        retrieval_block = "Retrieved Knowledge:\n" + "\n".join(
            f"- {rk}" for rk in retrieved_knowledge
        ) + "\n\n"

    agent_prompt = (
        f"{retrieval_block}"
        f"{instruction}\n\n"
        f"Text:\n{text}\n\n"
        "You are an expert clinician. Follow this strict format:\n"
        "Reasoning: <one or two sentences explaining your decision>\n"
        "Label: <Present | Past | None>\n"
    )

    inputs = tokenizer(agent_prompt, return_tensors="pt").to("cuda")
    decoded = []
    for _ in range(g):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded.append(tokenizer.decode(out[0], skip_special_tokens=True))
        torch.cuda.empty_cache()
    # return decoded, agent_prompt
    # outputs = model.generate(
    #     **inputs,
    #     do_sample=True,
    #     top_p=top_p,
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    #     num_return_sequences=g,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    return decoded, agent_prompt


# =====================================================
# MAIN EVALUATION LOOP (RAG-Enhanced GRPO)
# =====================================================
def evaluate_rag_grpo_single_agent(model_path, test_path, results_dir="outputs/rag_grpo_eval",
                                   G=4, top_p=0.9, temperature=0.7, unknown_reward=0.2, k_retrieve=3):

    os.makedirs(results_dir, exist_ok=True)
    tokenizer, model = load_model(model_path)
    rag_memory = RAGMemory()

    test_data = [json.loads(line) for line in open(test_path)]
    results, audit = [], []

    for sample in tqdm(test_data, desc="RAG-Enhanced Training-Free GRPO"):
        instruction = sample["instruction"]
        text = sample["input"]
        gold = sample["output"]

        # Retrieve relevant prior samples
        retrieved = rag_memory.retrieve(text, k=k_retrieve)

        # Generate multiple candidates
        candidates, prompt = query_single_agent(
            tokenizer, model, instruction, text,
            retrieved_knowledge=retrieved,
            g=G, top_p=top_p, temperature=temperature
        )

        # Compute rewards & advantages
        labels, rewards = compute_rewards(candidates, gold)
        advs = compute_advantages(rewards)

        # Pick best candidate
        best_idx = max(range(len(rewards)), key=lambda i: (rewards[i], advs[i]))
        best_output = candidates[best_idx]
        best_label = labels[best_idx]

        #  Add new experience to RAG memory (input + best_output + reasoning)
        rag_memory.add(f"Text: {text}\nBestOutput: {best_output}\nLabel: {best_label}")

        # Log
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
            "retrieved": retrieved,
            "best_idx": best_idx
        })

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Compute metrics
    accuracy = sum(1 for r in results if r["match"]) / len(results)
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
    with open(os.path.join(results_dir, "audit.jsonl"), "w") as f:
        for row in audit:
            f.write(json.dumps(row) + "\n")

    print(f"\n RAG-Enhanced GRPO Complete — Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to {results_dir}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="outputs/rag_grpo_eval")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--unknown_reward", type=float, default=0.2)
    parser.add_argument("--k_retrieve", type=int, default=3)
    args = parser.parse_args()

    evaluate_rag_grpo_single_agent(
        model_path=args.model_path,
        test_path=args.test_path,
        results_dir=args.results_dir,
        G=args.num_candidates,
        top_p=args.top_p,
        temperature=args.temperature,
        unknown_reward=args.unknown_reward,
        k_retrieve=args.k_retrieve
    )
