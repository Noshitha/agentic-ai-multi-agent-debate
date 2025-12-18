# grpo_tf/memory.py
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGMemory:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", reward_threshold=0.5):
        self.encoder = SentenceTransformer(model_name)
        self.memory = []
        self.reward_threshold = reward_threshold

    def add_experiences(self, texts, reward=1.0):
        """
        texts: list of summarized experience strings
        """
        added = 0
        for text in texts:
            if any(text.strip() == m["text"].strip() for m in self.memory):
                continue

            emb = self.encoder.encode(text, normalize_embeddings=True)
            self.memory.append({
                "text": text,
                "embedding": emb,
                "reward": reward
            })
            added += 1
        return added

    def retrieve(self, query, k=3):
        if not self.memory:
            return []

        q_emb = self.encoder.encode(query, normalize_embeddings=True)
        sims = [np.dot(q_emb, m["embedding"]) for m in self.memory]
        topk = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self.memory[i]["text"] for i in topk]
