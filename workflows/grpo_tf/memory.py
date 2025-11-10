# grpo_tf/memory.py
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGMemory:
    """
    Simple retrieval-augmented memory for Training-Free GRPO.
    Stores (text, embedding, reward) tuples and retrieves top-k experiences.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.memory = []

    def add(self, text: str, reward: float):
        emb = self.encoder.encode(text, normalize_embeddings=True)
        self.memory.append({"text": text, "embedding": emb, "reward": reward})

    def retrieve(self, query: str, k: int = 3):
        if not self.memory:
            return []
        q_emb = self.encoder.encode(query, normalize_embeddings=True)
        sims = np.array([
            np.dot(q_emb, m["embedding"]) for m in self.memory
        ])
        topk_idx = sims.argsort()[-k:][::-1]
        return [self.memory[i]["text"] for i in topk_idx]
