# grpo_tf/memory.py
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGMemory:
    """
    Reward-aware Retrieval-Augmented Memory for Training-Free GRPO.
    - Stores (text, embedding, reward)
    - Filters low-reward experiences
    - Avoids duplicates
    - Retrieves top-k relevant high-quality memories
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", reward_threshold=0.5):
        self.encoder = SentenceTransformer(model_name)
        self.memory = []
        self.reward_threshold = reward_threshold

    def add(self, text: str, reward: float):
        """Add new experience if reward is good and not duplicate."""
        if reward < self.reward_threshold:
            return  # discard low-quality experiences

        # prevent near-duplicates
        if any(text.strip() == m["text"].strip() for m in self.memory):
            return

        emb = self.encoder.encode(text, normalize_embeddings=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        self.memory.append({"text": text, "embedding": emb, "reward": reward})

    def retrieve(self, query: str, k: int = 3):
        """Retrieve top-k experiences filtered by reward."""
        if not self.memory:
            return []

        # keep only good experiences
        high_quality = [m for m in self.memory if m["reward"] >= self.reward_threshold]
        if not high_quality:
            return []

        q_emb = self.encoder.encode(query, normalize_embeddings=True)
        sims = np.array([np.dot(q_emb, m["embedding"]) for m in high_quality])

        # top-k similarity indices
        topk_idx = sims.argsort()[-k:][::-1]
        return [high_quality[i]["text"] for i in topk_idx]
