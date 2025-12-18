# grpo_tf/llm_vllm.py
import os
from vllm import LLM, SamplingParams

class ZeroShotPolicyVLLM:
    def __init__(self, model_path: str):
        tp = int(os.getenv("VLLM_TP", "1"))

        self.llm = LLM(
            model=model_path,
            dtype="float16",
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.80,   # ✅ prevents startup OOM
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:

        sampling = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_new_tokens,
        )

        outputs = self.llm.generate(prompt, sampling)

        # vLLM returns a batch → take first completion
        return outputs[0].outputs[0].text.strip()
