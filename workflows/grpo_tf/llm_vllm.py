import os
from vllm import LLM, SamplingParams

class ZeroShotPolicyVLLM:
    def __init__(self, model_path: str):
        tp = int(os.getenv("VLLM_TP", "1"))
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            #tensor_parallel_size=1  # or torch.cuda.device_count(),
            tensor_parallel_size=tp
        )

    def generate(self, prompt: str,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7):

        sampling = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_new_tokens
        )

        outputs = self.llm.generate(prompt, sampling)
        return outputs[0].outputs[0].text.strip()
