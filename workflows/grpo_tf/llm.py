# grpo_tf/llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, random, re

class ZeroShotPolicy:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True
        ).to("cuda")
        print(">>> Model loaded on:", next(self.model.parameters()).device)

    def generate(self, prompt: str,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """Ask model to reason step-by-step and produce a single label."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, do_sample=True, top_p=0.9,
            temperature=temperature, max_new_tokens=max_new_tokens
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # remove the prompt portion so only new text remains
        if prompt in output_text:
            output_text = output_text[len(prompt):].strip()
        return output_text

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
