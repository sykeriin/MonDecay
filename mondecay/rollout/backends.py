"""Generation backends behind one tiny interface.

HFBackend      : transformers (+ optional LoRA adapter) - works on CPU, used for smoke tests and small runs.
VLLMBackend    : vLLM (+ optional LoRA) - the GPU path for real runs. NOT exercised in the CPU dev container.
ScriptedBackend: deterministic fake used by unit tests / pipeline smoke tests (no model download).
"""
from __future__ import annotations

from typing import Callable, Protocol


class Backend(Protocol):
    name: str

    def generate(self, messages: list[dict], max_new_tokens: int = 1024, temperature: float = 0.6, seed: int | None = None) -> str: ...


class HFBackend:
    def __init__(self, model: str, adapter: str | None = None, dtype: str = "float32", device: str = "cpu", enable_thinking: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model}" + (f"+{adapter}" if adapter else "")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, dtype)).to(device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.device = device
        self.enable_thinking = enable_thinking

    def _prompt(self, messages):
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, messages, max_new_tokens=1024, temperature=0.6, seed=None):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        ids = self.tok(self._prompt(messages), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-5), top_p=0.95, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        # R1-distill templates put the opening <think> in the prompt; re-add so parsing is uniform.
        if "</think>" in text and "<think>" not in text:
            text = "<think>" + text
        return text

    def generate_with_prefix(self, messages, prefix: str, max_new_tokens=64) -> str:
        """Greedy continuation after a forced assistant prefix (used by the load-bearing truncation test)."""
        import torch

        ids = self.tok(self._prompt(messages) + prefix, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def next_token_probs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        """Probability mass on each candidate first-token (used by the monitor for P(YES))."""
        import torch

        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        probs = torch.softmax(logits.float(), -1)
        out = {}
        for c in candidates:
            tid = self.tok.encode(c, add_special_tokens=False)[0]
            out[c] = float(probs[tid])
        return out


class VLLMBackend:
    """GPU path. Written to the vLLM API but untested in the CPU dev container - validate on first GPU run."""

    def __init__(self, model: str, adapter: str | None = None, max_model_len: int = 8192, gpu_mem: float = 0.85, enable_thinking: bool = True):
        from vllm import LLM

        self.name = f"vllm:{model}" + (f"+{adapter}" if adapter else "")
        self.llm = LLM(model=model, enable_lora=bool(adapter), max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       max_lora_rank=64 if adapter else None)
        self.adapter = adapter
        self.enable_thinking = enable_thinking

    def _lora(self):
        if not self.adapter:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("a", 1, self.adapter)

    def generate_batch(self, batch: list[list[dict]], max_new_tokens=2048, temperature=0.6, seed=None) -> list[str]:
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=0.95, seed=seed)
        outs = self.llm.chat(batch, sp, lora_request=self._lora(), chat_template_kwargs={"enable_thinking": self.enable_thinking})
        res = []
        for o in outs:
            t = o.outputs[0].text
            res.append("<think>" + t if "</think>" in t and "<think>" not in t else t)
        return res

    def generate(self, messages, max_new_tokens=2048, temperature=0.6, seed=None):
        return self.generate_batch([messages], max_new_tokens, temperature, seed)[0]

    def next_token_probs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        import math

        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
        o = self.llm.generate([prompt], sp)[0].outputs[0].logprobs[0]
        by_text = {v.decoded_token.strip(): math.exp(v.logprob) for v in o.values()}
        return {c: by_text.get(c, 0.0) for c in candidates}


class ScriptedBackend:
    """Fake model: `fn(messages) -> text`. Used in tests and the no-model smoke pipeline."""

    def __init__(self, fn: Callable[[list[dict]], str], name: str = "scripted"):
        self.fn, self.name = fn, name

    def generate(self, messages, max_new_tokens=1024, temperature=0.6, seed=None):
        return self.fn(messages)

    def generate_with_prefix(self, messages, prefix, max_new_tokens=64):
        return self.fn(messages + [{"role": "assistant", "content": prefix}])

    def next_token_probs(self, prompt, candidates):
        p = 0.9 if "TOKEN_YES" in prompt else 0.1
        return {c: (p if c.lower().startswith("y") else 1 - p) for c in candidates}
