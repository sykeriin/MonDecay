"""Post-training weight quantisation.

`rtn_quantize` = round-to-nearest, per-group, symmetric, 'fake quant' (weights stay in fp but take only 2^bits levels).
It reproduces the weight-error of int8/int4 PTQ on any hardware (incl. CPU) but NOT kernel-level behaviour.
For headline GPU numbers prefer AWQ/GPTQ checkpoints; RTN is the reproducible, dependency-free baseline.
"""
from __future__ import annotations

from pathlib import Path

from ..registry import run_dir, save_manifest
from ..schema import RunManifest


def rtn_quantize_tensor(w, bits: int, group_size: int = 128):
    import torch

    shape = w.shape
    g = group_size if w.shape[-1] % group_size == 0 else w.shape[-1]
    x = w.reshape(-1, g).float()
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return (q * scale).reshape(shape).to(w.dtype)


def run_quantize(run_id: str, base_model: str, bits: int, group_size: int = 128, adapter: str | None = None, out_root: Path | None = None,
                 seed: int = 0) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                mod.weight.copy_(rtn_quantize_tensor(mod.weight.data, bits, group_size))
    out = run_dir(run_id, out_root) / "model"
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out)
    save_manifest(RunManifest(run_id=run_id, base_model=str(out), intervention="quant", dose=bits, seed=seed, quant=f"rtn-int{bits}-g{group_size}",
                              extra={"source_model": base_model}), out_root)
    return out
