"""LoRA SFT (used for both 'SFT on CoT-free customer data' and 'distillation from short CoT'). Dose = number of steps."""
from __future__ import annotations

import json
from pathlib import Path

from ..registry import save_manifest
from ..schema import RunManifest
from .data import data_sha


def run_sft(run_id: str, base_model: str, rows: list[dict], steps: int, intervention: str = "sft", dose=None, seed: int = 0,
            lora_r: int = 16, lr: float = 1e-4, batch: int = 4, max_len: int = 1024, out_root: Path | None = None, device: str = "auto",
            extra: dict | None = None, grad_accum: int = 1, qlora: bool = False) -> Path:
    """device='auto' picks cuda if present. Precision: bf16 if the GPU supports it, else fp16 (T4 / old cards); fp32 on CPU.
    qlora=True loads the base in 4-bit NF4 (bitsandbytes, CUDA only). CUDA/QLoRA paths are UNTESTED in the CPU dev container."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from ..registry import run_dir

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device == "cuda"
    bf16 = cuda and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16
    mik: dict = {}
    if qlora:
        if not cuda:
            raise ValueError("qlora needs CUDA")
        from transformers import BitsAndBytesConfig

        mik["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        mik["device_map"] = {"": 0}
    elif cuda:
        mik["torch_dtype"] = torch.bfloat16 if bf16 else torch.float32   # fp16 AMP needs fp32 master weights for the LoRA params
    out = run_dir(run_id, out_root) / "adapter"
    tok = AutoTokenizer.from_pretrained(base_model)
    cfg = SFTConfig(output_dir=str(out), max_steps=steps, per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum,
                    learning_rate=lr, seed=seed, logging_steps=max(1, steps // 10), save_strategy="no", report_to=[], max_length=max_len,
                    use_cpu=not cuda, bf16=bf16, fp16=fp16, assistant_only_loss=False, gradient_checkpointing=cuda,
                    model_init_kwargs=mik or None)
    lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=base_model, args=cfg, train_dataset=Dataset.from_list(rows), peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(str(out))
    save_manifest(RunManifest(run_id=run_id, base_model=base_model, intervention=intervention, dose=dose if dose is not None else steps,
                              seed=seed, adapter_path=str(out), data_sha=data_sha(rows), extra={"lora_r": lora_r, "lr": lr, **(extra or {})}), out_root)
    (out.parent / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    return out
