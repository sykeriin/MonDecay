"""LoRA SFT (used for both 'SFT on CoT-free customer data' and 'distillation from short CoT'). Dose = number of steps."""
from __future__ import annotations

import json
from pathlib import Path

from ..registry import save_manifest
from ..schema import RunManifest
from .data import data_sha


def run_sft(run_id: str, base_model: str, rows: list[dict], steps: int, intervention: str = "sft", dose=None, seed: int = 0,
            lora_r: int = 16, lr: float = 1e-4, batch: int = 4, max_len: int = 1024, out_root: Path | None = None, device: str = "cpu",
            extra: dict | None = None) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from ..registry import run_dir

    out = run_dir(run_id, out_root) / "adapter"
    tok = AutoTokenizer.from_pretrained(base_model)
    cfg = SFTConfig(output_dir=str(out), max_steps=steps, per_device_train_batch_size=batch, learning_rate=lr, seed=seed,
                    logging_steps=max(1, steps // 10), save_strategy="no", report_to=[], max_length=max_len,
                    use_cpu=(device == "cpu"), bf16=(device != "cpu"), assistant_only_loss=False,
                    gradient_checkpointing=(device != "cpu"))
    lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=base_model, args=cfg, train_dataset=Dataset.from_list(rows), peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(str(out))
    save_manifest(RunManifest(run_id=run_id, base_model=base_model, intervention=intervention, dose=dose if dose is not None else steps,
                              seed=seed, adapter_path=str(out), data_sha=data_sha(rows), extra={"lora_r": lora_r, "lr": lr, **(extra or {})}), out_root)
    (out.parent / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    return out
