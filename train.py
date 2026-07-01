#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL-2B for handwritten Arabic OCR ( LoRA + conversational SFT).

Enhancements vs. the paper:
- Real handwriting: KHATT (KFUPM) via Hugging Face.
- Handwriting-focused augmentations (noise, blur, illumination, mild elastic warp, JPEG).
- Prompts that avoid inventing diacritics when absent in the image.

Requires: CUDA GPU strongly recommended (24GB+ comfortable for 2B + LoRA at 1280px cap).

Quickstart:
  pip install -r requirements.txt
  # Optional dry run (few lines, still downloads ~4GB base model on first run):
  python train.py --max_samples 64 --epochs 0.01 --save_steps 50 --output_dir ./outputs/smoke
  python inference.py --image path/to/crop.png --adapter ./outputs/handwriting_arabic_lora

Data: KHATT via Hugging Face (academic use; accept terms on the dataset card if prompted).
"""

from __future__ import annotations

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from preprocess.config import HandwritingOCRConfig
from data.khatt import build_handwriting_dataset
from data.mixed import build_mixed_dataset
from data.coru import build_handwriting_dataset_coru
from OCR_Training.collator import make_qwen2_vl_collator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train handwriting Arabic OCR LoRA on Qwen2-VL-2B")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="Cap rows for a dry run")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--augment_prob", type=float, default=None, help="Probability of applying aug pipeline")
    p.add_argument("--save_steps", type=int, default=None, help="Checkpoint every N optimizer steps")
    p.add_argument("--logging_steps", type=int, default=None, help="Log every N steps")
    p.add_argument("--save_total_limit", type=int, default=None, help="Max checkpoints to keep on disk")
    p.add_argument("--ifn_enit_root", type=str, default=None,
                   help="Path to IFN/ENIT dataset root (e.g. ./data/ifn_enit). Enables mixed training.")
    p.add_argument("--ifn_enit_split", type=str, default=None, help="IFN/ENIT split: train/val/test")
    p.add_argument("--ifn_enit_max_samples", type=int, default=None, help="Cap IFN/ENIT rows")
    p.add_argument("--arabic_digits_root", type=str, default=None,
                   help="Path to Arabic digits dataset root (e.g. ./data/arabic_digits). "
                        "Same on-disk layout as IFN/ENIT.")
    p.add_argument("--arabic_digits_split", type=str, default=None,
                   help="Arabic digits split: train/val/test")
    p.add_argument("--arabic_digits_max_samples", type=int, default=None,
                   help="Cap Arabic digits rows")
    p.add_argument("--no_khatt", action="store_true",
                   help="Skip KHATT entirely (no HuggingFace download). "
                        "Use with --arabic_digits_root and/or --ifn_enit_root.")
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA (Linux/CUDA + bitsandbytes)")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a Trainer checkpoint dir (e.g. ./outputs/smoke_linux/checkpoint-79). "
            "Loads LoRA + optimizer from there and continues training. "
            "Increase --epochs to a larger *total* if the previous run already reached max steps."
        ),
    )
    p.add_argument(
        "--init_adapter",
        type=str,
        default=None,
        help=(
            "Path to an existing LoRA adapter dir (e.g. ./outputs/my_handwritting_lora). "
            "Loads those weights as the starting point but begins a FRESH trainer run "
            "(new optimizer, new schedule) and writes new checkpoints to --output_dir. "
            "Use this to fine-tune a previously trained adapter on a new dataset."
        ),
    )
    p.add_argument("--max_steps", type=int, default=-1, help="Total training steps. Overrides epochs if set.")


    # === CORU OCR Dataset ===
    p.add_argument("--coru_root", type=str, default=None,
                   help="Path to CORU dataset base (e.g., ./data/CORU). "
                        "Expects structure: CORU/OCR/split/split/")
    p.add_argument("--coru_split", type=str, default=None,
                   help="CORU split: train/val/test")
    p.add_argument("--coru_max_samples", type=int, default=None,
                   help="Cap CORU rows")
    p.add_argument("--use_coru", action="store_true",
                   help="Enable CORU dataset loading")



    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HandwritingOCRConfig()
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.model_id is not None:
        cfg.model_id = args.model_id
    if args.dataset_name is not None:
        cfg.dataset_name = args.dataset_name
    if args.dataset_split is not None:
        cfg.dataset_split = args.dataset_split
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.per_device_train_batch_size is not None:
        cfg.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.augment_prob is not None:
        cfg.augment_prob = args.augment_prob
    if args.save_steps is not None:
        cfg.save_steps = args.save_steps
    if args.logging_steps is not None:
        cfg.logging_steps = args.logging_steps
    if args.save_total_limit is not None:
        cfg.save_total_limit = args.save_total_limit
    if args.ifn_enit_root is not None:
        cfg.ifn_enit_root = args.ifn_enit_root
    if args.ifn_enit_split is not None:
        cfg.ifn_enit_split = args.ifn_enit_split
    if args.ifn_enit_max_samples is not None:
        cfg.ifn_enit_max_samples = args.ifn_enit_max_samples
    if args.arabic_digits_root is not None:
        cfg.arabic_digits_root = args.arabic_digits_root
    if args.arabic_digits_split is not None:
        cfg.arabic_digits_split = args.arabic_digits_split
    if args.arabic_digits_max_samples is not None:
        cfg.arabic_digits_max_samples = args.arabic_digits_max_samples
    if args.no_khatt:
        cfg.use_khatt = False
    if args.load_in_4bit:
        cfg.load_in_4bit = True
    if args.no_bf16:
        cfg.bf16 = False

    # ... existing arg handling ...

    # CORU args
    if args.coru_root is not None:
        cfg.coru_root = args.coru_root
    if args.coru_split is not None:
        cfg.coru_split = args.coru_split
    if args.coru_max_samples is not None:
        cfg.coru_max_samples = args.coru_max_samples
    if args.use_coru:
        cfg.use_coru = True
    if args.save_total_limit is not None:
        cfg.save_total_limit = args.save_total_limit
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps



    os.makedirs(cfg.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        print(
            f"CUDA: available — device 0 = {torch.cuda.get_device_name(0)} "
            f"(capability {torch.cuda.get_device_capability(0)})"
        )
    else:
        print(
            "CUDA: NOT available — training will use CPU (very slow). "
            "On WSL, install an NVIDIA driver that supports WSL2 and a GPU-enabled "
            "`pip install torch` wheel (see pytorch.org)."
        )

    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        trust_remote_code=cfg.trust_remote_code,
    )

    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map="auto" if quant_config else None,
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    resume_ckpt = args.resume_from_checkpoint
    init_adapter = args.init_adapter
    if resume_ckpt:
        resume_ckpt = os.path.normpath(resume_ckpt)
        if not os.path.isdir(resume_ckpt):
            raise FileNotFoundError(f"--resume_from_checkpoint not a directory: {resume_ckpt}")
        if not os.path.isfile(os.path.join(resume_ckpt, "adapter_config.json")):
            raise FileNotFoundError(
                f"No adapter_config.json in {resume_ckpt} — pass a Trainer checkpoint folder "
                "(e.g. .../checkpoint-79), not only --output_dir."
            )
        print(f"Loading LoRA from checkpoint: {resume_ckpt}")
        model = PeftModel.from_pretrained(model, resume_ckpt, is_trainable=True)
    elif init_adapter:
        init_adapter = os.path.normpath(init_adapter)
        if not os.path.isdir(init_adapter):
            raise FileNotFoundError(f"--init_adapter not a directory: {init_adapter}")
        if not os.path.isfile(os.path.join(init_adapter, "adapter_config.json")):
            raise FileNotFoundError(
                f"No adapter_config.json in {init_adapter} — pass a saved LoRA adapter folder."
            )
        print(f"Initializing LoRA from existing adapter (fresh trainer state): {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    elif os.path.exists(os.path.join(cfg.output_dir, "adapter_config.json")):
        print("Loading existing LoRA adapter...")
        model = PeftModel.from_pretrained(model, cfg.output_dir, is_trainable=True)
    else:
        print("Creating new LoRA adapter...")
        model = get_peft_model(model, lora)

    model.train()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Without 4-bit, from_pretrained(..., device_map=None) leaves weights on CPU.
    # Trainer does not always move the full VL model reliably; force GPU when available.
    if cfg.load_in_4bit:
        print(f"Model device (4-bit): {next(model.parameters()).device}")
    elif torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Model moved to GPU: {next(model.parameters()).device}")
    else:
        print(f"Model staying on CPU: {next(model.parameters()).device}")

    use_mixed_builder = (
        bool(cfg.ifn_enit_root)
        or bool(cfg.arabic_digits_root)
        or not cfg.use_khatt
        or cfg.use_coru #coru
    )
    if use_mixed_builder:
        train_ds = build_mixed_dataset(
            khatt_dataset_name=cfg.dataset_name,
            khatt_split=cfg.dataset_split,
            khatt_text_column=cfg.text_column,
            khatt_image_column=cfg.image_column,
            ifn_enit_root=cfg.ifn_enit_root,
            ifn_enit_split=cfg.ifn_enit_split,
            seed=cfg.seed,
            augment_prob=cfg.augment_prob,
            khatt_max_samples=cfg.max_samples,
            ifn_enit_max_samples=cfg.ifn_enit_max_samples,
            use_khatt=cfg.use_khatt,
            arabic_digits_root=cfg.arabic_digits_root,
            arabic_digits_split=cfg.arabic_digits_split,
            arabic_digits_max_samples=cfg.arabic_digits_max_samples,
            # CORU <-- ADD THIS BLOCK
            coru_root=cfg.coru_root,
            coru_split=cfg.coru_split,
            coru_max_samples=cfg.coru_max_samples,
            use_coru=cfg.use_coru,
        )
    else:
        train_ds = build_handwriting_dataset(
            dataset_name=cfg.dataset_name,
            split=cfg.dataset_split,
            text_column=cfg.text_column,
            image_column=cfg.image_column,
            seed=cfg.seed,
            augment_prob=cfg.augment_prob,
            max_samples=cfg.max_samples,
        )

    use_bf16 = bool(cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(not use_bf16 and torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,  # <--- ADD THIS LINE
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type="linear",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=make_qwen2_vl_collator(processor),
        processing_class=processor,
    )
    if resume_ckpt:
        print(f"Resuming Trainer state from: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"Saved adapter and processor to {cfg.output_dir}")


if __name__ == "__main__":
    main()