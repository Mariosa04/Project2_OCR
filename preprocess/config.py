"""Training defaults aligned with QARI-OCR (Qwen2-VL-2B + LoRA) + handwriting scope."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HandwritingOCRConfig:
    # In config.py, find max_pixels and change it to this:
    max_pixels: int = 200704  # Optimized for single-line OCR (256 * 28 * 28)
    # Base model (paper: Qwen2-VL-2B-Instruct)
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str = "./outputs/handwriting_arabic_lora"

    # Image bounds (Qwen2-VL uses dynamic resolution; cap VRAM — tune per GPU)
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28

    # LoRA (paper: rank 16; targets LM + light touch on vision projections)
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkv",
        "proj",
    )

    # Optimization (paper: 1 epoch, AdamW 2e-4, wd 0.01, linear schedule)
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3

    # Data — KHATT (real handwritten Arabic paragraphs, HuggingFace)
    dataset_name: str = "johnlockejrr/KHATT_v1.0_dataset"
    dataset_split: str = "train"
    text_column: str = "text"
    image_column: str = "image"
    max_samples: int | None = None

    # Data — IFN/ENIT (short Arabic words/phrases, local directory)
    ifn_enit_root: str | None = None
    ifn_enit_split: str = "train"
    ifn_enit_max_samples: int | None = None

    # Data — Arabic digit crops (local directory, same on-disk layout as IFN/ENIT)
    arabic_digits_root: str | None = None
    arabic_digits_split: str = "train"
    arabic_digits_max_samples: int | None = None

    # Add these fields to HandwritingOCRConfig class

    # CORU OCR dataset
    coru_root: str = ""  # Path to CORU base (e.g., ./data/CORU)
    coru_split: str = "train"
    coru_max_samples: int | None = None
    use_coru: bool = False

    # Toggle KHATT (set False for digits-only / IFN-only fine-tuning, no HF download)
    use_khatt: bool = True

    # Handwriting-focused augmentation (applied on CPU before batching)
    augment_prob: float = 0.45
    seed: int = 42

    # Quantization: paper used 4-bit; on many Windows setups bnb is painful — default off
    load_in_4bit: bool = False
    bf16: bool = True

    # HF
    trust_remote_code: bool = True


