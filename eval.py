#!/usr/bin/env python3
"""
Evaluate a trained LoRA (or base model) on KHATT, IFN/ENIT, or CORU using corpus CER / WER
(micro-averaged: total edits over total reference length; not the mean of per-line rates).

Run after training, pointing --adapter at the same folder you passed as --output_dir to train.py.

Supports:
  - HuggingFace datasets (KHATT): default, uses --dataset_name / --split
  - Local IFN/ENIT: pass --ifn_enit_root and --ifn_enit_split
  - Local CORU: pass --coru_root and --coru_split
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import jiwer
import torch
from datasets import load_dataset
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from preprocess.prompts import USER_OCR_PROMPT
# --- NEW IMPORT FOR CORU ---
from data.coru import _crop_to_content


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR evaluation: CER & WER on a dataset split")
    p.add_argument("--adapter", type=str, default=None, help="LoRA folder from training (omit for base-only)")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")

    data_g = p.add_argument_group("HuggingFace dataset (default: KHATT)")
    data_g.add_argument("--dataset_name", type=str, default="johnlockejrr/KHATT_v1.0_dataset")
    data_g.add_argument("--split", type=str, default="validation", help="Dataset split, e.g. validation or test")
    data_g.add_argument("--image_column", type=str, default="image")
    data_g.add_argument("--text_column", type=str, default="text")

    ifn_g = p.add_argument_group("Local IFN/ENIT dataset (overrides HF dataset when set)")
    ifn_g.add_argument("--ifn_enit_root", type=str, default=None,
                       help="Path to IFN/ENIT root dir (e.g. ./data/ifn_enit)")
    ifn_g.add_argument("--ifn_enit_split", type=str, default="test",
                       help="IFN/ENIT split: train/val/test")

    # --- NEW ARGUMENT GROUP FOR CORU ---
    coru_g = p.add_argument_group("Local CORU dataset (overrides HF and IFN/ENIT when set)")
    coru_g.add_argument("--coru_root", type=str, default=None,
                        help="Path to CORU base dir (e.g. ./data/CORU). Expects CORU/OCR/split/split/ structure.")
    coru_g.add_argument("--coru_split", type=str, default="test",
                        help="CORU split: train/val/test")

    p.add_argument("--max_samples", type=int, default=None, help="Cap number of examples (faster smoke eval)")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy)")
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument(
        "--preprocess", action="store_true",
        help="Apply grid-removal / binarization preprocessing before OCR",
    )
    p.add_argument(
        "--strip_tatweel",
        action="store_true",
        help="Remove Arabic kashida (U+0640) before scoring",
    )
    p.add_argument(
        "--no_normalize_ws",
        action="store_true",
        help="Do not collapse whitespace after strip (default: normalize spaces)",
    )
    return p.parse_args()


def normalize_text(s: str, *, collapse_ws: bool, strip_tatweel: bool) -> str:
    s = unicodedata.normalize("NFC", (s or "").strip())
    if strip_tatweel:
        s = s.replace("\u0640", "")
    if collapse_ws:
        s = " ".join(s.split())
    return s


def ocr_one_image(
        model,
        processor,
        pil_image: Image.Image,
        device: str,
        max_new_tokens: int,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image.convert("RGB")},
                {"type": "text", "text": USER_OCR_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    if device == "cuda":
        inputs = inputs.to("cuda")

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _load_ifn_enit_pairs(root: str, split: str) -> list[dict[str, str]]:
    """Load image_path + text pairs from the local IFN/ENIT directory."""
    split_dir = Path(root) / split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    pairs = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        stem = lbl.stem
        img_file = img_dir / f"{stem}.png"
        if not img_file.exists():
            img_file = img_dir / f"{stem}.jpg"
        if not img_file.exists():
            continue
        text = lbl.read_text(encoding="utf-8").strip().replace(".", " ").strip()
        if text:
            pairs.append({"image_path": str(img_file), "text": text})
    return pairs


# --- NEW LOADER FUNCTION FOR CORU ---
def _load_coru_pairs(root: str, split: str) -> list[dict[str, str]]:
    """Load image_path + text pairs from the local CORU directory."""
    # CORU structure: root/OCR/split/split/
    data_dir = Path(root) / "OCR" / split / split
    if not data_dir.exists():
        raise FileNotFoundError(f"CORU data directory not found: {data_dir}")

    pairs = []
    valid_exts = {".jpg", ".jpeg", ".png"}

    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() not in valid_exts:
            continue

        lbl_path = data_dir / f"{f.stem}.txt"
        if not lbl_path.exists():
            continue

        text = lbl_path.read_text(encoding="utf-8").strip()

        # Parse CORU specific label format: '["extracted text"]'
        if text.startswith('["') and text.endswith('"]'):
            text = text[2:-2].strip()

        if text:
            pairs.append({"image_path": str(f), "text": text})

    return pairs


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: evaluation on CPU is very slow.", file=sys.stderr)

    collapse_ws = not args.no_normalize_ws
    use_ifn = bool(args.ifn_enit_root)
    use_coru = bool(args.coru_root)  # --- NEW FLAG ---

    preprocess_fn = None
    if args.preprocess:
        from preprocess.preprocessing import preprocess_pil
        preprocess_fn = preprocess_pil

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Build evaluation list: [(pil_image, reference_text), ...]
    eval_items: list[tuple[Image.Image, str]] = []
    dataset_label = ""

    # --- NEW CORU LOADING BRANCH ---
    if use_coru:
        dataset_label = f"CORU ({args.coru_root})  split: {args.coru_split}"
        pairs = _load_coru_pairs(args.coru_root, args.coru_split)
        if args.max_samples is not None:
            pairs = pairs[: min(args.max_samples, len(pairs))]
        for p in pairs:
            pil = Image.open(p["image_path"]).convert("RGB")
            # CRITICAL: Crop out the massive white padding in CORU images
            pil, _, _ = _crop_to_content(pil)
            eval_items.append((pil, p["text"]))

    elif use_ifn:
        dataset_label = f"IFN/ENIT ({args.ifn_enit_root})  split: {args.ifn_enit_split}"
        pairs = _load_ifn_enit_pairs(args.ifn_enit_root, args.ifn_enit_split)
        if args.max_samples is not None:
            pairs = pairs[: min(args.max_samples, len(pairs))]
        for p in pairs:
            pil = Image.open(p["image_path"]).convert("RGB")
            eval_items.append((pil, p["text"]))
    else:
        dataset_label = f"{args.dataset_name}  split: {args.split}"
        ds = load_dataset(args.dataset_name, split=args.split, trust_remote_code=True)
        n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
        ds = ds.select(range(n))
        for i in range(n):
            row = ds[i]
            img = row[args.image_column]
            pil = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            eval_items.append((pil, str(row[args.text_column])))

    # Corpus-level (micro) CER/WER calculation
    refs_scored: list[str] = []
    hyps_scored: list[str] = []
    skipped = 0

    for i, (pil, ref_raw) in enumerate(tqdm(eval_items, desc="Evaluating")):
        ref = normalize_text(ref_raw, collapse_ws=collapse_ws, strip_tatweel=args.strip_tatweel)
        if not ref:
            skipped += 1
            continue

        if preprocess_fn is not None:
            try:
                pil = preprocess_fn(pil)
            except Exception as e:
                print(f"\n[skip {i}] preprocessing error: {e}", file=sys.stderr)
                skipped += 1
                continue

        try:
            hyp_raw = ocr_one_image(
                model, processor, pil, device, args.max_new_tokens,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as e:
            print(f"\n[skip {i}] generation error: {e}", file=sys.stderr)
            skipped += 1
            continue

        hyp = normalize_text(hyp_raw, collapse_ws=collapse_ws, strip_tatweel=args.strip_tatweel)

        refs_scored.append(ref)
        hyps_scored.append(hyp)

    evaluated = len(refs_scored)
    if evaluated == 0:
        print("No examples evaluated.", file=sys.stderr)
        sys.exit(1)

    try:
        corpus_cer = jiwer.cer(refs_scored, hyps_scored)
        corpus_wer = jiwer.wer(refs_scored, hyps_scored)
    except Exception as e:
        print(f"Corpus metric error: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=== OCR evaluation ===")
    print(f"dataset:   {dataset_label}")
    print(f"samples:   {evaluated}  skipped: {skipped}")
    print(f"adapter:   {args.adapter or '(base model only)'}")
    if args.preprocess:
        print(f"preprocess: ON")
    print(f"CER: {corpus_cer:.4f}   (corpus / micro; lower is better, 0 = perfect)")
    print(f"WER: {corpus_wer:.4f}   (corpus / micro; lower is better, 0 = perfect)")
    print("======================")


if __name__ == "__main__":
    main()