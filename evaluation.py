#!/usr/bin/env python3
"""
Evaluate a trained LoRA (or base model) on KHATT (HuggingFace) or local IFN/ENIT using mean CER / WER.

Run after training, pointing --adapter at the same folder you passed as --output_dir to train.py.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR evaluation: CER & WER on a dataset split")
    p.add_argument("--adapter", type=str, default=None, help="LoRA folder from training (omit for base-only)")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")

    hf_g = p.add_argument_group("HuggingFace dataset (default: KHATT)")
    hf_g.add_argument("--dataset_name", type=str, default="johnlockejrr/KHATT_v1.0_dataset")
    hf_g.add_argument("--split", type=str, default="validation", help="Dataset split, e.g. validation or test")
    hf_g.add_argument("--image_column", type=str, default="image")
    hf_g.add_argument("--text_column", type=str, default="text")

    ifn_g = p.add_argument_group("Local IFN/ENIT (overrides HuggingFace dataset when set)")
    ifn_g.add_argument(
        "--ifn_enit_root",
        type=str,
        default=None,
        help="Path to IFN/ENIT root (e.g. ./data/ifn_enit)",
    )
    ifn_g.add_argument("--ifn_enit_split", type=str, default="test", help="IFN/ENIT split: train/val/test")

    p.add_argument("--max_samples", type=int, default=None, help="Cap number of examples (faster smoke eval)")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding (not recommended for OCR metrics)",
    )
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument(
        "--preprocess",
        action="store_true",
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
    repetition_penalty: float | None = None,
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
    if repetition_penalty is not None and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _load_ifn_enit_pairs(root: str, split: str) -> list[dict[str, str]]:
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


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: evaluation on CPU is very slow.", file=sys.stderr)

    collapse_ws = not args.no_normalize_ws
    use_ifn = bool(args.ifn_enit_root)

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

    eval_items: list[tuple[Image.Image, str]] = []
    dataset_label = ""

    if use_ifn:
        dataset_label = f"IFN/ENIT ({args.ifn_enit_root})  split: {args.ifn_enit_split}"
        pairs = _load_ifn_enit_pairs(args.ifn_enit_root, args.ifn_enit_split)
        if args.max_samples is not None:
            pairs = pairs[: min(args.max_samples, len(pairs))]
        for p in pairs:
            eval_items.append((Image.open(p["image_path"]).convert("RGB"), p["text"]))
    else:
        dataset_label = f"{args.dataset_name}  split: {args.split}"
        ds = load_dataset(args.dataset_name, split=args.split, trust_remote_code=True)
        n_ds = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
        ds = ds.select(range(n_ds))
        for i in range(n_ds):
            row = ds[i]
            img = row[args.image_column]
            pil = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            eval_items.append((pil, str(row[args.text_column])))

    n = len(eval_items)
    cers: list[float] = []
    wers: list[float] = []
    skipped = 0

    for i in tqdm(range(n), desc="Evaluating"):
        pil, ref_raw = eval_items[i]
        ref = normalize_text(str(ref_raw), collapse_ws=collapse_ws, strip_tatweel=args.strip_tatweel)
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
                model,
                processor,
                pil,
                device,
                args.max_new_tokens,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as e:
            print(f"\n[skip {i}] generation error: {e}", file=sys.stderr)
            skipped += 1
            continue

        hyp = normalize_text(hyp_raw, collapse_ws=collapse_ws, strip_tatweel=args.strip_tatweel)

        try:
            cers.append(jiwer.cer(ref, hyp))
            wers.append(jiwer.wer(ref, hyp))
        except Exception as e:
            print(f"\n[skip {i}] metric error: {e}", file=sys.stderr)
            skipped += 1
            continue

    evaluated = len(cers)
    if evaluated == 0:
        print("No examples evaluated.", file=sys.stderr)
        sys.exit(1)

    mean_cer = sum(cers) / evaluated
    mean_wer = sum(wers) / evaluated

    print()
    print("=== OCR evaluation ===")
    print(f"dataset:   {dataset_label}")
    print(f"samples:   {evaluated}  skipped: {skipped}")
    print(f"adapter:   {args.adapter or '(base model only)'}")
    if args.preprocess:
        print("preprocess: ON")
    print(f"CER (mean): {mean_cer:.4f}   (lower is better, 0 = perfect)")
    print(f"WER (mean): {mean_wer:.4f}   (lower is better, 0 = perfect)")
    print("======================")


if __name__ == "__main__":
    main()
