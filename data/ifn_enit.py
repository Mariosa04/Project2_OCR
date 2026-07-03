"""Load IFN/ENIT (local image+label pairs) for handwritten Arabic OCR training."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any
import  json
from datasets import Dataset
from PIL import Image

from data.augment import augment_handwriting_image
from preprocess.prompts import USER_OCR_PROMPT


def _row_to_messages(image_path: str, text: str) -> list[dict[str, Any]]:
    t = (text or "").strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_OCR_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": t}],
        },
    ]


def _load_ifn_enit_pairs(root_dir: str, split: str) -> list[dict[str, str]]:
    """Scan split directory for paired image/label files."""
    split_dir = Path(root_dir) / split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {img_dir} and {lbl_dir} to exist. "
            f"Check that data/ifn_enit/{split}/images and labels folders are present."
        )

    pairs = []
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        stem = lbl_file.stem
        # Check for multiple image formats
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        img_file = None

        for ext in valid_exts:
            potential_file = img_dir / f"{stem}{ext}"
            if potential_file.exists():
                img_file = potential_file
                break

        # If no valid image was found for this text file, skip it
        if img_file is None:
            continue

        text = lbl_file.read_text(encoding="utf-8").strip()
        # IFN/ENIT uses dots as word separators
        text = text.replace(".", " ").strip()
        if not text:
            continue

        pairs.append({"image_path": str(img_file), "text": text})

    return pairs


def build_ifn_enit_dataset(
    root_dir: str,
    split: str = "train",
    seed: int = 42,
    augment_prob: float = 0.45,
    max_samples: int | None = None,
) -> Dataset:
    """Build a Qwen2-VL chat-formatted dataset from local IFN/ENIT files."""
    pairs = _load_ifn_enit_pairs(root_dir, split)

    if max_samples is not None:
        pairs = pairs[: min(max_samples, len(pairs))]

    rng = random.Random(seed)

    ds = Dataset.from_list(pairs)

    def _map(batch):
        out = {"messages": []}
        for img_path, text in zip(batch["image_path"], batch["text"]):
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            pil = augment_handwriting_image(pil, rng=rng, p=augment_prob)

            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f"hajji_ifn_{rng.randint(0, int(1e9))}.png")
            pil.save(tmp_path)

            out["messages"].append(json.dumps(_row_to_messages(tmp_path, text)))
        return out

    return ds.map(
        _map,
        batched=True,
        batch_size=16,
        remove_columns=ds.column_names,
        desc="IFN/ENIT augment + build messages",
    )
