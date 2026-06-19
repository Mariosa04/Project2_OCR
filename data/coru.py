"""dthdh Load CORU OCR dataset and format conversational SFT rows for Qwen2-VL."""

from __future__ import annotations

import os
import random
import tempfile
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from PIL import Image

from data.augment import augment_handwriting_image
from preprocess.prompts import USER_OCR_PROMPT


def _row_to_messages(image: str, text: str) -> dict[str, Any]:
  """Format a single image-text pair into SFT conversation format."""
  t = (text or "").strip()
  return {
    "messages": [
      {
        "role": "user",
        "content": [
          # Added dummy "text": "" so HF doesn't inject null
          {"type": "image", "image": image, "text": ""},
          # Added dummy "image": "" so HF doesn't inject null
          {"type": "text", "image": "", "text": USER_OCR_PROMPT},
        ],
      },
      {
        "role": "assistant",
        # Added dummy "image": "" so HF doesn't inject null
        "content": [{"type": "text", "image": "", "text": t}],
      },
    ],
  }


def _crop_to_content(
    image: Image.Image,
    threshold: int = 240,
    margin: int = 2,
) -> tuple[Image.Image, int, int]:
    """Crop image to bounding box of non-background content."""
    gray = image.convert("L")
    np_img = np.array(gray)

    rows_with_content = np.any(np_img < threshold, axis=1)
    cols_with_content = np.any(np_img < threshold, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return image, image.width, image.height

    y_min, y_max = np.where(rows_with_content)[0][[0, -1]]
    x_min, x_max = np.where(cols_with_content)[0][[0, -1]]

    y_min = max(0, y_min - margin)
    y_max = min(image.height - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(image.width - 1, x_max + margin)

    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    return cropped, *cropped.size


class CORUOCRDataset(torch.utils.data.Dataset):
    """Dataset loader for CORU OCR format."""
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self, root_dir: str, split: str):
        # Expects: root_dir/OCR/split/split/
        self.data_dir = os.path.join(root_dir, "OCR", split, split)

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"CORU data directory not found: {self.data_dir}\n"
                f"Expected structure: {root_dir}/OCR/<split>/<split>/"
            )

        self.image_files = sorted(
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith(self.VALID_EXTENSIONS)
        )

        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.data_dir, os.path.splitext(img_name)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        image, w, h = _crop_to_content(image)

        texts = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith('["') and content.endswith('"]'):
                texts = [content[2:-2]]

        target = {
            "boxes": torch.tensor([[0, 0, w, h]], dtype=torch.float32) if texts else torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64) if texts else torch.empty((0,), dtype=torch.int64),
            "texts": texts,
            "image_id": torch.tensor([idx]),
        }
        return image, target


def build_handwriting_dataset_coru(
    dataset_name: str,
    split: str,
    text_column: str = "text",
    image_column: str = "image",
    seed: int = 42,
    augment_prob: float = 0.0,
    max_samples: int | None = None,
) -> Dataset:
    """
    Load CORU handwriting dataset and format for Qwen2-VL SFT training.
    Strictly uses local file loading. Does NOT fall back to HuggingFace.
    """
    coru_split_path = os.path.join(dataset_name, "OCR", split, split)

    # STRICT CHECK: If the exact CORU folder structure doesn't exist, fail immediately.
    if not os.path.exists(coru_split_path):
        raise FileNotFoundError(
            f"CORU directory not found at {coru_split_path}.\n"
            f"Make sure your dataset is unzipped and matches the pattern: "
            f"{dataset_name}/OCR/{split}/{split}/"
        )

    print(f"[CORU] Loading local files from {coru_split_path}")
    coru = CORUOCRDataset(root_dir=dataset_name, split=split)

    limit = max_samples if max_samples is not None else len(coru)
    records = []

    for i in range(min(limit, len(coru))):
        image, target = coru[i]
        text = target["texts"][0] if target["texts"] else ""
        records.append({"image": image, "text": text})

    ds = Dataset.from_list(records)

    rng = random.Random(seed)
    tmp_dir = tempfile.gettempdir()

    def _map(batch: dict) -> dict:
        images = batch[image_column]
        texts = batch[text_column]
        out = {"messages": []}

        for im, tx in zip(images, texts):
            pil = im.convert("RGB") if hasattr(im, "convert") else im
            pil = augment_handwriting_image(pil, rng=rng, p=augment_prob)

            img_path = os.path.join(tmp_dir, f"hajji_{rng.randint(0, 10**9 - 1)}.png")
            pil.save(img_path)

            out["messages"].append(_row_to_messages(img_path, str(tx))["messages"])
        return out

    return ds.map(
        _map,
        batched=True,
        batch_size=16,
        remove_columns=ds.column_names,
        desc="CORU: augment + build messages",
    )