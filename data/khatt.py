"""Load KHATT (handwritten Arabic) and format conversational SFT rows for Qwen2-VL."""

from __future__ import annotations

import random
from typing import Any

from datasets import Dataset, load_dataset
import json
from data.augment import  augment_handwriting_image
from preprocess.prompts import USER_OCR_PROMPT
import tempfile
import os



def _row_to_messages(image, text: str) -> dict[str, Any]:
    t = (text or "").strip()
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_OCR_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": t}],
            },
        ],
    }


def build_handwriting_dataset(
    dataset_name: str,
    split: str,
    text_column: str,
    image_column: str,
    seed: int,
    augment_prob: float,
    max_samples: int | None = None,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    rng = random.Random(seed)

    def _map(batch):
        images = batch[image_column]
        texts = batch[text_column]
        out = {"messages": []}
        for im, tx in zip(images, texts):
            pil = im.convert("RGB") if hasattr(im, "convert") else im
            pil = augment_handwriting_image(pil, rng=rng, p=augment_prob)
            # 🔥 SAFE TEMP PATH
            tmp_dir = tempfile.gettempdir()
            img_path = os.path.join(tmp_dir, f"hajji_{rng.randint(0, 1e9)}.png")

            pil.save(img_path)

            out["messages"].append(json.dumps(_row_to_messages(img_path, str(tx))["messages"]))
        return out

    return ds.map(
        _map,
        batched=True,
        batch_size=16,
        remove_columns=ds.column_names,
        desc="Handwriting augment + build messages",
    )
