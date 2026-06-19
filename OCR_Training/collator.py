"""Batch Qwen2-VL OCR examples with loss only on assistant tokens."""

from __future__ import annotations

from collections.abc import Callable

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def _resolve_im_start_end_ids(tokenizer) -> tuple[int, int]:
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.eos_token_id
    return im_start, im_end


def make_qwen2_vl_collator(processor: AutoProcessor) -> Callable:
    tokenizer = processor.tokenizer
    im_start_id, im_end_id = _resolve_im_start_end_ids(tokenizer)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

    def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        messages_batch = [ex["messages"] for ex in examples]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_batch
        ]
        all_images: list = []
        for m in messages_batch:
            imgs, _ = process_vision_info(m)
            if imgs:
                all_images.extend(imgs)

        batch = processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for idx in range(labels.shape[0]):
            ids = batch["input_ids"][idx].tolist()
            in_assistant = False
            i = 0
            while i < len(ids):
                if ids[i] == im_start_id:
                    match = all(
                        i + 1 + j < len(ids) and ids[i + 1 + j] == assistant_ids[j]
                        for j in range(len(assistant_ids))
                    )
                    if match:
                        for k in range(1 + len(assistant_ids)):
                            labels[idx, i + k] = -100
                        skip = 1 + len(assistant_ids)
                        if i + skip < len(ids):
                            tok = tokenizer.decode([ids[i + skip]])
                            if tok.strip() == "":
                                labels[idx, i + skip] = -100
                                skip += 1
                        i += skip
                        in_assistant = True
                        continue
                    labels[idx, i] = -100
                elif ids[i] == im_end_id:
                    if in_assistant:
                        in_assistant = False
                    else:
                        labels[idx, i] = -100
                elif not in_assistant:
                    labels[idx, i] = -100
                i += 1

        batch["labels"] = labels
        return batch

    return collate_fn
