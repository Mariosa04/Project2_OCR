"""Build a mixed training dataset from KHATT (HuggingFace) + IFN/ENIT + Arabic digits (local)."""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets

from data.arabic_digits import build_arabic_digits_dataset
from data.ifn_enit import build_ifn_enit_dataset
from data.khatt import build_handwriting_dataset
from data.coru import build_handwriting_dataset_coru as build_coru_dataset


def build_mixed_dataset(
    *,
    khatt_dataset_name: str,
    khatt_split: str,
    khatt_text_column: str,
    khatt_image_column: str,
    ifn_enit_root: str | None,
    ifn_enit_split: str = "train",
    seed: int = 42,
    augment_prob: float = 0.45,
    khatt_max_samples: int | None = None,
    ifn_enit_max_samples: int | None = None,
    use_khatt: bool = True,
    arabic_digits_root: str | None = None,
    arabic_digits_split: str = "train",
    arabic_digits_max_samples: int | None = None,
    # CORU
    coru_root: str | None = None,
    coru_split: str = "train",
    coru_max_samples: int | None = None,
    use_coru: bool = False,
) -> Dataset:
    """Concatenate KHATT + IFN/ENIT + Arabic digits into one shuffled dataset.

    Any source can be skipped:
      * KHATT     — set ``use_khatt=False`` (avoids the HuggingFace download).
      * IFN/ENIT  — leave ``ifn_enit_root`` as None.
      * Digits    — leave ``arabic_digits_root`` as None.

    Raises if no sources are enabled.
    """
    parts: list[Dataset] = []

    if use_khatt:
        print(f"Loading KHATT ({khatt_dataset_name}, split={khatt_split}) ...")
        khatt_ds = build_handwriting_dataset(
            dataset_name=khatt_dataset_name,
            split=khatt_split,
            text_column=khatt_text_column,
            image_column=khatt_image_column,
            seed=seed,
            augment_prob=augment_prob,
            max_samples=khatt_max_samples,
        )
        print(f"  KHATT rows: {len(khatt_ds)}")
        parts.append(khatt_ds)

    if ifn_enit_root:
        print(f"Loading IFN/ENIT ({ifn_enit_root}, split={ifn_enit_split}) ...")
        ifn_ds = build_ifn_enit_dataset(
            root_dir=ifn_enit_root,
            split=ifn_enit_split,
            seed=seed,
            augment_prob=augment_prob,
            max_samples=ifn_enit_max_samples,
        )
        print(f"  IFN/ENIT rows: {len(ifn_ds)}")
        parts.append(ifn_ds)

    if arabic_digits_root:
        print(f"Loading Arabic digits ({arabic_digits_root}, split={arabic_digits_split}) ...")
        dg_ds = build_arabic_digits_dataset(
            root_dir=arabic_digits_root,
            split=arabic_digits_split,
            seed=seed,
            augment_prob=augment_prob,
            max_samples=arabic_digits_max_samples,
        )
        print(f"  Arabic digits rows: {len(dg_ds)}")
        parts.append(dg_ds)
    # === CORU ===
    if use_coru and coru_root:
        print(f"[Mixed] Including CORU: {coru_root}")
        coru_ds = build_coru_dataset(
            dataset_name=coru_root,
            split=coru_split,
            seed=seed,
            augment_prob=augment_prob,
            max_samples=coru_max_samples,
        )
        parts.append(coru_ds)
    if not parts:
        raise ValueError(
            "No training sources enabled. Pass --arabic_digits_root and/or "
            "--ifn_enit_root, or keep KHATT enabled."
        )

    combined = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    combined = combined.shuffle(seed=seed)
    print(f"Mixed dataset total: {len(combined)} rows (shuffled)")
    return combined
