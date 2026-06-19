"""Preprocessing pipeline to make arbitrary handwriting photos look like clean KHATT crops.

Removes grid lines (horizontal dotted, vertical bars), cleans noise via connected-component
filtering, and produces a tight-cropped black-on-white binary image that matches the
distribution KHATT training images have.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(
    image_path: str,
    *,
    scale: float = 4.0,
    adaptive_block: int = 51,
    adaptive_c: int = 15,
    vert_kernel_height: int = 80,
    horiz_join_width: int = 15,
    horiz_detect_width: int = 120,
    min_component_area: int = 60,
    crop_pad: int = 30,
    save_path: str | None = None,
) -> str:
    """Clean a handwriting photo into a KHATT-like binary crop.

    Returns the path to the saved preprocessed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    thresh = cv2.adaptiveThreshold(
        upscaled, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        adaptive_block, adaptive_c,
    )

    clean_text = thresh.copy()

    # Remove vertical bars (tall thin structures)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_height))
    detected_vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=3)
    clean_text = cv2.subtract(clean_text, detected_vert)

    # Remove horizontal dotted lines: dilate dots to join, then detect long runs
    horiz_join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_join_width, 1))
    horiz_joined = cv2.dilate(thresh, horiz_join_kernel, iterations=1)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_detect_width, 1))
    detected_horiz = cv2.morphologyEx(horiz_joined, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    clean_text = cv2.subtract(clean_text, detected_horiz)

    # Connected-component noise removal
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_text, connectivity=8)
    final_mask = np.zeros_like(clean_text)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_component_area:
            final_mask[labels == i] = 255

    # Light dilation to repair holes from line removal, then smooth
    final_mask = cv2.dilate(final_mask, np.ones((2, 2), np.uint8), iterations=1)
    smoothed = cv2.GaussianBlur(final_mask, (3, 3), 0)
    _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY_INV)

    cropped = _crop_content(binary, pad=crop_pad)

    if save_path is None:
        save_path = os.path.join(tempfile.gettempdir(), "hajji_preprocessed.png")
    cv2.imwrite(save_path, cropped)
    return save_path


def preprocess_pil(
    pil_image: Image.Image,
    **kwargs,
) -> Image.Image:
    """Preprocess a PIL image and return the cleaned result as PIL."""
    tmp_in = os.path.join(tempfile.gettempdir(), "hajji_prep_in.png")
    tmp_out = os.path.join(tempfile.gettempdir(), "hajji_prep_out.png")
    pil_image.save(tmp_in)
    preprocess_for_ocr(tmp_in, save_path=tmp_out, **kwargs)
    result = Image.open(tmp_out).convert("RGB")
    for p in (tmp_in, tmp_out):
        try:
            os.remove(p)
        except OSError:
            pass
    return result


def _crop_content(img: np.ndarray, pad: int = 30) -> np.ndarray:
    """Tight-crop around non-white pixels with padding."""
    coords = cv2.findNonZero(255 - img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = img.shape[:2]
    y0 = max(0, y - pad)
    y1 = min(h_img, y + h + pad)
    x0 = max(0, x - pad)
    x1 = min(w_img, x + w + pad)
    return img[y0:y1, x0:x1]
