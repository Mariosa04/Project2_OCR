"""Handwriting-realistic augmentations: noise, blur, illumination, mild elastic warp."""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _elastic_warp_gray(gray: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    h, w = gray.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def augment_handwriting_image(img: Image.Image, rng: random.Random, p: float = 0.45) -> Image.Image:
    """Apply random degradations common in phone scans and uneven ink."""
    if rng.random() > p:
        return img.convert("RGB")

    a = np.array(img.convert("RGB"))
    rgb = a.copy()

    # Mild illumination / contrast
    if rng.random() < 0.5:
        pil = Image.fromarray(rgb)
        pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(0.75, 1.2))
        pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.85, 1.25))
        rgb = np.array(pil)

    # Gaussian noise on luminance
    if rng.random() < 0.45:
        noise = rng.gauss(0, rng.uniform(4.0, 14.0))
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Motion / defocus blur
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        if rng.random() < 0.5:
            rgb = np.array(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2))))
        else:
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            rgb = cv2.filter2D(rgb, -1, kernel)

    # Elastic warp (grayscale-driven, applied per channel)
    if rng.random() < 0.25:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        alpha = rng.uniform(4.0, 10.0)
        sigma = rng.uniform(3.0, 5.0)
        warped = _elastic_warp_gray(gray, alpha=alpha, sigma=sigma)
        # blend small amount to preserve color while bending strokes
        blend = rng.uniform(0.35, 0.65)
        for c in range(3):
            ch = rgb[:, :, c].astype(np.float32)
            rgb[:, :, c] = np.clip(blend * warped + (1 - blend) * ch, 0, 255).astype(np.uint8)

    # JPEG recompression artifacts
    if rng.random() < 0.3:
        q = int(rng.uniform(45, 85))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            rgb = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb)
