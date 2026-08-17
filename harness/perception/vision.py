"""Vision helpers: encode images for vision-capable LLMs."""
from __future__ import annotations

import base64
import io

import numpy as np


def encode_image(image: np.ndarray, media_type: str = "image/png") -> str:
    """Encode an image (H, W[, C]) to a base64 payload string (no data-URI prefix)."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise ImportError("Pillow is required to encode images; run: pip install pillow") from e

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    fmt = media_type.split("/")[-1].upper()
    if fmt == "JPG":
        fmt = "JPEG"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_data_uri(image: np.ndarray, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{encode_image(image, media_type)}"
