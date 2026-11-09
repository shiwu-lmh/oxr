import cv2
import numpy as np


def _estimate_text_skew(gray: np.ndarray) -> float | None:
    """Estimate only small page skew from long, near-horizontal ink lines."""
    height, width = gray.shape[:2]
    if min(height, width) < 48:
        return None

    scale = min(1.0, 900 / max(height, width))
    sample = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else gray
    )
    edges = cv2.Canny(sample, 60, 160)
    sample_width = sample.shape[1]
    min_length = max(24, int(sample_width * 0.22))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(18, min_length // 2),
        minLineLength=min_length, maxLineGap=max(5, min_length // 5),
    )
    if lines is None:
        return None

    candidates: list[tuple[float, float]] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = (int(value) for value in line)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if dx == 0 or length < min_length:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) <= 6.0:
            candidates.append((angle, length))

    if len(candidates) < 2:
        return None
    angles = np.array([angle for angle, _ in candidates], dtype=np.float32)
    weights = np.array([length for _, length in candidates], dtype=np.float32)
    median = float(np.median(angles))
    deviations = np.abs(angles - median)
    consistent = deviations <= 1.25
    if int(consistent.sum()) < 2:
        return None
    if float(weights[consistent].sum()) < sample_width * 0.5:
        return None
    return median if 0.35 <= abs(median) <= 5.0 else None


def preprocess_text_image(image: np.ndarray) -> np.ndarray:
    """Apply low-cost contrast, sharpness, and reliable small-skew correction."""
    if image.size == 0 or image.ndim not in {2, 3}:
        return image
    height, width = image.shape[:2]
    if min(height, width) < 16:
        return image

    color = image.ndim == 3 and image.shape[2] >= 3
    if color:
        lab = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0]
    else:
        lightness = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    skew = _estimate_text_skew(lightness)
    low, high = np.percentile(lightness, (2, 98))
    spread = float(high - low)
    enhanced = lightness
    if spread < 210:
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        equalized = clahe.apply(lightness)
        blend = min(0.45, max(0.12, (210.0 - spread) / 240.0))
        enhanced = cv2.addWeighted(lightness, 1.0 - blend, equalized, blend, 0)

    sharpness = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())
    if sharpness < 260:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
        amount = 0.22 if sharpness < 100 else 0.14
        enhanced = cv2.addWeighted(enhanced, 1.0 + amount, blurred, -amount, 0)

    if color:
        lab[:, :, 0] = enhanced
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        output = enhanced

    if skew is not None:
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, skew, 1.0)
        border = (255, 255, 255) if color else 255
        output = cv2.warpAffine(
            output, matrix, (width, height), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border,
        )
    return output


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize an image isotropically while keeping both dimensions nonzero."""
    if scale <= 0:
        raise ValueError("Image scale must be greater than zero.")
    height, width = image.shape[:2]
    target_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def crop_image(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Crop an image using the given bounding box.
    bbox: [x1, y1, x2, y2]
    """
    # D-FINE emits floating-point pixel coordinates. Quantize them to the
    # nearest pixel so recognition crops match the reference pipeline.
    x1, y1, x2, y2 = (round(value) for value in bbox)
    # Ensure coordinates are within image boundaries
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return image[y1:y2, x1:x2].copy()
