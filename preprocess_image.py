import time
import math
from typing import Tuple, Dict, Any, Optional
import yaml
import numpy as np
from PIL import Image, ImageOps
import cv2


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    default = {
        "complexity_analysis": {
            "use_edge_detection": True,
            "use_entropy": True,
            "use_text_detection": False,
        },
        "resolution_tiers": {"low": 336, "medium": 512, "high": 768, "ultra": 1024},
        "thresholds": {"low_complexity": 0.3, "high_complexity": 0.7},
        "cropping": {"enabled": True, "min_margin": 10, "max_crop_ratio": 0.3, "background_threshold": None},
        "output": {"save_visualizations": False, "save_metrics": True, "verbose": False},
    }
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # merge defaults shallowly
    for k, v in default.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                cfg[k].setdefault(kk, vv)
    return cfg


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def analyze_complexity(pil_img: Image.Image, config: Dict[str, Any]) -> Dict[str, float]:
    # returns metrics and combined complexity score in [0,1]
    img_cv = pil_to_cv2(pil_img)
    h, w = img_cv.shape[:2]
    area = float(w * h)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    edge_density = 0.0
    entropy_norm = 0.0
    text_density = 0.0

    if config["complexity_analysis"].get("use_edge_detection", True):
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / area

    if config["complexity_analysis"].get("use_entropy", True):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        p = hist / (hist.sum() + 1e-12)
        p_nonzero = p[p > 0]
        entropy = -float((p_nonzero * np.log2(p_nonzero)).sum())
        entropy_norm = min(entropy / 8.0, 1.0)

    if config["complexity_analysis"].get("use_text_detection", False):
        try:
            import pytesseract

            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            n_boxes = sum(1 for t in data.get("text", []) if t.strip())
            text_density = float(n_boxes) / (area / (100 * 100))  # boxes per 100x100 patch
            text_density = min(text_density, 1.0)
        except Exception:
            text_density = 0.0

    # combine metrics with simple weights
    w_edge = 0.5
    w_entropy = 0.4
    w_text = 0.1
    score = w_edge * min(edge_density * 5.0, 1.0) + w_entropy * entropy_norm + w_text * text_density
    score = max(0.0, min(score, 1.0))

    return {
        "edge_density": edge_density,
        "entropy_norm": entropy_norm,
        "text_density": text_density,
        "complexity_score": score,
    }


def select_resolution(score: float, config: Dict[str, Any]) -> int:
    tiers = config.get("resolution_tiers", {"low": 336, "medium": 512, "high": 768, "ultra": 1024})
    th_low = config.get("thresholds", {}).get("low_complexity", 0.3)
    th_high = config.get("thresholds", {}).get("high_complexity", 0.7)
    if score < th_low:
        return int(tiers.get("low", 336))
    if score < th_high:
        return int(tiers.get("medium", 512))
    # high or ultra thresholding
    if score >= 0.9 and "ultra" in tiers:
        return int(tiers.get("ultra", tiers.get("high", 768)))
    return int(tiers.get("high", 768))


def crop_content(pil_img: Image.Image, config: Dict[str, Any]) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    crop_cfg = config.get("cropping", {})
    if not crop_cfg.get("enabled", True):
        return pil_img, None

    img_cv = pil_to_cv2(pil_img)
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    thr = crop_cfg.get("background_threshold", None)
    if thr is None:
        # Otsu threshold to detect background
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(thr), 255, cv2.THRESH_BINARY)

    # invert mask so content is white
    mask_inv = 255 - mask

    # remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img, None

    # pick the union bounding box of contours that represent >= 1% of image area
    areas = [(cv2.contourArea(c), c) for c in contours]
    areas = [a for a in areas if a[0] >= 0.001 * (w * h)]
    if not areas:
        return pil_img, None

    xs, ys, xe, ye = w, h, 0, 0
    for _, c in areas:
        x, y, cw, ch = cv2.boundingRect(c)
        xs = min(xs, x)
        ys = min(ys, y)
        xe = max(xe, x + cw)
        ye = max(ye, y + ch)

    # safety
    xs = max(0, xs)
    ys = max(0, ys)
    xe = min(w, xe)
    ye = min(h, ye)

    crop_area = (xe - xs) * (ye - ys)
    total_area = w * h
    crop_ratio = 1.0 - (crop_area / float(total_area))
    max_crop = float(crop_cfg.get("max_crop_ratio", 0.3))
    if crop_ratio > max_crop:
        # cropping would remove too much, skip cropping
        return pil_img, None

    min_margin = int(crop_cfg.get("min_margin", 10))
    xs = max(0, xs - min_margin)
    ys = max(0, ys - min_margin)
    xe = min(w, xe + min_margin)
    ye = min(h, ye + min_margin)

    cropped = pil_img.crop((xs, ys, xe, ye))
    return cropped, (xs, ys, xe, ye)


def resize_and_pad(pil_img: Image.Image, target: int) -> Image.Image:
    # Resize preserving aspect ratio and pad to square (target x target)
    w, h = pil_img.size
    if w == target and h == target:
        return pil_img
    scale = target / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    # pad to target
    padded = Image.new("RGB", (target, target), (0, 0, 0))
    off_x = (target - new_w) // 2
    off_y = (target - new_h) // 2
    padded.paste(resized, (off_x, off_y))
    return padded


def preprocess_image(input_image: Any, config: Optional[Dict[str, Any]] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    # input_image: path or PIL Image
    cfg = config or load_config(None)
    if isinstance(input_image, str):
        pil_img = Image.open(input_image).convert("RGB")
    elif isinstance(input_image, Image.Image):
        pil_img = input_image.convert("RGB")
    else:
        raise ValueError("input_image must be a file path or PIL.Image.Image")

    meta: Dict[str, Any] = {"original_size": pil_img.size}
    t0 = time.time()
    metrics = analyze_complexity(pil_img, cfg)
    meta.update(metrics)

    target = select_resolution(metrics["complexity_score"], cfg)
    meta["selected_resolution"] = target

    cropped_img, crop_coords = (pil_img, None)
    if cfg.get("cropping", {}).get("enabled", True):
        cropped_img, crop_coords = crop_content(pil_img, cfg)
    meta["crop_coords"] = crop_coords

    final = resize_and_pad(cropped_img, target)
    meta["final_size"] = final.size
    meta["processing_time_s"] = time.time() - t0
    return final, meta


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive-resolution + content-aware image preprocessing for FastVLM")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output", help="Output image path (saved as PNG)")
    parser.add_argument("--config", help="YAML config path", default="preprocess_config.yaml")
    parser.add_argument("--show", action="store_true", help="Open resulting image after processing (PIL.show)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_img, meta = preprocess_image(args.input, cfg)
    if args.output:
        out_img.save(args.output, format="PNG")
    if args.show:
        out_img.show()
    print("METADATA:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
