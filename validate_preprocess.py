"""Small-scale validator for preprocess_image.

Computes per-image:
- preprocessing time (from the preprocessing meta)
- simulated inference time before/after (proportional to visual token count)
- token reduction
- quality (SSIM if available, otherwise MSE) between baseline-resized and processed image

Outputs CSV with per-image rows and a summary row.
"""
import argparse
from pathlib import Path
import csv
import math
import time
from typing import Optional

from preprocess_image import preprocess_image, load_config
from PIL import Image
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False


def token_count_for_resolution(res: int, patch_size: int) -> int:
    return (res // patch_size) * (res // patch_size)


def image_to_square(img: Image.Image, target: int) -> Image.Image:
    # resize preserving aspect ratio then pad to square
    w, h = img.size
    scale = target / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    out = Image.new("RGB", (target, target), (0, 0, 0))
    out.paste(resized, ((target - new_w) // 2, (target - new_h) // 2))
    return out


def compute_quality(baseline_img: Image.Image, proc_img: Image.Image) -> float:
    a = np.array(baseline_img.convert("L"), dtype=np.float32)
    b = np.array(proc_img.convert("L"), dtype=np.float32)
    if HAVE_SSIM:
        try:
            val = ssim(a, b, data_range=b.max() - b.min())
            return float(val)
        except Exception:
            pass
    # fallback: negative MSE (so higher is better)
    mse = float(((a - b) ** 2).mean())
    # convert to a bounded score: 1 / (1 + mse_norm)
    mse_norm = mse / (255.0 ** 2)
    return 1.0 / (1.0 + mse_norm)


def run(images_dir: Path, cfg_path: Optional[str], out_csv: Path, baseline_res: int, patch_size: int, token_time: float):
    cfg = load_config(cfg_path)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')])
    rows = []
    sum_pre = 0.0
    sum_sim_before = 0.0
    sum_sim_after = 0.0
    sum_tokens_before = 0
    sum_tokens_after = 0
    for p in images:
        try:
            t0 = time.time()
            proc_img, meta = preprocess_image(str(p), cfg)
            preproc_time = meta.get('processing_time_s', time.time() - t0)

            # baseline: no preprocessing, inference at baseline_res
            baseline_tokens = token_count_for_resolution(baseline_res, patch_size)
            baseline_infer_time = baseline_tokens * token_time

            # after: tokens from processed final size
            fw, fh = meta.get('final_size', proc_img.size)
            # final is square; use max
            final_res = max(fw, fh)
            after_tokens = token_count_for_resolution(final_res, patch_size)
            after_infer_time = after_tokens * token_time

            # token reduction
            token_reduction = 1.0 - (after_tokens / float(baseline_tokens)) if baseline_tokens > 0 else 0.0

            # quality: compare baseline-resized image vs processed image
            orig = Image.open(p).convert('RGB')
            baseline_img = image_to_square(orig, baseline_res)
            # ensure processed image is same size as baseline for comparison
            proc_for_q = proc_img.resize(baseline_img.size, Image.LANCZOS)
            quality = compute_quality(baseline_img, proc_for_q)

            rows.append({
                'image': p.name,
                'original_size': f"{orig.size[0]}x{orig.size[1]}",
                'preproc_time_s': round(preproc_time, 6),
                'baseline_tokens': baseline_tokens,
                'after_tokens': after_tokens,
                'token_reduction': round(token_reduction, 4),
                'baseline_infer_s': round(baseline_infer_time, 6),
                'after_infer_s': round(after_infer_time, 6),
                'total_before_s': round(baseline_infer_time, 6),
                'total_after_s': round(preproc_time + after_infer_time, 6),
                'quality_score': round(float(quality), 6),
            })

            sum_pre += preproc_time
            sum_tokens_before += baseline_tokens
            sum_tokens_after += after_tokens
            sum_sim_after += quality
        except Exception as e:
            print(f"ERR processing {p}: {e}")

    # write CSV
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['image','original_size','preproc_time_s','baseline_tokens','after_tokens','token_reduction','baseline_infer_s','after_infer_s','total_before_s','total_after_s','quality_score']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        # summary row
        if rows:
            avg_pre = sum_pre / len(rows)
            avg_token_red = 1.0 - (sum_tokens_after / float(sum_tokens_before)) if sum_tokens_before>0 else 0.0
            avg_quality = sum_sim_after / len(rows)
            w.writerow({
                'image':'__SUMMARY__',
                'original_size':'',
                'preproc_time_s':round(avg_pre,6),
                'baseline_tokens':sum_tokens_before,
                'after_tokens':sum_tokens_after,
                'token_reduction':round(avg_token_red,6),
                'baseline_infer_s':'',
                'after_infer_s':'',
                'total_before_s':'',
                'total_after_s':'',
                'quality_score':round(avg_quality,6),
            })

    print(f"Wrote {out_csv} with {len(rows)} rows")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images-dir', default='images')
    p.add_argument('--config', default='preprocess_config.yaml')
    p.add_argument('--baseline-res', type=int, default=512)
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--token-time', type=float, default=0.0005, help='Simulated inference time per visual token (s)')
    p.add_argument('--out-csv', default='validation_small.csv')
    args = p.parse_args()
    run(Path(args.images_dir), args.config, Path(args.out_csv), args.baseline_res, args.patch_size, args.token_time)
