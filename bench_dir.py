# infer_bench_folder.py
# Benchmarks a folder of images using the FastVLM/LLaVA pipeline.
# Writes CSV with per-image metrics including encoder_time_s, ttft_s, full_gen_s,
# tokens_generated, tokens_per_sec, peak_gpu_mem_gb, and a 20-char truncated preview of the output.
#
# Usage:
#   python infer_bench_folder.py
# (defaults: ./images, ./checkpoints/llava-fastvithd_0.5b_stage3, results.csv)
#
import argparse
import time
import csv
from pathlib import Path

import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from preprocess_image import preprocess_image, load_config


class VLMRunner:
    def __init__(self, model_path, model_base=None, conv_mode="qwen_2", device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")
        self.device = device

        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, model_name, device=self.device
        )
        self.conv_mode = conv_mode

        # Ensure pad token is set if possible
        try:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        except Exception:
            pass

    def run_once(self, image_path: Path, prompt: str, max_new_tokens: int = 256, gen_kwargs=None):
        """Run inference on a single image and return metrics dict."""
        if gen_kwargs is None:
            gen_kwargs = dict(do_sample=True, temperature=0.2, top_p=None, num_beams=1, use_cache=True)

        # Build conversation prompt (same as repo expects)
        if self.model.config.mm_use_im_start_end:
            full_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Tokenize (repo helper)
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        # Load and process image
        image = Image.open(str(image_path)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)[0].to(self.device)
        # match repo usage (unsqueeze + .half() if model expects)
        try:
            image_tensor = image_tensor.unsqueeze(0).half()
        except Exception:
            image_tensor = image_tensor.unsqueeze(0)

        results = {"filename": str(image_path.name)}

        # Reset peak mem before each run for per-image peak
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # 1) Optional encoder timing
        try:
            torch.cuda.synchronize() if self.device == "cuda" else None
            t0 = time.perf_counter()
            if hasattr(self.model, "vision_encoder"):
                _ = self.model.vision_encoder(image_tensor)
            elif hasattr(self.model, "encode_image"):
                _ = self.model.encode_image(image_tensor)
            else:
                # No explicit encoder to time
                pass
            torch.cuda.synchronize() if self.device == "cuda" else None
            t1 = time.perf_counter()
            results["encoder_time_s"] = t1 - t0
        except Exception:
            results["encoder_time_s"] = None

        # 2) TTFT (generate 1 token)
        try:
            torch.cuda.synchronize() if self.device == "cuda" else None
            t_start = time.perf_counter()
            try:
                first_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    max_new_tokens=1,
                    **gen_kwargs
                )
            except TypeError:
                # fallback if signature different
                first_ids = self.model.generate(input_ids, max_new_tokens=1, **gen_kwargs)
            torch.cuda.synchronize() if self.device == "cuda" else None
            t_end = time.perf_counter()
            results["ttft_s"] = t_end - t_start
        except Exception as e:
            first_ids = None

        # 3) Full generation
        try:
            torch.cuda.synchronize() if self.device == "cuda" else None
            t2 = time.perf_counter()
            try:
                full_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs
                )
            except TypeError:
                full_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
            torch.cuda.synchronize() if self.device == "cuda" else None
            t3 = time.perf_counter()
            results["full_gen_s"] = t3 - t2

            # compute number of generated tokens
            try:
                in_len = int(input_ids.shape[-1])
                out_len = int(full_ids.shape[-1]) if len(full_ids.shape) >= 1 else len(full_ids[0])
                # Some models return shape [batch, seq], others return list-like; handle common cases
                # If full_ids is tensor with batch dimension, take first row length
                if hasattr(full_ids, "shape") and len(full_ids.shape) == 2:
                    out_len = int(full_ids.shape[-1])
                # tokens generated = out_len - in_len (floor to 0)
                gen_tokens = max(0, out_len - in_len)
            except Exception:
                # fallback: try decoding and counting tokens via tokenizer (less reliable)
                try:
                    decoded = self.tokenizer.batch_decode(full_ids, skip_special_tokens=True)
                    tok_count = sum([len(self.tokenizer.encode(d)) for d in decoded])
                    gen_tokens = tok_count
                except Exception:
                    gen_tokens = None

            results["tokens_generated"] = gen_tokens
            if gen_tokens is None or results["full_gen_s"] is None or results["full_gen_s"] == 0:
                results["tokens_per_sec"] = None
            else:
                results["tokens_per_sec"] = gen_tokens / results["full_gen_s"]

            # decode final output (full)
            try:
                output_text = self.tokenizer.batch_decode(full_ids, skip_special_tokens=True)[0].strip()
            except Exception:
                try:
                    output_text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
                except Exception:
                    output_text = str(full_ids)
            # Normalize whitespace and truncate preview to 20 chars
            if output_text is None:
                output_text = ""
            output_preview = " ".join(output_text.splitlines())
            output_preview = output_preview.replace("\t", " ")
            output_preview = output_preview[:200]
            results["output_preview"] = output_preview
            # Optionally keep full text (comment out if CSV should not include full text)
            results["output_full"] = output_text
        except Exception as e:
            results["full_gen_s"] = None
            results["tokens_generated"] = None
            results["tokens_per_sec"] = None
            results["output_preview"] = ""

        # 4) Peak GPU memory
        try:
            if self.device == "cuda":
                results["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                results["peak_gpu_mem_gb"] = None
        except Exception:
            results["peak_gpu_mem_gb"] = None

        return results


def iter_images_and_bench(model_path, images_dir, out_csv, model_base=None,
                          conv_mode="qwen_2", max_new_tokens=256, glob="*.png",
                          preprocess: bool = False, preprocess_config: str = 'preprocess_config.yaml'):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # load preprocess config if requested
    pre_cfg = None
    if preprocess:
        pre_cfg = load_config(preprocess_config)

    runner = VLMRunner(model_path=model_path, model_base=model_base, conv_mode=conv_mode, device="cuda")

    img_paths = sorted([p for p in images_dir.rglob(glob)])
    if not img_paths:
        # Try more extensions if nothing found
        img_paths = sorted([p for p in images_dir.rglob("*.jpg")] + [p for p in images_dir.rglob("*.jpeg")])
    if not img_paths:
        raise RuntimeError(f"No images found in {images_dir} with patterns '{glob}', *.jpg, *.jpeg")

    # Prepare CSV
    out_csv = Path(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "filename", "encoder_time_s", "ttft_s", "full_gen_s",
            "tokens_generated", "tokens_per_sec", "peak_gpu_mem_gb",
            "output_preview", 
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # temp dir for preprocessed images
        tmp_dir = images_dir / '.preprocessed'
        if preprocess:
            tmp_dir.mkdir(exist_ok=True)

        for idx, img in enumerate(img_paths, start=1):
            print(f"[{idx}/{len(img_paths)}] Running inference on {img.name} ...")
            try:
                run_img = img
                if preprocess:
                    # preprocess_image accepts path and returns (PIL.Image, meta)
                    proc_img, meta = preprocess_image(str(img), pre_cfg)
                    tmp_path = tmp_dir / img.name
                    proc_img.save(tmp_path)
                    run_img = tmp_path

                metrics = runner.run_once(run_img, prompt="Describe the image.", max_new_tokens=max_new_tokens)
            except Exception as e:
                # If runner.run_once itself fails, record an error row
                print(f"  ERROR loading/running on {img.name}: {e}")
                metrics = {
                    "filename": img.name, "encoder_time_s": None, "ttft_s": None, "full_gen_s": None,
                    "tokens_generated": None, "tokens_per_sec": None, "peak_gpu_mem_gb": None,
                    "output_preview": ""
                }
            # ensure all fields present and write row
            row = {k: metrics.get(k, "") for k in fieldnames}
            writer.writerow(row)
            csvfile.flush()

    print(f"Benchmark finished. Results written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a folder of images with FastVLM and write CSV.")
    parser.add_argument("--model-path", type=str, default="./checkpoints/llava-fastvithd_0.5b_stage3",
                        help="Path to model checkpoint directory (default: ./checkpoints/llava-fastvithd_0.5b_stage3)")
    parser.add_argument("--model-base", type=str, default=None, help="Optional model base")
    parser.add_argument("--images-dir", type=str, default="./images",
                        help="Directory containing images (recursively searched). Default: ./images")
    parser.add_argument("--out-csv", type=str, default="results.csv", help="Output CSV file path (default: results.csv)")
    parser.add_argument("--conv-mode", type=str, default="qwen_2", help="Conversation mode (repo conv_templates key)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for full generation timing")
    parser.add_argument("--glob", type=str, default="*.png", help="Glob pattern to search for images (default: *.png)")
    args = parser.parse_args()

    iter_images_and_bench(
        model_path=args.model_path,
        images_dir=args.images_dir,
        out_csv=args.out_csv,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        max_new_tokens=args.max_new_tokens,
        glob=args.glob
    )
