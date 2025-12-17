"""End-to-end small-scale DocVQA evaluation.

This script:
- Loads a small number of examples from HuggingFace `docvqa` (default validation split)
- Runs baseline inference (raw images) and preprocessed inference (uses `preprocess_image`)
- Computes exact-match accuracy over provided references (simple normalized match)
- Records timing/metrics and writes a CSV and plots

Notes:
- Requires a GPU and a compatible checkpoint for real inference (`--model-path`).
- If you don't have a model available, run with `--dry-run` to exercise the pipeline without calling the model.
"""
import argparse
from pathlib import Path
import csv
import time
import re
import math

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
except Exception:
    pd = None

from bench_dir import VLMRunner
from preprocess_image import preprocess_image, load_config


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, refs) -> int:
    npred = normalize_answer(pred)
    for r in refs:
        if npred == normalize_answer(r):
            return 1
    return 0


def extract_refs(example: dict):
    # DocVQA variant support: try common fields
    if 'answers' in example:
        vals = example['answers']
        # answers might be list of dicts or list of strings
        refs = []
        for a in vals:
            if isinstance(a, dict):
                if 'text' in a:
                    refs.append(a['text'])
                elif 'answer' in a:
                    refs.append(a['answer'])
            elif isinstance(a, str):
                refs.append(a)
        return refs
    if 'answers_text' in example:
        return example['answers_text']
    if 'gt_answers' in example:
        return example['gt_answers']
    # fallback: any field named 'answer' or 'text'
    for k in ('answer','text','gt'):
        if k in example:
            v = example[k]
            if isinstance(v, (list,tuple)):
                return list(v)
            return [str(v)]
    return []


def run(args):
    if load_dataset is None:
        raise RuntimeError('datasets library not installed; pip install datasets')
    if pd is None:
        raise RuntimeError('pandas/matplotlib/seaborn are required; pip install -r requirements.txt')

    ds = load_dataset("nielsr/docvqa_1200_examples_donut")
    split = args.split
    if split not in ds:
        # try smaller variants
        if 'validation' in ds:
            split = 'validation'
        else:
            split = list(ds.keys())[0]
    data = ds[split]

    total = min(len(data), args.limit) if args.limit and args.limit > 0 else len(data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = None
    if not args.dry_run:
        runner = VLMRunner(model_path=args.model_path, model_base=args.model_base, conv_mode=args.conv_mode, device='cuda' if args.use_cuda else 'cpu')

    pre_cfg = load_config(args.preprocess_config) if args.preprocess else None

    import numpy as np
    try:
        from skimage.metrics import structural_similarity as ssim
        HAVE_SSIM = True
    except Exception:
        HAVE_SSIM = False

    def token_count_for_resolution(res: int, patch_size: int) -> int:
        return max(1, (res // patch_size)) * max(1, (res // patch_size))

    def image_to_square(img_path: Path, target: int):
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        scale = target / float(max(w, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        out = Image.new('RGB', (target, target), (0, 0, 0))
        out.paste(resized, ((target - new_w) // 2, (target - new_h) // 2))
        return out

    def compute_quality(baseline_img, proc_img):
        a = np.array(baseline_img.convert('L'), dtype=np.float32)
        b = np.array(proc_img.convert('L'), dtype=np.float32)
        if HAVE_SSIM:
            try:
                return float(ssim(a, b, data_range=b.max() - b.min()))
            except Exception:
                pass
        mse = float(((a - b) ** 2).mean())
        mse_norm = mse / (255.0 ** 2)
        return 1.0 / (1.0 + mse_norm)

    rows = []
    for i in range(total):
        ex = data[i]
        # images field may be an Image object or dict with 'image'
        image = ex.get('image') or ex.get('img') or ex.get('image_path')
        # save image to a temp path
        img_path = out_dir / f"img_{i}.png"
        try:
            if hasattr(image, 'save'):
                image.save(img_path)
            elif isinstance(image, str):
                # path
                # copy reference
                p = Path(image)
                if p.exists():
                    img_path.write_bytes(p.read_bytes())
                else:
                    # dataset may store bytes
                    img_path.write_bytes(image.encode('utf-8'))
            else:
                # try dataset-provided PIL via to_pil
                try:
                    im = ex['image'].to_pil() if hasattr(ex['image'], 'to_pil') else None
                    if im:
                        im.save(img_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"Skipping example {i}: cannot save image: {e}")
            continue

        # Normalize question field: dataset may store question as a dict (e.g., multilingual)
        def _extract_question(example):
            q = example.get('question') or example.get('query') or example.get('caption') or ''
            if isinstance(q, dict):
                # prefer common keys
                for k in ('text', 'question', 'query', 'content'):
                    if k in q and isinstance(q[k], str):
                        return q[k]
                # take first string value
                for v in q.values():
                    if isinstance(v, str):
                        return v
                return str(q)
            return q if isinstance(q, str) else str(q)

        question = _extract_question(ex)
        # language instruction
        if args.answer_lang and args.answer_lang.lower() == 'de':
            prompt = "Bitte antworte auf Deutsch: " + question
        else:
            prompt = question
        refs = extract_refs(ex)

        # baseline inference
        baseline_metrics = {}
        baseline_pred = ''
        if args.dry_run:
            baseline_metrics = {'ttft_s': 0.0, 'full_gen_s': 0.0}
            baseline_pred = 'DRYRUN'
        else:
            m = runner.run_once(img_path, prompt=prompt, max_new_tokens=args.max_new_tokens)
            baseline_metrics = m
            baseline_pred = m.get('output_full','')

        # preprocessed inference
        pre_metrics = {}
        pre_pred = ''
        preproc_time = 0.0
        selected_resolution = args.baseline_res
        if args.preprocess:
            proc_img, meta = preprocess_image(str(img_path), pre_cfg)
            tmp = out_dir / '.pre' / f"img_{i}.png"
            tmp.parent.mkdir(exist_ok=True)
            proc_img.save(tmp)
            if args.dry_run:
                pre_metrics = {'ttft_s': 0.0, 'full_gen_s': 0.0}
                pre_pred = 'DRYRUN'
            else:
                # use same language prefixed prompt for preprocessed run
                m2 = runner.run_once(tmp, prompt=prompt, max_new_tokens=args.max_new_tokens)
                pre_metrics = m2
                pre_pred = m2.get('output_full','')
            preproc_time = float(meta.get('processing_time_s', 0.0))
            selected_resolution = int(meta.get('selected_resolution', args.baseline_res))
        else:
            # no preprocess: after tokens equal baseline
            selected_resolution = args.baseline_res

        em_baseline = exact_match(baseline_pred, refs) if refs else None
        em_pre = exact_match(pre_pred, refs) if args.preprocess and refs else None

        # token counts and simulated inference times
        baseline_tokens = token_count_for_resolution(args.baseline_res, args.patch_size)
        after_tokens = token_count_for_resolution(selected_resolution, args.patch_size)
        token_reduction = 1.0 - (after_tokens / float(baseline_tokens)) if baseline_tokens>0 else 0.0
        baseline_infer = baseline_tokens * args.token_time
        after_infer = after_tokens * args.token_time

        # quality: compare baseline-resized original vs processed (if preprocessed)
        try:
            baseline_img_sq = image_to_square(img_path, args.baseline_res)
            if args.preprocess:
                proc_for_q = proc_img.resize(baseline_img_sq.size)
                quality = compute_quality(baseline_img_sq, proc_for_q)
            else:
                quality = 1.0
        except Exception:
            quality = None

        row = {
            'index': i,
            'image': img_path.name,
            'question': question,
            'refs': refs,
            'baseline_pred': baseline_pred,
            'pre_pred': pre_pred,
            'em_baseline': em_baseline,
            'em_pre': em_pre,
            'baseline_full_gen_s': baseline_metrics.get('full_gen_s') if baseline_metrics else None,
            'pre_full_gen_s': pre_metrics.get('full_gen_s') if pre_metrics else None,
            'baseline_ttft_s': baseline_metrics.get('ttft_s') if baseline_metrics else None,
            'pre_ttft_s': pre_metrics.get('ttft_s') if pre_metrics else None,
            'original_size': f"{img_path.stat().st_size}",
            'preproc_time_s': round(preproc_time, 6),
            'baseline_tokens': int(baseline_tokens),
            'after_tokens': int(after_tokens),
            'token_reduction': round(token_reduction, 6),
            'baseline_infer_s': round(baseline_infer, 6),
            'after_infer_s': round(after_infer, 6),
            'total_before_s': round(baseline_infer, 6),
            'total_after_s': round(preproc_time + after_infer, 6),
            'quality_score': round(float(quality), 6) if quality is not None else None,
        }
        rows.append(row)
        print(f"[{i+1}/{total}] Q={question[:40]}... EM baseline={em_baseline} EM pre={em_pre}")

    # write CSV in validation_small-like format and full results
    csv_path = out_dir / args.out_csv
    full_csv = out_dir / ('full_' + args.out_csv)
    # full CSV with detailed rows
    with full_csv.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['index','image','question','refs','baseline_pred','pre_pred','em_baseline','em_pre','baseline_full_gen_s','pre_full_gen_s','baseline_ttft_s','pre_ttft_s','original_size','preproc_time_s','baseline_tokens','after_tokens','token_reduction','baseline_infer_s','after_infer_s','total_before_s','total_after_s','quality_score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r2 = r.copy()
            r2['refs'] = '|'.join([str(x) for x in r2['refs']])
            writer.writerow(r2)

    # validation_small style summary CSV
    val_csv = out_dir / 'validation_docvqa.csv'
    with val_csv.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['image','original_size','preproc_time_s','baseline_tokens','after_tokens','token_reduction','baseline_infer_s','after_infer_s','total_before_s','total_after_s','quality_score']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                'image': r['image'],
                'original_size': r.get('original_size',''),
                'preproc_time_s': r.get('preproc_time_s',''),
                'baseline_tokens': r.get('baseline_tokens',''),
                'after_tokens': r.get('after_tokens',''),
                'token_reduction': r.get('token_reduction',''),
                'baseline_infer_s': r.get('baseline_infer_s',''),
                'after_infer_s': r.get('after_infer_s',''),
                'total_before_s': r.get('total_before_s',''),
                'total_after_s': r.get('total_after_s',''),
                'quality_score': r.get('quality_score',''),
            })
        # add summary row
        if rows:
            avg_pre = sum([float(r.get('preproc_time_s',0) or 0) for r in rows]) / len(rows)
            sum_before = sum([int(r.get('baseline_tokens',0) or 0) for r in rows])
            sum_after = sum([int(r.get('after_tokens',0) or 0) for r in rows])
            avg_quality = None
            quals = [r.get('quality_score') for r in rows if r.get('quality_score') is not None]
            if quals:
                avg_quality = sum([float(q) for q in quals]) / len(quals)
            w.writerow({
                'image':'__SUMMARY__',
                'original_size':'',
                'preproc_time_s': round(avg_pre,6),
                'baseline_tokens': sum_before,
                'after_tokens': sum_after,
                'token_reduction': round(1.0 - (sum_after / float(sum_before)) if sum_before>0 else 0.0,6),
                'baseline_infer_s':'',
                'after_infer_s':'',
                'total_before_s':'',
                'total_after_s':'',
                'quality_score': round(avg_quality,6) if avg_quality is not None else '',
            })

    print(f'Wrote full results to {full_csv} and validation CSV to {val_csv}')

    # basic summary and plots
    df = pd.DataFrame(rows)
    summary = {}
    if 'em_baseline' in df.columns:
        summary['acc_baseline'] = df['em_baseline'].dropna().astype(int).mean()
    if 'em_pre' in df.columns:
        summary['acc_pre'] = df['em_pre'].dropna().astype(int).mean()
    summary['mean_full_gen_baseline_s'] = df['baseline_full_gen_s'].dropna().mean()
    summary['mean_full_gen_pre_s'] = df['pre_full_gen_s'].dropna().mean()
    print('Summary:', summary)

    # plots
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    try:
        sns.set()
        # Accuracy comparison
        if 'em_baseline' in df.columns and 'em_pre' in df.columns:
            acc_df = pd.DataFrame({'baseline': df['em_baseline'].astype(float), 'pre': df['em_pre'].astype(float)})
            acc_df.mean().plot(kind='bar', title='Exact-match accuracy (baseline vs preprocessed)')
            plt.ylabel('Exact-match')
            plt.tight_layout()
            plt.savefig(plot_dir / 'accuracy_compare.png')
            plt.close()

        # Time comparison
        tdf = df[['baseline_full_gen_s','pre_full_gen_s']].dropna()
        if not tdf.empty:
            tdf_mean = tdf.mean()
            tdf_mean.plot(kind='bar', title='Mean full generation time (s)')
            plt.ylabel('Seconds')
            plt.tight_layout()
            plt.savefig(plot_dir / 'time_compare.png')
            plt.close()

        # Scatter token/time/quality if available
    except Exception as e:
        print('Plotting failed:', e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--model-base', default=None)
    ap.add_argument('--conv-mode', default='qwen_2')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--limit', type=int, default=16)
    ap.add_argument('--out-dir', default='eval_docvqa')
    ap.add_argument('--out-csv', default='docvqa_results.csv')
    ap.add_argument('--preprocess', action='store_true')
    ap.add_argument('--preprocess-config', default='preprocess_config.yaml')
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--dry-run', action='store_true', help='Do not call model; useful for testing pipeline')
    ap.add_argument('--use-cuda', dest='use_cuda', action='store_true')
    ap.add_argument('--answer-lang', default=None, help='Force answer language (e.g., de) by prefixing prompt')
    ap.add_argument('--baseline-res', dest='baseline_res', type=int, default=512, help='Baseline square resolution (pixels)')
    ap.add_argument('--patch-size', dest='patch_size', type=int, default=16, help='Visual patch size for token count')
    ap.add_argument('--token-time', dest='token_time', type=float, default=0.0005, help='Simulated inference time per visual token (s)')
    args = ap.parse_args()
    run(args)
