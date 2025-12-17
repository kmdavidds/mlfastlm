"""Generate validation plots from validation_small.csv

Produces PNGs in an output directory (default: analysis/validation_plots):
- token_reduction.png
- inference_times.png
- quality_scores.png
- token_vs_quality.png
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # drop summary row if present
    df = df[df['image'] != '__SUMMARY__'].copy()
    # coerce numeric
    for col in ['preproc_time_s','baseline_tokens','after_tokens','token_reduction','baseline_infer_s','after_infer_s','total_before_s','total_after_s','quality_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_token_reduction(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(8,4))
    sns.barplot(x='image', y='token_reduction', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Token reduction (fraction)')
    plt.title('Token Reduction per Image')
    plt.tight_layout()
    plt.savefig(out / 'token_reduction.png')
    plt.close()


def plot_inference_times(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(8,4))
    x = range(len(df))
    plt.bar(x, df['baseline_infer_s'], width=0.4, label='baseline')
    plt.bar([i+0.4 for i in x], df['after_infer_s'], width=0.4, label='after')
    plt.xticks([i+0.2 for i in x], df['image'], rotation=45, ha='right')
    plt.ylabel('Simulated inference time (s)')
    plt.title('Inference Time Before vs After')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'inference_times.png')
    plt.close()


def plot_quality(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(8,4))
    sns.barplot(x='image', y='quality_score', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Quality score (SSIM or fallback)')
    plt.title('Quality Score per Image')
    plt.tight_layout()
    plt.savefig(out / 'quality_scores.png')
    plt.close()


def plot_tokens_vs_quality(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='after_tokens', y='quality_score', data=df)
    plt.xlabel('After tokens')
    plt.ylabel('Quality score')
    plt.title('Tokens vs Quality')
    plt.tight_layout()
    plt.savefig(out / 'token_vs_quality.png')
    plt.close()


def main(csv_path: str, out_dir: str):
    p = Path(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_csv(p)
    if df.empty:
        print('No data to plot')
        return
    plot_token_reduction(df, out)
    plot_inference_times(df, out)
    plot_quality(df, out)
    plot_tokens_vs_quality(df, out)
    print(f'Wrote plots to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='validation_small.csv')
    ap.add_argument('--out', default='analysis/validation_plots')
    args = ap.parse_args()
    main(args.csv, args.out)
