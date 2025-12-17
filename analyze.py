import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze(csv_path):
    df = pd.read_csv(csv_path)

    print("\n=== Descriptive Statistics ===")
    print(df.describe(include="all"))
    print("\n=== Correlations ===")
    print(df.corr(numeric_only=True))

    # Output directory for graphs
    out_dir = Path("analysis_outputs")
    out_dir.mkdir(exist_ok=True)

    # -------------------------------
    # TTFT Bar Chart
    # -------------------------------
    plt.figure()
    plt.bar(df["filename"], df["ttft_s"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Time To First Token (s)")
    plt.title("TTFT per Image")
    plt.tight_layout()
    plt.savefig(out_dir / "ttft_per_image.png")
    plt.close()

    # -------------------------------
    # Full Generation Time Bar Chart
    # -------------------------------
    plt.figure()
    plt.bar(df["filename"], df["full_gen_s"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Full Generation Time (s)")
    plt.title("Full Generation Time per Image")
    plt.tight_layout()
    plt.savefig(out_dir / "full_gen_per_image.png")
    plt.close()

    # -------------------------------
    # Tokens per Second Bar Chart
    # -------------------------------
    plt.figure()
    plt.bar(df["filename"], df["tokens_per_sec"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Tokens per Second")
    plt.title("Tokens/Second per Image")
    plt.tight_layout()
    plt.savefig(out_dir / "tokens_per_sec.png")
    plt.close()

    # -------------------------------
    # Scatter: Tokens Generated vs Full Gen Time
    # -------------------------------
    plt.figure()
    plt.scatter(df["tokens_generated"], df["full_gen_s"])
    plt.xlabel("Tokens Generated")
    plt.ylabel("Full Generation Time (s)")
    plt.title("Tokens Generated vs Full Generation Time")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_tokens_vs_fullgen.png")
    plt.close()

    # -------------------------------
    # Histogram: TTFT
    # -------------------------------
    plt.figure()
    plt.hist(df["ttft_s"], bins=10)
    plt.xlabel("TTFT (s)")
    plt.ylabel("Count")
    plt.title("TTFT Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_ttft.png")
    plt.close()

    # -------------------------------
    # Histogram: Full Generation Time
    # -------------------------------
    plt.figure()
    plt.hist(df["full_gen_s"], bins=10)
    plt.xlabel("Full Gen Time (s)")
    plt.ylabel("Count")
    plt.title("Full Generation Time Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_full_gen.png")
    plt.close()

    print("\nAll graphs saved to:", out_dir)


if __name__ == "__main__":
    analyze("results.csv")
