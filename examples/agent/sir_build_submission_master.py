"""Tile the three reviewer-facing figures into a single ``submission_master``
PDF+PNG composite for the ICML 2026 AI4Science workshop paper.

Inputs (produced by ``sir_sequential_agent.py`` and ``sir_sbc_postprocess.py``):

* ``<artifacts>/submission_combo.png`` — 4-panel mechanism figure
  (β/γ posterior, σ trajectory, R₀ ridge, design-on-curve).
* ``<artifacts>/boed_vs_uniform_3panel.png`` — BOED vs uniform-grid
  comparison (σ shrinkage, EIG, |bias| boxplot).
* ``<sbc-dir>/sbc_rank_histograms.png`` — SBC rank histograms per
  parameter (β, γ, R₀) with 99% uniform envelope.

Output:

* ``<out>.pdf`` and ``<out>.png`` — a single composite at submission size.

Usage::

    python examples/agent/sir_build_submission_master.py \
        --artifacts artifacts/sir_seq_final \
        --sbc-dir artifacts/sir_sbc \
        --output artifacts/sir_seq_final/submission_master
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing figure: {path}")
    return mpimg.imread(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, required=True,
                        help="Directory holding submission_combo.png and boed_vs_uniform_3panel.png.")
    parser.add_argument("--sbc-dir", type=str, required=True,
                        help="Directory holding sbc_rank_histograms.png.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output stem — writes <stem>.pdf and <stem>.png.")
    parser.add_argument("--title", type=str,
                        default="Sequential BOED on the stochastic SIR (ICML 2026 AI4Science workshop)",
                        help="Super-title placed above the composite.")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    sbc_dir = Path(args.sbc_dir)
    out_stem = Path(args.output)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    img_main = _load(artifacts / "submission_combo.png")
    img_cmp = _load(artifacts / "boed_vs_uniform_3panel.png")
    img_sbc = _load(sbc_dir / "sbc_rank_histograms.png")

    # 3-row layout: main figure on top, comparison middle, SBC bottom.
    # Rows sized by image aspect ratios so we don't distort anything.
    def _ar(img):
        return img.shape[1] / img.shape[0]  # width / height

    ar_main, ar_cmp, ar_sbc = _ar(img_main), _ar(img_cmp), _ar(img_sbc)
    # Choose a common width in inches; compute row heights.
    W = 12.0
    H_main = W / ar_main
    H_cmp = W / ar_cmp
    H_sbc = W / ar_sbc
    title_pad = 0.5
    total_H = H_main + H_cmp + H_sbc + title_pad

    fig = plt.figure(figsize=(W, total_H))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[H_main, H_cmp, H_sbc],
        hspace=0.06, left=0.015, right=0.985, top=1.0 - title_pad / total_H, bottom=0.01,
    )

    for ax, img, caption in [
        (fig.add_subplot(gs[0, 0]), img_main, "(I) Mechanism: posterior contraction and informative designs."),
        (fig.add_subplot(gs[1, 0]), img_cmp, "(II) BOED vs uniform-grid baseline (15 seeds)."),
        (fig.add_subplot(gs[2, 0]), img_sbc, "(III) Simulation-based calibration (Talts et al. 2018)."),
    ]:
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.01, 0.985, caption, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left",
            color="#333333",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.25"),
        )

    fig.suptitle(args.title, fontsize=12, y=0.998)

    pdf_path = out_stem.with_suffix(".pdf")
    png_path = out_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[master] -> {pdf_path}")
    print(f"[master] -> {png_path}")


if __name__ == "__main__":
    main()
