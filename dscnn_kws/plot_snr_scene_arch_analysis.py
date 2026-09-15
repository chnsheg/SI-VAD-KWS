from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        "Install plotting dependencies first, for example:\n"
        "  pip install pandas matplotlib\n"
        "or, from the project root:\n"
        "  pip install -r dscnn_kws/requirements.txt"
    ) from exc


SCENE_ORDER = [
    "shopping_mall",
    "street_traffic",
    "airport",
    "street_pedestrian",
    "metro_station",
    "public_square",
    "park",
    "metro",
    "tram",
    "bus",
]

SNR_ORDER = [20.0, 10.0, 5.0, 0.0, -5.0]

DEFAULT_SCENE_ARCH = "L5_C64"

TEXT = {
    "en": {
        "scene_snr_title": "{arch} accuracy vs. SNR across noise scenes",
        "scene_snr_heatmap_title": "{arch} accuracy heatmap by scene and SNR",
        "arch_snr_title": "Accuracy vs. SNR across network scales",
        "param_tradeoff_title": "Model size vs. mean and worst-case accuracy",
        "minus5_scene_bar_title": "Mean accuracy by scene under -5 dB noise",
        "arch_scene_heatmap_title": "{snr:g} dB architecture-scene accuracy heatmap",
        "mean_acc": "Mean accuracy",
        "worst_acc": "Worst-case accuracy",
        "arch_legend": "Architecture",
        "params": "Expected parameters",
    },
    "zh": {
        "scene_snr_title": "不同噪声场景下准确率随 SNR 的变化",
        "scene_snr_heatmap_title": "场景 × SNR 平均准确率热力图",
        "arch_snr_title": "不同网络规模下准确率随 SNR 的变化",
        "param_tradeoff_title": "模型参数量与平均/最坏工况准确率的关系",
        "minus5_scene_bar_title": "-5 dB 强噪声下各场景平均准确率",
        "arch_scene_heatmap_title": "{snr:g} dB 下架构 × 场景准确率热力图",
        "mean_acc": "平均准确率",
        "worst_acc": "最坏工况准确率",
        "arch_legend": "Architecture",
        "params": "Expected parameters",
    },
}


def text(lang: str, key: str, **kwargs: object) -> str:
    return TEXT[lang][key].format(**kwargs)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.unicode_minus": False,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
        }
    )


def dataset_short_name(name: str) -> str:
    if "hi_xiaowen" in name:
        return "hi_xiaowen"
    if "nihao_wenwen" in name:
        return "nihao_wenwen"
    return name


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def filter_arch(grid: pd.DataFrame, arch: str) -> pd.DataFrame:
    if arch not in set(grid["arch"]):
        available = ", ".join(sorted(grid["arch"].unique()))
        raise ValueError(f"Architecture {arch!r} was not found. Available architectures: {available}")
    return grid[grid["arch"] == arch].copy()


def ordered_architectures(df: pd.DataFrame) -> list[str]:
    arch_meta = (
        df.groupby("arch", as_index=False)
        .agg(
            expected_params=("expected_params", "min"),
            layers=("layers", "min"),
            channels=("channels", "min"),
        )
        .sort_values(["expected_params", "layers", "channels", "arch"], kind="mergesort")
    )
    return arch_meta["arch"].tolist()


def plot_scene_snr_curves(grid: pd.DataFrame, out_dir: Path, lang: str, arch: str) -> None:
    """Mean accuracy vs SNR for every noise scene."""
    grid = filter_arch(grid, arch)
    df = (
        grid.groupby(["scene", "snr_db"], as_index=False)["acc"]
        .mean()
        .assign(acc_pct=lambda x: x["acc"] * 100)
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    cmap = plt.get_cmap("tab10")
    for i, scene in enumerate(SCENE_ORDER):
        sub = df[df["scene"] == scene].set_index("snr_db").reindex(SNR_ORDER).reset_index()
        ax.plot(
            sub["snr_db"],
            sub["acc_pct"],
            marker="o",
            linewidth=1.9,
            markersize=4.2,
            color=cmap(i % 10),
            label=scene,
        )

    ax.set_title(text(lang, "scene_snr_title", arch=arch))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(SNR_ORDER)
    ax.invert_xaxis()
    ax.set_ylim(80, 100)
    ax.legend(ncol=2, frameon=True, loc="lower left")
    save_figure(fig, out_dir, f"01_{arch}_scene_snr_accuracy_curves")


def plot_scene_snr_heatmap(grid: pd.DataFrame, out_dir: Path, lang: str, arch: str) -> None:
    """Heatmap of mean accuracy by scene and SNR."""
    grid = filter_arch(grid, arch)
    pivot = (
        grid.pivot_table(index="scene", columns="snr_db", values="acc", aggfunc="mean")
        .reindex(index=SCENE_ORDER, columns=SNR_ORDER)
        * 100
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=80, vmax=99)

    ax.set_title(text(lang, "scene_snr_heatmap_title", arch=arch))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Noise scene")
    ax.set_xticks(np.arange(len(SNR_ORDER)), [str(int(x)) for x in SNR_ORDER])
    ax.set_yticks(np.arange(len(SCENE_ORDER)), SCENE_ORDER)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            color = "white" if val < 88 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Accuracy (%)")
    save_figure(fig, out_dir, f"02_{arch}_scene_snr_accuracy_heatmap")


def plot_arch_snr_curves(grid: pd.DataFrame, out_dir: Path, lang: str) -> None:
    """Accuracy vs SNR for every network scale."""
    selected_archs = ordered_architectures(grid)
    datasets = list(grid["dataset"].drop_duplicates())

    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 5.4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab20", max(len(selected_archs), 1))
    for ax, dataset in zip(axes, datasets):
        d = grid[grid["dataset"] == dataset]
        summary = (
            d[d["arch"].isin(selected_archs)]
            .groupby(["arch", "snr_db"], as_index=False)["acc"]
            .mean()
            .assign(acc_pct=lambda x: x["acc"] * 100)
        )
        for i, arch in enumerate(selected_archs):
            sub = summary[summary["arch"] == arch].set_index("snr_db").reindex(SNR_ORDER).reset_index()
            params = int(d.loc[d["arch"] == arch, "expected_params"].iloc[0])
            ax.plot(
                sub["snr_db"],
                sub["acc_pct"],
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                color=cmap(i),
                label=f"{arch} ({params})",
            )

        ax.set_title(dataset_short_name(dataset))
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNR_ORDER)
        ax.invert_xaxis()
        ax.set_ylim(80, 100)
        ax.legend(title=text(lang, "arch_legend"), frameon=True, ncol=2)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(text(lang, "arch_snr_title"), y=1.02)
    save_figure(fig, out_dir, "03_arch_snr_accuracy_curves")


def plot_arch_param_tradeoff(arch_summary: pd.DataFrame, out_dir: Path, lang: str) -> None:
    """Mean and worst-case accuracy vs number of parameters."""
    datasets = list(arch_summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        d = arch_summary[arch_summary["dataset"] == dataset].sort_values("expected_params")
        ax.plot(
            d["expected_params"],
            d["mean_acc"] * 100,
            marker="o",
            linewidth=2.2,
            label=text(lang, "mean_acc"),
        )
        ax.plot(
            d["expected_params"],
            d["min_acc"] * 100,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label=text(lang, "worst_acc"),
        )

        for _, row in d.iterrows():
            ax.annotate(
                row["arch"],
                (row["expected_params"], row["mean_acc"] * 100),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7.5,
            )

        ax.set_title(dataset_short_name(dataset))
        ax.set_xlabel(text(lang, "params"))
        ax.set_xscale("log")
        ax.set_ylim(74, 100)
        ax.legend(frameon=True)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(text(lang, "param_tradeoff_title"), y=1.02)
    save_figure(fig, out_dir, "04_param_accuracy_tradeoff")


def plot_minus5_scene_bar(grid: pd.DataFrame, out_dir: Path, lang: str) -> None:
    """Scene difficulty under the hardest SNR condition."""
    d = (
        grid[grid["snr_db"] == -5.0]
        .groupby(["dataset", "scene"], as_index=False)["acc"]
        .mean()
        .assign(acc_pct=lambda x: x["acc"] * 100)
    )
    datasets = list(d["dataset"].drop_duplicates())
    x = np.arange(len(SCENE_ORDER))
    width = 0.38 if len(datasets) > 1 else 0.6

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, dataset in enumerate(datasets):
        sub = d[d["dataset"] == dataset].set_index("scene").reindex(SCENE_ORDER)
        offset = (i - (len(datasets) - 1) / 2) * width
        ax.bar(x + offset, sub["acc_pct"], width=width, label=dataset_short_name(dataset))

    ax.set_title(text(lang, "minus5_scene_bar_title"))
    ax.set_xlabel("Noise scene")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, SCENE_ORDER, rotation=35, ha="right")
    ax.set_ylim(75, 96)
    ax.legend(frameon=True)
    save_figure(fig, out_dir, "05_minus5_scene_accuracy_bar")


def plot_arch_scene_heatmap_at_snr(grid: pd.DataFrame, out_dir: Path, lang: str, snr: float = -5.0) -> None:
    """Architecture by scene heatmaps at a chosen SNR, one panel per dataset."""
    arch_order = ordered_architectures(grid)
    datasets = list(grid["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.9 * len(datasets), 6.7), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    ims = []
    for ax, dataset in zip(axes, datasets):
        d = grid[(grid["dataset"] == dataset) & (grid["snr_db"] == snr)]
        pivot = (
            d.pivot_table(index="arch", columns="scene", values="acc", aggfunc="mean")
            .reindex(index=arch_order, columns=SCENE_ORDER)
            * 100
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=75, vmax=96)
        ims.append(im)
        ax.set_title(dataset_short_name(dataset))
        ax.set_xlabel("Noise scene")
        ax.set_xticks(np.arange(len(SCENE_ORDER)), SCENE_ORDER, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(arch_order)), arch_order)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                color = "white" if val < 84 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=6.8)

    axes[0].set_ylabel("Architecture")
    fig.suptitle(text(lang, "arch_scene_heatmap_title", snr=snr), y=1.02)
    cbar = fig.colorbar(ims[0], ax=axes, pad=0.02)
    cbar.set_label("Accuracy (%)")
    save_figure(fig, out_dir, f"06_arch_scene_heatmap_snr_{snr:g}db".replace("-", "minus"))


def find_csv_dir(csv_dir: Path | None) -> Path:
    required_files = [
        "snr_scene_arch_sweep_grid_results.csv",
        "snr_scene_arch_sweep_arch_summary.csv",
    ]

    if csv_dir is not None:
        candidates = [csv_dir]
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            Path.cwd(),
            script_dir,
            script_dir.parent,
            Path.cwd() / "dscnn_kws",
        ]

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if all((candidate / name).exists() for name in required_files):
            return candidate

    searched = "\n  ".join(str(p) for p in unique_candidates)
    missing = ", ".join(required_files)
    raise FileNotFoundError(
        "Could not find required CSV files.\n"
        f"Required: {missing}\n"
        f"Searched:\n  {searched}\n"
        "You can also pass the directory explicitly, for example:\n"
        "  python dscnn_kws/plot_snr_scene_arch_analysis.py --csv-dir /root/kws/dscnn_kws"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DSCNN SNR/scene/architecture analysis figures.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory containing snr_scene_arch_sweep_*.csv. Defaults to auto-detection.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures_snr_scene_arch")
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Figure title/label language. Default is English to avoid missing CJK font boxes on servers.",
    )
    parser.add_argument(
        "--scene-arch",
        default=DEFAULT_SCENE_ARCH,
        help="Architecture used for figure 01 and 02 scene/SNR plots.",
    )
    args = parser.parse_args()

    setup_matplotlib()

    csv_dir = find_csv_dir(args.csv_dir)
    grid_path = csv_dir / "snr_scene_arch_sweep_grid_results.csv"
    arch_path = csv_dir / "snr_scene_arch_sweep_arch_summary.csv"

    grid = pd.read_csv(grid_path)
    arch_summary = pd.read_csv(arch_path)

    plot_scene_snr_curves(grid, args.out_dir, args.lang, args.scene_arch)
    plot_scene_snr_heatmap(grid, args.out_dir, args.lang, args.scene_arch)
    plot_arch_snr_curves(grid, args.out_dir, args.lang)
    plot_arch_param_tradeoff(arch_summary, args.out_dir, args.lang)
    plot_minus5_scene_bar(grid, args.out_dir, args.lang)
    for snr in [-5.0, 0.0, 5.0, 10.0, 20.0]:
        plot_arch_scene_heatmap_at_snr(grid, args.out_dir, args.lang, snr=snr)

    print(f"CSV files loaded from: {csv_dir}")
    print(f"Figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
