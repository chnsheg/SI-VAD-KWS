from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(".")
OUT_DIR = ROOT / "dscnn_kws" / "streaming" / "results" / "compare_1"

CRNN_CLEAN_CSV = ROOT / "dscnn_kws" / "streaming" / "results" / "streaming_crnn_clean_sweep_results.csv"
CRNN_NOISE_GRID_CSV = (
    ROOT / "dscnn_kws" / "streaming" / "results" / "streaming_crnn_noise_snr_scene_sweep_grid_results.csv"
)
DSCNN_CLEAN_XLSX = ROOT / "sweep_dscnn_acc_results.xlsx"
DSCNN_NOISE_GRID_CSV = ROOT / "snr_scene_arch_sweep_grid_results.csv"

L5_C64_MACS_PER_WINDOW = 2_167_424
UPDATES_PER_SECOND_32MS = 31.25
PLOT_DPI = 180


def ensure_inputs() -> None:
    required = [CRNN_CLEAN_CSV, CRNN_NOISE_GRID_CSV, DSCNN_CLEAN_XLSX, DSCNN_NOISE_GRID_CSV]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(str(p) for p in missing))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crnn_clean = pd.read_csv(CRNN_CLEAN_CSV)
    crnn_noise = pd.read_csv(CRNN_NOISE_GRID_CSV)
    dscnn_clean = pd.read_excel(DSCNN_CLEAN_XLSX)
    dscnn_noise_grid = pd.read_csv(DSCNN_NOISE_GRID_CSV)
    return crnn_clean, crnn_noise, dscnn_clean, dscnn_noise_grid


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def clean_comparison(crnn_clean: pd.DataFrame, dscnn_clean: pd.DataFrame) -> pd.DataFrame:
    l5 = (
        dscnn_clean[dscnn_clean["arch"].eq("L5_C64")]
        [
            [
                "dataset",
                "arch",
                "expected_params",
                "printed_params",
                "best_valid_acc",
                "test_loss",
                "test_acc",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .rename(
            columns={
                "arch": "baseline_arch",
                "expected_params": "l5_params",
                "printed_params": "l5_printed_params",
                "best_valid_acc": "l5_best_valid_acc",
                "test_loss": "l5_test_loss",
                "test_acc": "l5_clean_acc",
                "precision": "l5_clean_precision",
                "recall": "l5_clean_recall",
                "f1": "l5_clean_f1",
            }
        )
    )

    cols = [
        "dataset",
        "arch",
        "cnn_channels",
        "gru_hidden",
        "expected_params",
        "expected_macs_window",
        "expected_macs_per_frame",
        "best_valid_acc",
        "test_loss",
        "test_acc",
        "precision",
        "recall",
        "f1",
    ]
    out = crnn_clean[cols].merge(l5, on="dataset", how="left")
    out["clean_acc_delta_vs_l5"] = out["test_acc"] - out["l5_clean_acc"]
    out["clean_f1_delta_vs_l5"] = out["f1"] - out["l5_clean_f1"]
    out["param_delta_vs_l5"] = out["expected_params"] - out["l5_params"]
    out["param_ratio_vs_l5"] = out["expected_params"] / out["l5_params"]
    out["param_reduction_vs_l5_pct"] = (1.0 - out["param_ratio_vs_l5"]) * 100.0
    return out.sort_values(["dataset", "expected_params", "arch"]).reset_index(drop=True)


def l5_noise_grid(dscnn_noise_grid: pd.DataFrame) -> pd.DataFrame:
    l5 = dscnn_noise_grid[dscnn_noise_grid["arch"].eq("L5_C64")].copy()
    if l5.empty:
        raise ValueError(f"No L5_C64 rows found in {DSCNN_NOISE_GRID_CSV}")
    return l5.reset_index(drop=True)


def l5_noise_summary(l5_grid: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset"]
    summary = (
        l5_grid.groupby(group_cols)
        .agg(
            l5_params=("expected_params", "first"),
            l5_mean_grid_acc=("acc", "mean"),
            l5_median_grid_acc=("acc", "median"),
            l5_min_grid_acc=("acc", "min"),
            l5_max_grid_acc=("acc", "max"),
            l5_std_grid_acc=("acc", "std"),
            l5_mean_grid_f1=("f1", "mean"),
            l5_min_grid_f1=("f1", "min"),
            l5_num_grid_points=("acc", "size"),
        )
        .reset_index()
    )
    acc_5db = (
        l5_grid[l5_grid["snr_db"].eq(5.0)]
        .groupby(group_cols)
        .agg(l5_mean_acc_at_5db=("acc", "mean"), l5_min_acc_at_5db=("acc", "min"), l5_mean_f1_at_5db=("f1", "mean"))
        .reset_index()
    )
    low = (
        l5_grid[l5_grid["snr_db"].isin([-5.0, 0.0])]
        .groupby(group_cols)
        .agg(l5_mean_acc_low_snr=("acc", "mean"), l5_min_acc_low_snr=("acc", "min"), l5_mean_f1_low_snr=("f1", "mean"))
        .reset_index()
    )
    high = (
        l5_grid[l5_grid["snr_db"].isin([10.0, 20.0])]
        .groupby(group_cols)
        .agg(l5_mean_acc_high_snr=("acc", "mean"), l5_mean_f1_high_snr=("f1", "mean"))
        .reset_index()
    )
    return summary.merge(acc_5db, on=group_cols, how="left").merge(low, on=group_cols, how="left").merge(
        high, on=group_cols, how="left"
    )


def noise_cell_comparison(crnn_noise: pd.DataFrame, l5_grid: pd.DataFrame) -> pd.DataFrame:
    l5_cols = [
        "dataset",
        "scene",
        "snr_db",
        "expected_params",
        "acc",
        "precision",
        "recall",
        "f1",
        "num_samples",
        "ckpt",
    ]
    l5 = l5_grid[l5_cols].rename(
        columns={
            "expected_params": "l5_params",
            "acc": "l5_acc",
            "precision": "l5_precision",
            "recall": "l5_recall",
            "f1": "l5_f1",
            "num_samples": "l5_num_samples",
            "ckpt": "l5_ckpt",
        }
    )
    out = crnn_noise.merge(l5, on=["dataset", "scene", "snr_db"], how="left", validate="many_to_one")
    if out["l5_acc"].isna().any():
        missing = out[out["l5_acc"].isna()][["dataset", "scene", "snr_db"]].drop_duplicates()
        raise ValueError("Missing matched L5_C64 cells:\n" + missing.to_string(index=False))
    out["acc_delta_cell_vs_l5"] = out["acc"] - out["l5_acc"]
    out["precision_delta_cell_vs_l5"] = out["precision"] - out["l5_precision"]
    out["recall_delta_cell_vs_l5"] = out["recall"] - out["l5_recall"]
    out["f1_delta_cell_vs_l5"] = out["f1"] - out["l5_f1"]
    out["param_delta_vs_l5"] = out["expected_params"] - out["l5_params"]
    out["param_ratio_vs_l5"] = out["expected_params"] / out["l5_params"]
    out["param_reduction_vs_l5_pct"] = (1.0 - out["param_ratio_vs_l5"]) * 100.0
    return out.sort_values(["dataset", "arch", "scene", "snr_db"], ascending=[True, True, True, False]).reset_index(drop=True)


def noise_summaries(cell_comp: pd.DataFrame, l5_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["dataset", "arch", "cnn_channels", "gru_hidden", "expected_params"]
    arch = (
        cell_comp.groupby(group_cols)
        .agg(
            mean_grid_acc=("acc", "mean"),
            median_grid_acc=("acc", "median"),
            min_grid_acc=("acc", "min"),
            max_grid_acc=("acc", "max"),
            std_grid_acc=("acc", "std"),
            mean_grid_f1=("f1", "mean"),
            min_grid_f1=("f1", "min"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_cell_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "mean"),
            min_cell_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "min"),
            max_cell_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "max"),
            mean_cell_f1_delta_vs_l5=("f1_delta_cell_vs_l5", "mean"),
            num_grid_points=("acc", "size"),
            expected_macs_window=("expected_macs_window", "first"),
            expected_macs_per_frame=("expected_macs_per_frame", "first"),
        )
        .reset_index()
    )

    acc_5db = (
        cell_comp[cell_comp["snr_db"].eq(5.0)]
        .groupby(group_cols)
        .agg(
            mean_acc_at_5db=("acc", "mean"),
            min_acc_at_5db=("acc", "min"),
            mean_f1_at_5db=("f1", "mean"),
            mean_cell_acc_delta_at_5db_vs_l5=("acc_delta_cell_vs_l5", "mean"),
        )
        .reset_index()
    )
    low = (
        cell_comp[cell_comp["snr_db"].isin([-5.0, 0.0])]
        .groupby(group_cols)
        .agg(
            mean_acc_low_snr=("acc", "mean"),
            min_acc_low_snr=("acc", "min"),
            mean_f1_low_snr=("f1", "mean"),
            mean_cell_acc_delta_low_snr_vs_l5=("acc_delta_cell_vs_l5", "mean"),
        )
        .reset_index()
    )
    high = (
        cell_comp[cell_comp["snr_db"].isin([10.0, 20.0])]
        .groupby(group_cols)
        .agg(
            mean_acc_high_snr=("acc", "mean"),
            mean_f1_high_snr=("f1", "mean"),
            mean_cell_acc_delta_high_snr_vs_l5=("acc_delta_cell_vs_l5", "mean"),
        )
        .reset_index()
    )
    arch = arch.merge(acc_5db, on=group_cols, how="left").merge(low, on=group_cols, how="left").merge(
        high, on=group_cols, how="left"
    )
    arch = arch.merge(l5_summary, on="dataset", how="left")
    arch["mean_grid_acc_delta_vs_l5_mean"] = arch["mean_grid_acc"] - arch["l5_mean_grid_acc"]
    arch["mean_acc_5db_delta_vs_l5_5db_mean"] = arch["mean_acc_at_5db"] - arch["l5_mean_acc_at_5db"]
    arch["worst_grid_acc_delta_vs_l5_worst"] = arch["min_grid_acc"] - arch["l5_min_grid_acc"]
    arch["param_ratio_vs_l5"] = arch["expected_params"] / arch["l5_params"]
    arch["param_reduction_vs_l5_pct"] = (1.0 - arch["param_ratio_vs_l5"]) * 100.0
    arch = arch.sort_values(["dataset", "mean_grid_acc"], ascending=[True, False]).reset_index(drop=True)

    snr = (
        cell_comp.groupby(group_cols + ["snr_db"])
        .agg(
            mean_acc=("acc", "mean"),
            min_acc=("acc", "min"),
            max_acc=("acc", "max"),
            mean_f1=("f1", "mean"),
            l5_mean_acc=("l5_acc", "mean"),
            l5_min_acc=("l5_acc", "min"),
            l5_mean_f1=("l5_f1", "mean"),
            mean_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "mean"),
            min_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "min"),
            max_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "arch", "snr_db"], ascending=[True, True, False])
    )

    scene = (
        cell_comp.groupby(group_cols + ["scene"])
        .agg(
            mean_acc=("acc", "mean"),
            min_acc=("acc", "min"),
            max_acc=("acc", "max"),
            mean_f1=("f1", "mean"),
            l5_mean_acc=("l5_acc", "mean"),
            l5_min_acc=("l5_acc", "min"),
            l5_mean_f1=("l5_f1", "mean"),
            mean_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "mean"),
            min_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "min"),
            max_acc_delta_vs_l5=("acc_delta_cell_vs_l5", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "arch", "mean_acc_delta_vs_l5"])
    )
    return arch, snr, scene


def efficiency_comparison(clean_comp: pd.DataFrame) -> pd.DataFrame:
    arch_rows = (
        clean_comp[
            [
                "arch",
                "cnn_channels",
                "gru_hidden",
                "expected_params",
                "expected_macs_window",
                "expected_macs_per_frame",
                "l5_params",
            ]
        ]
        .drop_duplicates("arch")
        .copy()
    )
    arch_rows["model_type"] = "CRNN"
    arch_rows["macs_per_32ms_update"] = arch_rows["expected_macs_per_frame"]
    arch_rows["macs_per_second_if_32ms_update"] = arch_rows["expected_macs_per_frame"] * UPDATES_PER_SECOND_32MS
    arch_rows["macs_per_1s_window"] = arch_rows["expected_macs_window"]
    arch_rows["params"] = arch_rows["expected_params"]

    l5_params = int(clean_comp["l5_params"].dropna().iloc[0])
    l5 = pd.DataFrame(
        [
            {
                "arch": "L5_C64",
                "cnn_channels": "64",
                "gru_hidden": "",
                "expected_params": l5_params,
                "expected_macs_window": L5_C64_MACS_PER_WINDOW,
                "expected_macs_per_frame": L5_C64_MACS_PER_WINDOW,
                "l5_params": l5_params,
                "model_type": "DSCNN",
                "macs_per_32ms_update": L5_C64_MACS_PER_WINDOW,
                "macs_per_second_if_32ms_update": L5_C64_MACS_PER_WINDOW * UPDATES_PER_SECOND_32MS,
                "macs_per_1s_window": L5_C64_MACS_PER_WINDOW,
                "params": l5_params,
            }
        ]
    )
    out = pd.concat([l5, arch_rows], ignore_index=True)
    out["param_ratio_vs_l5"] = out["params"] / l5_params
    out["param_reduction_vs_l5_pct"] = (1.0 - out["param_ratio_vs_l5"]) * 100.0
    out["macs_1s_window_ratio_vs_l5"] = out["macs_per_1s_window"] / L5_C64_MACS_PER_WINDOW
    out["macs_1s_window_reduction_pct"] = (1.0 - out["macs_1s_window_ratio_vs_l5"]) * 100.0
    out["macs_32ms_update_ratio_vs_l5"] = out["macs_per_32ms_update"] / L5_C64_MACS_PER_WINDOW
    out["macs_32ms_update_reduction_pct"] = (1.0 - out["macs_32ms_update_ratio_vs_l5"]) * 100.0
    return out.sort_values(["model_type", "params"], ascending=[False, True]).reset_index(drop=True)


def best_summary(clean_comp: pd.DataFrame, noise_summary: pd.DataFrame, efficiency: pd.DataFrame) -> pd.DataFrame:
    clean_best = clean_comp.loc[clean_comp.groupby("dataset")["test_acc"].idxmax()][
        ["dataset", "arch", "test_acc", "clean_acc_delta_vs_l5", "f1", "expected_params", "param_reduction_vs_l5_pct"]
    ].rename(columns={"arch": "best_clean_crnn_arch", "test_acc": "best_clean_acc", "f1": "best_clean_f1"})
    noise_best_mean = noise_summary.loc[noise_summary.groupby("dataset")["mean_grid_acc"].idxmax()][
        [
            "dataset",
            "arch",
            "mean_grid_acc",
            "mean_grid_acc_delta_vs_l5_mean",
            "mean_cell_acc_delta_vs_l5",
            "mean_acc_at_5db",
            "mean_cell_acc_delta_at_5db_vs_l5",
            "min_grid_acc",
            "worst_grid_acc_delta_vs_l5_worst",
            "l5_mean_grid_acc",
            "l5_min_grid_acc",
        ]
    ].rename(
        columns={
            "arch": "best_noise_mean_crnn_arch",
            "mean_grid_acc": "best_noise_mean_grid_acc",
            "mean_grid_acc_delta_vs_l5_mean": "best_noise_mean_delta_vs_l5_mean",
            "mean_cell_acc_delta_vs_l5": "best_noise_mean_cell_delta_vs_l5",
            "mean_acc_at_5db": "best_noise_mean_acc_at_5db",
            "mean_cell_acc_delta_at_5db_vs_l5": "best_noise_5db_cell_delta_vs_l5",
            "min_grid_acc": "best_noise_mean_arch_worst_acc",
            "worst_grid_acc_delta_vs_l5_worst": "best_noise_worst_delta_vs_l5_worst",
        }
    )
    noise_best_worst = noise_summary.loc[noise_summary.groupby("dataset")["min_grid_acc"].idxmax()][
        ["dataset", "arch", "min_grid_acc", "worst_grid_acc_delta_vs_l5_worst"]
    ].rename(
        columns={
            "arch": "best_worst_case_crnn_arch",
            "min_grid_acc": "best_worst_case_acc",
            "worst_grid_acc_delta_vs_l5_worst": "best_worst_case_delta_vs_l5_worst",
        }
    )
    out = clean_best.merge(noise_best_mean, on="dataset").merge(noise_best_worst, on="dataset")
    eff = efficiency[["arch", "macs_32ms_update_reduction_pct", "macs_1s_window_reduction_pct"]].rename(
        columns={
            "arch": "best_clean_crnn_arch",
            "macs_32ms_update_reduction_pct": "best_clean_arch_macs_32ms_reduction_pct",
            "macs_1s_window_reduction_pct": "best_clean_arch_macs_1s_reduction_pct",
        }
    )
    return out.merge(eff, on="best_clean_crnn_arch", how="left")


def save_excel(tables: dict[str, pd.DataFrame]) -> Path:
    path = OUT_DIR / "crnn_vs_l5c64_comparison_grid_baseline.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
            ws = writer.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 42)
    return path


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_clean_delta(clean_comp: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 6))
    sns.barplot(data=clean_comp, x="arch", y="clean_acc_delta_vs_l5", hue="dataset")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Clean Test Accuracy Delta vs DSCNN L5_C64")
    plt.ylabel("Accuracy delta (CRNN - L5_C64)")
    plt.xlabel("CRNN architecture")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_fig("clean_accuracy_delta_vs_l5c64.png")


def plot_clean_tradeoff(clean_comp: pd.DataFrame, dscnn_clean_l5: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 6))
    sns.scatterplot(data=clean_comp, x="expected_params", y="test_acc", hue="arch", style="dataset", s=140)
    for _, row in dscnn_clean_l5.iterrows():
        plt.scatter(row["expected_params"], row["test_acc"], marker="*", s=260, color="black")
        label = "hi" if "hi_xiaowen" in row["dataset"] else "nihao"
        plt.text(row["expected_params"] + 120, row["test_acc"], f"L5_C64 {label}", fontsize=8)
    plt.title("Clean Accuracy vs Parameter Count")
    plt.xlabel("Parameters")
    plt.ylabel("Clean test accuracy")
    plt.grid(alpha=0.25)
    save_fig("clean_accuracy_params_tradeoff.png")


def plot_noise_snr(noise_snr: pd.DataFrame) -> None:
    datasets = list(noise_snr["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        data = noise_snr[noise_snr["dataset"].eq(dataset)]
        sns.lineplot(data=data, x="snr_db", y="mean_acc", hue="arch", marker="o", ax=ax)
        l5 = data[["snr_db", "l5_mean_acc"]].drop_duplicates().sort_values("snr_db")
        ax.plot(l5["snr_db"], l5["l5_mean_acc"], color="black", linestyle="--", marker="*", label="L5_C64")
        ax.set_title(dataset)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Mean accuracy across scenes")
        ax.grid(alpha=0.25)
        ax.invert_xaxis()
    save_fig("noise_mean_accuracy_by_snr_vs_l5c64_grid.png")


def plot_noise_delta_by_snr(noise_snr: pd.DataFrame) -> None:
    datasets = list(noise_snr["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        data = noise_snr[noise_snr["dataset"].eq(dataset)]
        sns.lineplot(data=data, x="snr_db", y="mean_acc_delta_vs_l5", hue="arch", marker="o", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Mean accuracy delta vs matched L5_C64")
        ax.grid(alpha=0.25)
        ax.invert_xaxis()
    save_fig("noise_accuracy_delta_by_snr_vs_l5c64_grid.png")


def plot_noise_box(cell_comp: pd.DataFrame) -> None:
    datasets = list(cell_comp["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        data = cell_comp[cell_comp["dataset"].eq(dataset)]
        sns.boxplot(data=data, x="arch", y="acc_delta_cell_vs_l5", ax=ax)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_title(dataset)
        ax.set_xlabel("CRNN architecture")
        ax.set_ylabel("Cell-wise accuracy delta vs L5_C64")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    save_fig("noise_cell_delta_distribution_scene_snr.png")


def plot_noise_worst(noise_summary: pd.DataFrame) -> None:
    plot_data = noise_summary.copy()
    plt.figure(figsize=(13, 6))
    sns.barplot(data=plot_data, x="arch", y="worst_grid_acc_delta_vs_l5_worst", hue="dataset")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Worst Scene/SNR Accuracy Delta vs L5_C64 Worst Point")
    plt.ylabel("Worst accuracy delta")
    plt.xlabel("CRNN architecture")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_fig("noise_worst_case_delta_vs_l5c64_grid.png")


def snr_token(snr_db: float) -> str:
    value = int(snr_db) if float(snr_db).is_integer() else snr_db
    text = str(value).replace("-", "m").replace(".", "p")
    if not text.startswith("m"):
        text = f"p{text}"
    return text


def plot_scene_heatmaps(cell_comp: pd.DataFrame) -> None:
    for old_path in OUT_DIR.glob("scene_delta_heatmap_grid_baseline*.png"):
        old_path.unlink()

    for dataset in cell_comp["dataset"].drop_duplicates():
        dataset_data = cell_comp[cell_comp["dataset"].eq(dataset)]
        for snr_db in sorted(dataset_data["snr_db"].drop_duplicates(), reverse=True):
            data = dataset_data[dataset_data["snr_db"].eq(snr_db)]
            pivot = data.pivot_table(index="arch", columns="scene", values="acc_delta_cell_vs_l5", aggfunc="mean")
            plt.figure(figsize=(15, 5.5))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, linewidths=0.4)
            plt.title(f"Accuracy Delta vs Matched L5_C64 by Scene - {dataset} - SNR {snr_db:g} dB")
            plt.xlabel("TAU scene")
            plt.ylabel("CRNN architecture")
            plt.xticks(rotation=35, ha="right")
            save_fig(f"scene_delta_heatmap_grid_baseline_{dataset}_snr_{snr_token(float(snr_db))}db.png")


def plot_efficiency(efficiency: pd.DataFrame) -> None:
    data = efficiency.copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=data, x="arch", y="params", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Parameter Count")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Parameters")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    sns.barplot(data=data, x="arch", y="macs_per_second_if_32ms_update", ax=axes[1], color="#F58518")
    axes[1].set_title("Estimated Network MAC/s if Updating Every 32 ms")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("MAC/s")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    save_fig("efficiency_params_and_macs.png")


def write_report(best: pd.DataFrame, efficiency: pd.DataFrame, generated: list[Path]) -> Path:
    lines = [
        "# CRNN vs DSCNN L5_C64 Comparison - Grid Baseline",
        "",
        "本报告仅将当前 StreamingMFCC + CRNN 的 5 个结构与旧 DSCNN `L5_C64` 进行对比。",
        "",
        "## 输入文件",
        f"- `{CRNN_CLEAN_CSV}`",
        f"- `{CRNN_NOISE_GRID_CSV}`",
        f"- `{DSCNN_CLEAN_XLSX}`",
        f"- `{DSCNN_NOISE_GRID_CSV}`",
        "",
        "## 口径修正",
        "- Noise 对比已改为使用 `snr_scene_arch_sweep_grid_results.csv` 中的 `L5_C64` 行。",
        "- 每条 CRNN noise grid 结果都按 `dataset + scene + snr_db` 匹配同格点的 L5_C64。",
        "- `sweep_dscnn_noise_acc_results.csv` 不再作为 noise grid 基准。",
        "- 复杂度对比只估算网络 MAC，不含 MFCC 前端。",
        "",
        "## 每个数据集的关键结论",
    ]
    for _, row in best.iterrows():
        lines.extend(
            [
                f"### {row['dataset']}",
                f"- Clean 最好 CRNN: `{row['best_clean_crnn_arch']}`, acc={row['best_clean_acc']:.4f}, "
                f"相对 L5_C64 差值={row['clean_acc_delta_vs_l5']:.4f}, 参数减少={row['param_reduction_vs_l5_pct']:.2f}%。",
                f"- Noise grid 均值最好 CRNN: `{row['best_noise_mean_crnn_arch']}`, mean_acc={row['best_noise_mean_grid_acc']:.4f}; "
                f"L5_C64 mean_acc={row['l5_mean_grid_acc']:.4f}; 同格点平均差值={row['best_noise_mean_cell_delta_vs_l5']:.4f}。",
                f"- Noise grid 最坏点最好 CRNN: `{row['best_worst_case_crnn_arch']}`, worst_acc={row['best_worst_case_acc']:.4f}; "
                f"L5_C64 worst_acc={row['l5_min_grid_acc']:.4f}; 最坏点差值={row['best_worst_case_delta_vs_l5_worst']:.4f}。",
            ]
        )

    eff_crnn = efficiency[efficiency["arch"].ne("L5_C64")]
    lines.extend(
        [
            "",
            "## 复杂度概览",
            f"- L5_C64 参数量: {int(efficiency[efficiency['arch'].eq('L5_C64')]['params'].iloc[0])}。",
            f"- CRNN 参数量范围: {int(eff_crnn['params'].min())} 到 {int(eff_crnn['params'].max())}。",
            f"- 若每 32 ms 更新一次，CRNN 网络 MAC/s 约为 L5_C64 滑窗重算的 "
            f"{eff_crnn['macs_32ms_update_ratio_vs_l5'].min():.3%} 到 {eff_crnn['macs_32ms_update_ratio_vs_l5'].max():.3%}。",
            "",
            "## 输出文件",
        ]
    )
    for path in generated:
        lines.append(f"- `{path}`")
    path = OUT_DIR / "comparison_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    crnn_clean, crnn_noise, dscnn_clean, dscnn_noise_grid = load_inputs()
    dscnn_clean_l5 = dscnn_clean[dscnn_clean["arch"].eq("L5_C64")].copy()
    l5_grid = l5_noise_grid(dscnn_noise_grid)
    l5_summary = l5_noise_summary(l5_grid)

    clean_comp = clean_comparison(crnn_clean, dscnn_clean)
    noise_cell = noise_cell_comparison(crnn_noise, l5_grid)
    noise_arch, noise_snr, noise_scene = noise_summaries(noise_cell, l5_summary)
    efficiency = efficiency_comparison(clean_comp)
    best = best_summary(clean_comp, noise_arch, efficiency)

    generated: list[Path] = []
    tables = {
        "best_summary": best,
        "clean_compare": clean_comp,
        "noise_grid_cell_compare": noise_cell,
        "noise_arch_summary": noise_arch,
        "noise_snr_summary": noise_snr,
        "noise_scene_summary": noise_scene,
        "dscnn_l5_noise_grid": l5_grid,
        "dscnn_l5_noise_summary": l5_summary,
        "efficiency": efficiency,
        "dscnn_l5_clean": dscnn_clean_l5,
    }
    for name, df in tables.items():
        generated.append(write_csv(df, f"{name}.csv"))
    generated.append(save_excel(tables))

    plot_clean_delta(clean_comp)
    plot_clean_tradeoff(clean_comp, dscnn_clean_l5)
    plot_noise_snr(noise_snr)
    plot_noise_delta_by_snr(noise_snr)
    plot_noise_box(noise_cell)
    plot_noise_worst(noise_arch)
    plot_scene_heatmaps(noise_cell)
    plot_efficiency(efficiency)
    for png in sorted(OUT_DIR.glob("*.png")):
        generated.append(png)

    generated.append(write_report(best, efficiency, generated))
    print(f"[INFO] comparison outputs saved to: {OUT_DIR.resolve()}")
    print("[INFO] key summary:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
