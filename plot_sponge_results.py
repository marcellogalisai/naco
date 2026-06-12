from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import torch

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

import sponge_early_exit_workbench as wb


ALL_METHODS = wb.ALL_METHOD_ORDER
ALL_CAPS = ["query", "wall_clock"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sponge-attack results.")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--caps", nargs="+", default=ALL_CAPS, choices=ALL_CAPS)
    return parser.parse_args()


def load_long_results(method, cap_mode):
    path = wb.RESULTS_DIR / f"{method}_{cap_mode}_long_results.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["original_accuracy"] = frame["original_accuracy"].astype(float)
    frame["accuracy"] = frame["accuracy"].astype(float)
    if "success" in frame.columns:
        frame["success"] = frame["success"].astype(float)
    else:
        frame["success"] = ((frame["exit"].astype(int) == wb.NUM_EXITS) & (frame["accuracy"] > 0.5)).astype(float)
    if "first_success_queries" in frame.columns:
        frame["first_success_queries"] = pd.to_numeric(frame["first_success_queries"], errors="coerce")
    return frame


def load_universal_best_results(method):
    path = wb.RESULTS_DIR / f"{method}_bestM_long_results.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["original_accuracy"] = frame["original_accuracy"].astype(float)
    frame["accuracy"] = frame["accuracy"].astype(float)
    frame["success"] = frame["success"].astype(float)
    return frame


def load_images(path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def load_diversity_traces(method):
    path = wb.RESULTS_DIR / f"{method}_query_diversity_traces.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["diversity"] = frame["diversity"].astype(float)
    return frame


def method_color(method):
    if method in wb.UNIVERSAL_METHOD_ORDER:
        return "#E45756"
    return "#4C78A8"


def line_method_color(method):
    palette = {
        "random": "#4C78A8",
        "pso": "#F58518",
        "pso_jitter": "#54A24B",
        "genetic": "#E45756",
        "apso": "#B279A2",
        "clpso": "#9D755D",
        "universal_ga_weighted": "#000000",
        "universal_pso_jitter_weighted": "#EECA3B",
        "universal_pso_jitter_multiswarm": "#72B7B2",
        "universal_apso_weighted": "#EECA3B",
        "universal_apso_multiswarm": "#72B7B2",
    }
    return palette.get(method, method_color(method))


def display_method_name(method):
    mapping = {
        "universal_ga_weighted": "uap_ga",
        "universal_pso_jitter_weighted": "uap_pso",
        "universal_pso_jitter_multiswarm": "uap_multi",
        "universal_apso_weighted": "uap_pso",
        "universal_apso_multiswarm": "uap_multi",
    }
    return mapping.get(method, method)


def aggregate_for_bars(frame):
    per_input = frame.groupby("input_id", as_index=False).agg({
        "exit_delta": "mean",
        "original_accuracy": "mean",
        "accuracy": "mean",
        "original_proxy": "mean",
        "proxy": "mean",
    })
    per_input["accuracy_delta"] = per_input["original_accuracy"] - per_input["accuracy"]
    per_input["proxy_delta"] = per_input["proxy"] - per_input["original_proxy"]
    per_input["exit_variance"] = frame.groupby("input_id")["exit"].var(ddof=1).fillna(0.0).to_numpy()
    sem = lambda series: 0.0 if len(series) < 2 else float(series.std(ddof=1) / np.sqrt(len(series)))
    summary = {
        "mean_exit_delta": per_input["exit_delta"].mean(),
        "sem_exit_delta": sem(per_input["exit_delta"]),
        "mean_accuracy_delta": per_input["accuracy_delta"].mean(),
        "sem_accuracy_delta": sem(per_input["accuracy_delta"]),
        "mean_proxy_delta": per_input["proxy_delta"].mean(),
        "sem_proxy_delta": sem(per_input["proxy_delta"]),
        "mean_exit_variance": per_input["exit_variance"].mean(),
        "sem_exit_variance": sem(per_input["exit_variance"]),
        "inputs": len(per_input),
        "attack_rows": len(frame),
    }
    return per_input, summary


def aggregate_query_analysis(frame, query_budget):
    if "success" not in frame.columns:
        raise ValueError("query analysis requires a success column in long results")
    per_input_rows = []
    for input_id, group in frame.groupby("input_id"):
        success_mask = group["success"].astype(bool)
        successful = group[success_mask]
        if not successful.empty and "first_success_queries" in group.columns:
            first_success_queries_series = successful["first_success_queries"].dropna()
            mean_queries_to_success = float(first_success_queries_series.mean()) if not first_success_queries_series.empty else float(query_budget)
        else:
            mean_queries_to_success = float(query_budget)
        first_success_queries = mean_queries_to_success
        per_input_rows.append({
            "input_id": int(input_id),
            "method": str(group["method"].iloc[0]) if "method" in group.columns else None,
            "mean_exit": float(group["exit"].mean()),
            "mean_accuracy": float(group["accuracy"].mean()),
            "mean_proxy": float(group["proxy"].mean()),
            "mean_success_rate": float(group["success"].mean()),
            "mean_delta_l2": float(group["delta_l2"].mean()) if "delta_l2" in group.columns else float("nan"),
            "exit_variance": float(group["exit"].var(ddof=1)) if len(group) > 1 else 0.0,
            "mean_queries_to_success": mean_queries_to_success,
            "first_success_queries": first_success_queries,
            "success_count": int(success_mask.sum()),
        })
    return pd.DataFrame(per_input_rows)


def plot_bar(summary_df, metric_column, sem_column, title, ylabel, output_path, order):
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.15), 5.5))
    x = np.arange(len(order))
    values = [float(summary_df.loc[summary_df["method"] == method, metric_column].iloc[0]) for method in order]
    errors = [float(summary_df.loc[summary_df["method"] == method, sem_column].iloc[0]) for method in order]
    colors = [method_color(method) for method in order]
    bars = ax.bar(x, values, yerr=errors, capsize=5, color=colors, alpha=0.88, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([display_method_name(method) for method in order], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_exit_swarm(frame, title, output_path, order):
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.15), 5.5))
    if sns is not None:
        for method in order:
            subset = frame[frame["method"] == method]
            if subset.empty:
                continue
            sns.stripplot(
                data=subset,
                x="method",
                y="exit_delta",
                order=order,
                ax=ax,
                size=2.6,
                jitter=0.25,
                color=method_color(method),
                alpha=0.6,
            )
        if ax.legend_ is not None:
            ax.legend_.remove()
    else:  # pragma: no cover
        for x_pos, method in enumerate(order):
            subset = frame[frame["method"] == method]
            jitter = np.random.default_rng(0).uniform(-0.25, 0.25, size=len(subset))
            ax.scatter(
                np.full(len(subset), x_pos) + jitter,
                subset["exit_delta"],
                s=18,
                alpha=0.6,
                color="#4C78A8",
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([display_method_name(method) for method in order], rotation=90, ha="center", va="top")
    summary = (
        frame.groupby("method", as_index=False)["exit_delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_exit_delta", "std": "std_exit_delta", "count": "count_inputs"})
    )
    summary["sem_exit_delta"] = summary.apply(
        lambda row: 0.0 if int(row["count_inputs"]) < 2 else float(row["std_exit_delta"] / np.sqrt(row["count_inputs"])),
        axis=1,
    )
    for x_pos, method in enumerate(order):
        subset = summary[summary["method"] == method]
        if subset.empty:
            continue
        mean_value = float(subset["mean_exit_delta"].iloc[0])
        sem_value = float(subset["sem_exit_delta"].iloc[0])
        ax.errorbar(
            x=x_pos,
            y=mean_value,
            yerr=sem_value,
            fmt="_",
            color="black",
            ecolor="black",
            elinewidth=1.0,
            capsize=5,
            capthick=1.0,
            markersize=18,
            zorder=10,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display_method_name(method) for method in order], rotation=90, ha="center", va="top")
    ax.set_title(title)
    ax.set_ylabel("Exit delta (perturbed - original)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_success_vs_perturbation(frame, output_path, order, query_budget, bins=10):
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    combined_min = float(frame["mean_delta_l2"].min())
    combined_max = float(frame["mean_delta_l2"].max())
    if np.isclose(combined_min, combined_max):
        combined_max = combined_min + 1e-6
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)
    for idx, method in enumerate(order):
        subset = frame[frame["method"] == method].copy()
        if subset.empty:
            continue
        subset["bin"] = pd.cut(subset["mean_delta_l2"], bins=bin_edges, include_lowest=True, labels=False)
        grouped = subset.groupby("bin", as_index=False).agg(
            perturbation=("mean_delta_l2", "mean"),
            success_rate=("mean_success_rate", "mean"),
            n=("input_id", "size"),
        ).dropna()
        if grouped.empty:
            continue
        color = line_method_color(method)
        ax.plot(grouped["perturbation"], grouped["success_rate"], marker="o", linewidth=2.0, label=display_method_name(method), color=color)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Mean perturbation size")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate vs perturbation size (query cap)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_queries_to_success_swarm(frame, title, output_path, order, query_budget):
    if frame is None or frame.empty or not order:
        return
    plot_frame = frame.copy()
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.15), 2.9))
    if sns is not None:
        try:
            sns.swarmplot(data=plot_frame, x="method", y="mean_queries_to_success", order=order, ax=ax, size=2.8, color="#F58518", alpha=0.75)
        except Exception:
            sns.stripplot(data=plot_frame, x="method", y="mean_queries_to_success", order=order, ax=ax, size=2.6, jitter=0.25, color="#F58518", alpha=0.6)
    else:  # pragma: no cover
        for x_pos, method in enumerate(order):
            subset = plot_frame[plot_frame["method"] == method]
            jitter = np.random.default_rng(0).uniform(-0.14, 0.14, size=len(subset))
            ax.scatter(np.full(len(subset), x_pos) + jitter, subset["mean_queries_to_success"], s=6, alpha=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([display_method_name(method) for method in order], rotation=20, ha="right")
    ax.axhline(query_budget, linestyle="--", linewidth=1.0, color="gray", alpha=0.6)
    positive_values = plot_frame["mean_queries_to_success"].replace([np.inf, -np.inf], np.nan).dropna()
    if not positive_values.empty:
        lower = max(0.0, float(positive_values.min()) - 0.02 * max(float(query_budget), float(positive_values.max())))
        upper = max(float(query_budget) * 1.01, float(positive_values.max()) * 1.01)
        ax.set_ylim(lower, upper)
    ax.set_title(title)
    ax.set_ylabel("Mean queries to success")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _fft_radius_grid(height, width):
    yy, xx = np.indices((height, width))
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    return np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)


def compute_fft_statistics(original_images, perturbed_images, bins=12):
    if original_images is None or perturbed_images is None or len(original_images) == 0 or len(perturbed_images) == 0:
        return None
    original = original_images.float().cpu()
    perturbed = perturbed_images.float().cpu()
    if original.shape != perturbed.shape:
        raise ValueError(f"FFT analysis expects matching tensors, got {tuple(original.shape)} vs {tuple(perturbed.shape)}")
    delta = (perturbed - original).numpy()
    if delta.ndim != 4:
        raise ValueError(f"Expected BCHW tensors, got {delta.shape}")
    batch, channels, height, width = delta.shape
    if channels != 1:
        delta = delta.mean(axis=1, keepdims=True)
        channels = 1
    delta = delta[:, 0]
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(delta, axes=(-2, -1)), axes=(-2, -1)))
    fft_mag_mean = fft_mag.mean(axis=0)
    fft_mag_log = np.log1p(fft_mag_mean)
    radius = _fft_radius_grid(height, width)
    radius_norm = radius / max(radius.max(), 1e-8)
    radial_bins = np.linspace(0.0, 1.0, bins + 1)
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_energy = []
    for left, right in zip(radial_bins[:-1], radial_bins[1:]):
        mask = (radius_norm >= left) & (radius_norm < right)
        radial_energy.append(float(fft_mag_mean[mask].mean()) if mask.any() else 0.0)
    radial_energy = np.asarray(radial_energy, dtype=float)
    total_energy = float(fft_mag_mean.sum())
    high_freq_mask = radius_norm >= 0.60
    low_freq_mask = radius_norm <= 0.25
    high_freq_ratio = float(fft_mag_mean[high_freq_mask].sum() / max(total_energy, 1e-8))
    low_freq_ratio = float(fft_mag_mean[low_freq_mask].sum() / max(total_energy, 1e-8))
    per_image_high_ratio = []
    for sample in fft_mag:
        sample_total = float(sample.sum())
        per_image_high_ratio.append(float(sample[high_freq_mask].sum() / max(sample_total, 1e-8)))
    return {
        "fft_mag_mean": fft_mag_mean,
        "fft_mag_log": fft_mag_log,
        "radial_centers": radial_centers,
        "radial_energy": radial_energy,
        "high_freq_ratio": high_freq_ratio,
        "low_freq_ratio": low_freq_ratio,
        "per_image_high_ratio": np.asarray(per_image_high_ratio, dtype=float),
    }


def align_original_images(frame, original_images):
    if original_images is None or frame is None or frame.empty:
        return None
    indices = frame["input_id"].astype(int).to_numpy()
    return original_images[indices]


def plot_fft_radial(summary_rows, output_path, order):
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, method in enumerate(order):
        rows = [row for row in summary_rows if row["method"] == method]
        if not rows:
            continue
        row = rows[0]
        color = line_method_color(method)
        x = row["radial_centers"]
        y = row["radial_energy"] / max(row["radial_energy"].sum(), 1e-8)
        ax.plot(x, y, marker="o", linewidth=2.0, label=display_method_name(method), color=color)
    ax.set_xlabel("Normalized frequency radius")
    ax.set_ylabel("Normalized mean FFT magnitude")
    ax.set_title("Radial FFT spectrum by method")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fft_grid(summary_rows, output_path, order):
    if not summary_rows:
        return
    n_methods = len(order)
    n_cols = 3
    n_rows = int(np.ceil(n_methods / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for idx, method in enumerate(order):
        rows = [row for row in summary_rows if row["method"] == method]
        if not rows:
            continue
        row = rows[0]
        ax = axes.flat[idx]
        ax.axis("on")
        im = ax.imshow(row["fft_mag_log"], cmap="magma")
        ax.set_title(f"{display_method_name(method)}\nHF ratio={row['high_freq_ratio']:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean log FFT magnitude of perturbations", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_traces(diversity_frame, output_path, order):
    if diversity_frame is None or diversity_frame.empty:
        return
    per_input = diversity_frame.groupby(["method", "input_id", "generation"], as_index=False).agg({
        "diversity": "mean",
    })
    summary = per_input.groupby(["method", "generation"], as_index=False).agg(
        mean_diversity=("diversity", "mean"),
        sem_diversity=("diversity", lambda s: 0.0 if len(s) < 2 else float(s.std(ddof=1) / np.sqrt(len(s)))),
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    palette = sns.color_palette("tab10", n_colors=max(1, len(order))) if sns is not None else None
    for idx, method in enumerate(order):
        subset = summary[summary["method"] == method].sort_values("generation")
        if subset.empty:
            continue
        color = palette[idx % len(palette)] if palette is not None else None
        x = subset["generation"].to_numpy()
        y = subset["mean_diversity"].to_numpy()
        sem = subset["sem_diversity"].to_numpy()
        ax.plot(x, y, label=display_method_name(method), linewidth=2.0, color=color)
        ax.fill_between(x, y - sem, y + sem, alpha=0.16, color=color)
    ax.set_title("Mean population diversity across query-budget attacks")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized mean pairwise distance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    wb.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    universal_frames = {}
    for method in args.methods:
        if method in wb.UNIVERSAL_METHOD_ORDER:
            universal_frame = load_universal_best_results(method)
            if universal_frame is not None and not universal_frame.empty:
                universal_frame["method"] = method
                universal_frame["cap_mode"] = "universal"
                universal_frames[method] = universal_frame
    for cap_mode in args.caps:
        cap_frames = []
        per_input_frames = []
        for method in args.methods:
            if method in wb.UNIVERSAL_METHOD_ORDER:
                continue
            frame = load_long_results(method, cap_mode)
            if frame is None or frame.empty:
                continue
            frame["method"] = method
            frame["cap_mode"] = cap_mode
            cap_frames.append(frame)
            per_input, summary = aggregate_for_bars(frame)
            per_input["method"] = method
            per_input["cap_mode"] = cap_mode
            per_input_frames.append(per_input)
            summary_rows.append({"method": method, "cap_mode": cap_mode, **summary})

        for method, frame in universal_frames.items():
            cap_frames.append(frame.copy())
            per_input, summary = aggregate_for_bars(frame)
            per_input["method"] = method
            per_input["cap_mode"] = cap_mode
            per_input_frames.append(per_input)
            summary_rows.append({"method": method, "cap_mode": cap_mode, **summary})

        if not cap_frames:
            continue

        combined = pd.concat(cap_frames, ignore_index=True)
        summary_frame = pd.DataFrame(summary_rows)
        cap_summary = summary_frame[summary_frame["cap_mode"] == cap_mode].copy()
        order = [method for method in args.methods if method in cap_summary["method"].tolist()]
        if cap_mode == "query":
            plot_bar(
                cap_summary,
                metric_column="mean_exit_variance",
                sem_column="sem_exit_variance",
                title="Across-repeat exit variance by method (query cap)",
                ylabel="Mean within-input exit variance",
                output_path=wb.PLOTS_DIR / "bar_exit_variance_query.png",
                order=order,
            )
        if per_input_frames:
            exit_swarm_frame = pd.concat(per_input_frames, ignore_index=True)
            plot_exit_swarm(
                exit_swarm_frame,
                title=f"Exit delta by input ({cap_mode} cap)",
                output_path=wb.PLOTS_DIR / f"exit_swarm_{cap_mode}.png",
                order=order,
            )
        if cap_mode == "query":
            query_inputs = []
            fft_rows = []
            baseline_images = load_images(wb.ARTIFACTS_DIR / "attack_baseline_clean_images.pt")
            for method in order:
                if method in wb.UNIVERSAL_METHOD_ORDER:
                    frame = universal_frames.get(method)
                else:
                    frame = load_long_results(method, cap_mode)
                if frame is None or frame.empty:
                    continue
                per_input = aggregate_query_analysis(frame, query_budget=wb.QUERY_BUDGET_PER_ATTACK)
                per_input["method"] = method
                query_inputs.append(per_input)
                if method in wb.UNIVERSAL_METHOD_ORDER:
                    perturbed_images = load_images(wb.ARTIFACTS_DIR / f"{method}_bestM_adversarial_images.pt")
                else:
                    perturbed_images = load_images(wb.ARTIFACTS_DIR / f"{method}_{cap_mode}_adversarial_images.pt")
                original_images = align_original_images(frame, baseline_images)
                fft_stats = compute_fft_statistics(original_images, perturbed_images)
                if fft_stats is not None:
                    fft_rows.append({
                        "method": method,
                        **fft_stats,
                    })
            if query_inputs:
                query_input_frame = pd.concat(query_inputs, ignore_index=True)
                plot_success_vs_perturbation(
                    query_input_frame,
                    output_path=wb.PLOTS_DIR / "success_vs_perturbation_query.png",
                    order=order,
                    query_budget=wb.QUERY_BUDGET_PER_ATTACK,
                )
                plot_queries_to_success_swarm(
                    query_input_frame[~query_input_frame["method"].isin(wb.UNIVERSAL_METHOD_ORDER)],
                    title="Mean queries to success by input (query cap)",
                    output_path=wb.PLOTS_DIR / "queries_to_success_swarm_query.png",
                    order=[method for method in order if method not in wb.UNIVERSAL_METHOD_ORDER],
                    query_budget=wb.QUERY_BUDGET_PER_ATTACK,
                )
            if fft_rows:
                plot_fft_radial(
                    fft_rows,
                    output_path=wb.PLOTS_DIR / "fft_radial_query.png",
                    order=order,
                )
            diversity_frames = []
            for method in order:
                diversity_frame = load_diversity_traces(method)
                if diversity_frame is not None and not diversity_frame.empty:
                    diversity_frame["method"] = method
                    diversity_frames.append(diversity_frame)
            if diversity_frames:
                diversity_combined = pd.concat(diversity_frames, ignore_index=True)
                plot_diversity_traces(
                    diversity_combined,
                    output_path=wb.PLOTS_DIR / "diversity_query.png",
                    order=order,
                )

    print("saved plots to:", wb.PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()
