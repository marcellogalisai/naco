from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

import sponge_early_exit_workbench as wb


STATISTICS_DIR = wb.RESULTS_DIR / "statistics"
STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
STATISTICS_PLOTS_DIR = STATISTICS_DIR / "plots"
STATISTICS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical analysis of saved sponge-attack results.")
    parser.add_argument("--results-dir", type=Path, default=wb.RESULTS_DIR)
    parser.add_argument("--methods", nargs="+", default=wb.ALL_METHOD_ORDER)
    parser.add_argument("--caps", nargs="+", default=["query", "wall_clock", "universal"], choices=["query", "wall_clock", "universal"])
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstraps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=wb.SEED)
    return parser.parse_args()


def load_long_results(results_dir: Path, method: str, cap_mode: str):
    if cap_mode == "universal":
        path = results_dir / f"{method}_bestM_long_results.csv"
    else:
        path = results_dir / f"{method}_{cap_mode}_long_results.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["method"] = method
    frame["cap_mode"] = cap_mode
    if "success" in frame.columns:
        frame["success"] = pd.to_numeric(frame["success"], errors="coerce").fillna(0.0)
    else:
        frame["success"] = ((frame["exit"].astype(int) == wb.NUM_EXITS) & (frame["accuracy"].astype(float) > 0.5)).astype(float)
    if "original_accuracy" in frame.columns:
        frame["original_accuracy"] = pd.to_numeric(frame["original_accuracy"], errors="coerce")
    if "accuracy" in frame.columns:
        frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
    if "original_proxy" in frame.columns:
        frame["original_proxy"] = pd.to_numeric(frame["original_proxy"], errors="coerce")
    if "proxy" in frame.columns:
        frame["proxy"] = pd.to_numeric(frame["proxy"], errors="coerce")
    if "first_success_queries" in frame.columns:
        frame["first_success_queries"] = pd.to_numeric(frame["first_success_queries"], errors="coerce")
    return frame


def infer_query_budget(frame: pd.DataFrame):
    if "queries_used" not in frame.columns or frame.empty:
        return np.nan
    value = float(frame["queries_used"].max())
    return value if np.isfinite(value) else np.nan


def aggregate_input_level(frame: pd.DataFrame, cap_mode: str):
    group_cols = ["input_id"]
    aggregations = {
        "exit_delta": "mean",
        "original_accuracy": "mean",
        "accuracy": "mean",
        "original_proxy": "mean",
        "proxy": "mean",
        "success": "mean",
        "delta_l2": "mean",
    }
    if "queries_used" in frame.columns:
        aggregations["queries_used"] = "mean"
    per_input = frame.groupby(group_cols, as_index=False).agg(aggregations)
    per_input["accuracy_delta"] = per_input["original_accuracy"] - per_input["accuracy"]
    per_input["proxy_delta"] = per_input["proxy"] - per_input["original_proxy"]
    per_input["exit_variance"] = frame.groupby("input_id")["exit"].var(ddof=1).fillna(0.0).to_numpy()
    per_input["method"] = str(frame["method"].iloc[0])
    per_input["cap_mode"] = cap_mode
    per_input["attempts"] = frame.groupby("input_id").size().to_numpy()
    if cap_mode == "query" and "first_success_queries" in frame.columns:
        query_budget = infer_query_budget(frame)
        rows = []
        for input_id, group in frame.groupby("input_id"):
            success_mask = group["success"].astype(float) > 0.0
            successful = group.loc[success_mask, "first_success_queries"].dropna()
            mean_q = float(successful.mean()) if not successful.empty else float(query_budget)
            rows.append((int(input_id), mean_q, bool(success_mask.any())))
        query_df = pd.DataFrame(rows, columns=["input_id", "mean_queries_to_success", "any_success"])
        per_input = per_input.merge(query_df, on="input_id", how="left")
    else:
        per_input["mean_queries_to_success"] = np.nan
        per_input["any_success"] = per_input["success"] > 0.0
    return per_input


def sign_flip_permutation(values, rng: np.random.Generator, permutations: int):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    observed = float(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(permutations, x.size), replace=True)
    permuted = (signs * x).mean(axis=1)
    p_two = (1.0 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1.0)
    p_greater = (1.0 + np.sum(permuted >= observed)) / (permutations + 1.0)
    return observed, float(p_two), float(p_greater)


def paired_permutation(differences, rng: np.random.Generator, permutations: int):
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, np.nan
    observed = float(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(permutations, diff.size), replace=True)
    permuted = (signs * diff).mean(axis=1)
    p_two = (1.0 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1.0)
    return observed, float(p_two)


def bootstrap_mean_ci(values, rng: np.random.Generator, bootstraps: int, alpha=0.05):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), float(x[0])
    samples = x[rng.integers(0, x.size, size=(bootstraps, x.size))]
    means = samples.mean(axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def cohen_dz(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    std = float(x.std(ddof=1))
    if std <= 1e-12:
        return np.nan
    return float(x.mean() / std)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    finite = p[finite_mask]
    if finite.size == 0:
        return adjusted
    order = np.argsort(finite)
    ranked = finite[order]
    m = ranked.size
    holm = np.empty(m, dtype=float)
    running = 0.0
    for i, value in enumerate(ranked):
        adjusted_value = (m - i) * value
        running = max(running, adjusted_value)
        holm[i] = min(running, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = holm
    adjusted[finite_mask] = restored
    return adjusted


def build_one_sample_table(input_frames: dict[tuple[str, str], pd.DataFrame], permutations: int, bootstraps: int, seed: int):
    rows = []
    metrics = ["exit_delta", "accuracy_delta", "proxy_delta", "success"]
    for (cap_mode, method), frame in sorted(input_frames.items()):
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            rng_perm = np.random.default_rng(seed + hash((cap_mode, method, metric, "perm")) % (2**32 - 1))
            rng_boot = np.random.default_rng(seed + hash((cap_mode, method, metric, "boot")) % (2**32 - 1))
            observed, p_two, p_greater = sign_flip_permutation(values, rng_perm, permutations)
            ci_low, ci_high = bootstrap_mean_ci(values, rng_boot, bootstraps)
            rows.append({
                "cap_mode": cap_mode,
                "method": method,
                "metric": metric,
                "n_inputs": int(np.isfinite(values).sum()),
                "mean": float(np.nanmean(values)) if len(values) else np.nan,
                "median": float(np.nanmedian(values)) if len(values) else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_two_sided": p_two,
                "p_greater_zero": p_greater,
                "effect_size_dz": cohen_dz(values),
                "observed_mean": observed,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p_greater_zero"] = np.nan
        out["holm_p_two_sided"] = np.nan
        for cap_mode in out["cap_mode"].unique():
            for metric in out["metric"].unique():
                mask = (out["cap_mode"] == cap_mode) & (out["metric"] == metric)
                out.loc[mask, "holm_p_greater_zero"] = holm_adjust(out.loc[mask, "p_greater_zero"].to_numpy())
                out.loc[mask, "holm_p_two_sided"] = holm_adjust(out.loc[mask, "p_two_sided"].to_numpy())
    return out


def build_pairwise_table(input_frames: dict[tuple[str, str], pd.DataFrame], permutations: int, bootstraps: int, seed: int):
    rows = []
    metrics = ["exit_delta", "accuracy_delta", "proxy_delta", "success", "exit_variance"]
    by_cap = {}
    for (cap_mode, method), frame in input_frames.items():
        by_cap.setdefault(cap_mode, {})[method] = frame
    for cap_mode, method_frames in sorted(by_cap.items()):
        methods = sorted(method_frames)
        for metric in metrics:
            for method_a, method_b in itertools.combinations(methods, 2):
                left = method_frames[method_a][["input_id", metric]].rename(columns={metric: "value_a"})
                right = method_frames[method_b][["input_id", metric]].rename(columns={metric: "value_b"})
                merged = left.merge(right, on="input_id", how="inner")
                diff = merged["value_a"].to_numpy(dtype=float) - merged["value_b"].to_numpy(dtype=float)
                rng_perm = np.random.default_rng(seed + hash((cap_mode, method_a, method_b, metric, "perm")) % (2**32 - 1))
                rng_boot = np.random.default_rng(seed + hash((cap_mode, method_a, method_b, metric, "boot")) % (2**32 - 1))
                observed, p_two = paired_permutation(diff, rng_perm, permutations)
                ci_low, ci_high = bootstrap_mean_ci(diff, rng_boot, bootstraps)
                finite_diff = diff[np.isfinite(diff)]
                rows.append({
                    "cap_mode": cap_mode,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_inputs": int(finite_diff.size),
                    "mean_diff_a_minus_b": float(np.nanmean(finite_diff)) if finite_diff.size else np.nan,
                    "median_diff_a_minus_b": float(np.nanmedian(finite_diff)) if finite_diff.size else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_two_sided": p_two,
                    "effect_size_dz": cohen_dz(finite_diff),
                    "wins_a": int(np.sum(finite_diff > 0)),
                    "ties": int(np.sum(np.isclose(finite_diff, 0.0))),
                    "wins_b": int(np.sum(finite_diff < 0)),
                    "observed_mean": observed,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p_two_sided"] = np.nan
        for cap_mode in out["cap_mode"].unique():
            for metric in out["metric"].unique():
                mask = (out["cap_mode"] == cap_mode) & (out["metric"] == metric)
                out.loc[mask, "holm_p_two_sided"] = holm_adjust(out.loc[mask, "p_two_sided"].to_numpy())
    return out


def build_query_to_success_table(input_frames: dict[tuple[str, str], pd.DataFrame], permutations: int, bootstraps: int, seed: int):
    cap_mode = "query"
    method_frames = {method: frame for (cap, method), frame in input_frames.items() if cap == cap_mode}
    rows = []
    if not method_frames:
        return pd.DataFrame(rows)
    all_success_table = None
    for method, frame in method_frames.items():
        cols = frame[["input_id", "success"]].rename(columns={"success": method})
        all_success_table = cols if all_success_table is None else all_success_table.merge(cols, on="input_id", how="outer")
    if all_success_table is None:
        return pd.DataFrame(rows)
    success_achievable_ids = all_success_table.loc[all_success_table.drop(columns=["input_id"]).max(axis=1) > 0.0, "input_id"].astype(int).tolist()
    methods = sorted(method_frames)
    for method_a, method_b in itertools.combinations(methods, 2):
        left = method_frames[method_a][["input_id", "mean_queries_to_success"]].rename(columns={"mean_queries_to_success": "value_a"})
        right = method_frames[method_b][["input_id", "mean_queries_to_success"]].rename(columns={"mean_queries_to_success": "value_b"})
        merged = left.merge(right, on="input_id", how="inner")
        merged = merged[merged["input_id"].isin(success_achievable_ids)]
        diff = merged["value_a"].to_numpy(dtype=float) - merged["value_b"].to_numpy(dtype=float)
        rng_perm = np.random.default_rng(seed + hash((method_a, method_b, "query_to_success", "perm")) % (2**32 - 1))
        rng_boot = np.random.default_rng(seed + hash((method_a, method_b, "query_to_success", "boot")) % (2**32 - 1))
        observed, p_two = paired_permutation(diff, rng_perm, permutations)
        ci_low, ci_high = bootstrap_mean_ci(diff, rng_boot, bootstraps)
        finite_diff = diff[np.isfinite(diff)]
        rows.append({
            "cap_mode": "query",
            "metric": "mean_queries_to_success_on_any-success_inputs",
            "method_a": method_a,
            "method_b": method_b,
            "n_inputs": int(finite_diff.size),
            "mean_diff_a_minus_b": float(np.nanmean(finite_diff)) if finite_diff.size else np.nan,
            "median_diff_a_minus_b": float(np.nanmedian(finite_diff)) if finite_diff.size else np.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_two_sided": p_two,
            "effect_size_dz": cohen_dz(finite_diff),
            "wins_a_earlier": int(np.sum(finite_diff < 0)),
            "ties": int(np.sum(np.isclose(finite_diff, 0.0))),
            "wins_b_earlier": int(np.sum(finite_diff > 0)),
            "observed_mean": observed,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p_two_sided"] = holm_adjust(out["p_two_sided"].to_numpy())
    return out


def display_method_name(method):
    mapping = {
        "universal_ga_weighted": "uap_ga",
        "universal_pso_jitter_weighted": "uap_pso",
        "universal_pso_jitter_multiswarm": "uap_multi",
        "universal_apso_weighted": "uap_pso",
        "universal_apso_multiswarm": "uap_multi",
    }
    return mapping.get(method, method)


def plot_one_sample_forest(one_sample: pd.DataFrame, metric: str, cap_mode: str, output_path: Path):
    subset = one_sample[(one_sample["metric"] == metric) & (one_sample["cap_mode"] == cap_mode)].copy()
    if subset.empty:
        return
    subset = subset.sort_values("mean", ascending=True).reset_index(drop=True)
    labels = [display_method_name(method) for method in subset["method"]]
    y = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.55 * len(subset) + 1.5)))
    ax.errorbar(
        subset["mean"],
        y,
        xerr=[subset["mean"] - subset["ci_low"], subset["ci_high"] - subset["mean"]],
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        markersize=5,
    )
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Mean {metric} with 95% bootstrap CI")
    ax.set_title(f"{metric} effectiveness ({cap_mode})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_combined_metric_forest(one_sample: pd.DataFrame, metric: str, primary_cap_mode: str, output_path: Path):
    subset = one_sample[
        (one_sample["metric"] == metric)
        & (one_sample["cap_mode"].isin([primary_cap_mode, "universal"]))
    ].copy()
    if subset.empty:
        return
    order_map = {method: idx for idx, method in enumerate(wb.ALL_METHOD_ORDER)}
    subset["method_rank"] = subset["method"].map(order_map).fillna(999)
    subset = subset.sort_values(["method_rank"]).reset_index(drop=True)
    subset["label"] = subset["method"].map(display_method_name)
    color_map = {
        "query": "#4C78A8",
        "wall_clock": "#F58518",
        "universal": "#E45756",
    }
    y = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(10.0, max(6.0, 0.42 * len(subset) + 1.8)))
    for idx, row in subset.iterrows():
        ax.errorbar(
            row["mean"],
            idx,
            xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt="o",
            color=color_map.get(row["cap_mode"], "black"),
            ecolor=color_map.get(row["cap_mode"], "black"),
            elinewidth=1.2,
            capsize=4,
            markersize=5,
        )
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(subset["label"].tolist())
    ax.set_xlabel(f"Mean {metric} with 95% bootstrap CI")
    title_metric = metric.replace("_", " ")
    ax.set_title(f"{title_metric} effectiveness ({primary_cap_mode.replace('_', '-')})")
    ax.grid(axis="x", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=color_map[primary_cap_mode], marker="o", linestyle="None", label=primary_cap_mode.replace("_", "-")),
        plt.Line2D([0], [0], color=color_map["universal"], marker="o", linestyle="None", label="universal"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _pairwise_matrix(pairwise: pd.DataFrame, metric: str, cap_mode: str, value_col: str):
    subset = pairwise[(pairwise["metric"] == metric) & (pairwise["cap_mode"] == cap_mode)].copy()
    if subset.empty:
        return None, None
    methods = sorted(set(subset["method_a"]).union(set(subset["method_b"])))
    matrix = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    for _, row in subset.iterrows():
        a = row["method_a"]
        b = row["method_b"]
        value = float(row[value_col])
        matrix.loc[a, b] = value
        matrix.loc[b, a] = -value if value_col == "mean_diff_a_minus_b" else value
    if value_col == "mean_diff_a_minus_b":
        for method in methods:
            matrix.loc[method, method] = 0.0
    return matrix, methods


def plot_pairwise_heatmap(pairwise: pd.DataFrame, metric: str, cap_mode: str, value_col: str, title: str, cmap: str, center: float | None, output_path: Path):
    matrix, methods = _pairwise_matrix(pairwise, metric, cap_mode, value_col)
    if matrix is None:
        return
    labels = [display_method_name(method) for method in methods]
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    values = matrix.to_numpy(dtype=float)
    if sns is not None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            center=center,
            annot=True,
            fmt=".2g",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.85},
        )
    else:  # pragma: no cover
        im = ax.imshow(values, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.85)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2g}", ha="center", va="center", fontsize=8)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_statistics_plots(one_sample: pd.DataFrame, pairwise: pd.DataFrame, query_success: pd.DataFrame):
    if not one_sample.empty:
        plot_combined_metric_forest(
            one_sample=one_sample,
            metric="exit_delta",
            primary_cap_mode="query",
            output_path=STATISTICS_PLOTS_DIR / "forest_exit_delta_all_query.png",
        )
        plot_combined_metric_forest(
            one_sample=one_sample,
            metric="exit_delta",
            primary_cap_mode="wall_clock",
            output_path=STATISTICS_PLOTS_DIR / "forest_exit_delta_all_wall_clock.png",
        )
        plot_combined_metric_forest(
            one_sample=one_sample,
            metric="success",
            primary_cap_mode="query",
            output_path=STATISTICS_PLOTS_DIR / "forest_success_all_query.png",
        )
    if not pairwise.empty:
        plot_pairwise_heatmap(
            pairwise=pairwise,
            metric="exit_delta",
            cap_mode="query",
            value_col="holm_p_two_sided",
            title="Holm-adjusted p-values: exit delta (query)",
            cmap="viridis_r",
            center=None,
            output_path=STATISTICS_PLOTS_DIR / "heatmap_pvalue_exit_delta_query.png",
        )
        plot_pairwise_heatmap(
            pairwise=pairwise,
            metric="success",
            cap_mode="query",
            value_col="holm_p_two_sided",
            title="Holm-adjusted p-values: success (query)",
            cmap="viridis_r",
            center=None,
            output_path=STATISTICS_PLOTS_DIR / "heatmap_pvalue_success_query.png",
        )


def main():
    args = parse_args()
    available_frames = {}
    raw_frames = {}
    for cap_mode in args.caps:
        for method in args.methods:
            frame = load_long_results(args.results_dir, method, cap_mode)
            if frame is None or frame.empty:
                continue
            raw_frames[(cap_mode, method)] = frame
            available_frames[(cap_mode, method)] = aggregate_input_level(frame, cap_mode)
    one_sample = build_one_sample_table(available_frames, permutations=args.permutations, bootstraps=args.bootstraps, seed=args.seed)
    pairwise = build_pairwise_table(available_frames, permutations=args.permutations, bootstraps=args.bootstraps, seed=args.seed)
    query_success = build_query_to_success_table(available_frames, permutations=args.permutations, bootstraps=args.bootstraps, seed=args.seed)
    generate_statistics_plots(one_sample=one_sample, pairwise=pairwise, query_success=query_success)
    print("saved statistics to:", STATISTICS_DIR.resolve())


if __name__ == "__main__":
    main()
