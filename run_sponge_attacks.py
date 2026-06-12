from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import sponge_early_exit_workbench as wb


ALL_METHODS = wb.ALL_METHOD_ORDER
NON_UNIVERSAL_METHODS = wb.METHOD_ORDER
UNIVERSAL_METHODS = wb.UNIVERSAL_METHOD_ORDER
ALL_CAPS = ["query", "wall_clock"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-parameter sponge attacks under query/time budgets.")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--caps", nargs="+", default=ALL_CAPS, choices=ALL_CAPS)
    parser.add_argument("--query-budget", type=int, default=wb.QUERY_BUDGET_PER_ATTACK)
    parser.add_argument("--attack-repeats", type=int, default=wb.ATTACK_REPEATS)
    parser.add_argument("--test-input-cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=wb.SEED)
    return parser.parse_args()


def load_attack_examples(device):
    attack_df = pd.read_csv(wb.RESULTS_DIR / "attack_candidates.csv")
    attack_images = torch.load(wb.ARTIFACTS_DIR / "attack_candidate_clean_images.pt", map_location="cpu")
    examples = []
    for row_index, row in attack_df.iterrows():
        examples.append({
            "source_index": int(row["source_index"]),
            "image": attack_images[row_index:row_index + 1],
            "label": int(row["label"]),
            "clean_exit": int(row["clean_exit"]),
            "clean_blackbox_pred": int(row["clean_blackbox_pred"]),
            "clean_blackbox_conf": float(row["clean_blackbox_conf"]),
            "clean_observed_cost": float(row["clean_observed_cost"]),
            "clean_matmul_proxy": float(row["clean_matmul_proxy"]),
            "clean_final_pred": int(row["clean_final_pred"]),
            "clean_final_conf": float(row["clean_final_conf"]),
        })
    split_payload = wb.load_json(wb.RESULTS_DIR / "apso_clpso_split_indices.json")
    train_indices = np.array(split_payload["train_indices"], dtype=int)
    test_indices = np.array(split_payload["test_indices"], dtype=int)
    tune_train_indices = np.array(split_payload["tune_train_indices"], dtype=int)
    baseline_df = pd.read_csv(wb.RESULTS_DIR / "attack_baseline_metrics.csv")
    baseline_df = baseline_df.set_index("input_id", drop=False)
    print("attack candidates:", len(examples))
    print("train split size:", len(train_indices))
    print("test split size:", len(test_indices))
    print("tune-train subset size:", len(tune_train_indices))
    return examples, train_indices, test_indices, tune_train_indices, baseline_df


def load_model_and_oracle(device):
    model = wb.BalancedFiveExitCNN().to(device)
    metadata_path = wb.RESULTS_DIR / "model_metadata.json"
    if metadata_path.exists():
        metadata = wb.load_json(metadata_path)
        if metadata.get("model_version") != wb.MODEL_VERSION:
            raise RuntimeError(
                f"Checkpoint version mismatch: found {metadata.get('model_version')!r}, expected {wb.MODEL_VERSION!r}. "
                "Please rerun `python3 train_sponge_cnn.py` to rebuild the model and artifacts."
            )
    try:
        model.load_state_dict(torch.load(wb.ARTIFACTS_DIR / "balanced_five_exit_cnn_state_dict.pt", map_location=device))
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the saved CNN checkpoint into the current architecture. "
            "This usually means the model was trained before the latest exit-head change. "
            "Please rerun `python3 train_sponge_cnn.py` and then rerun the attack script."
        ) from exc
    model.eval()
    threshold_payload = wb.load_json(wb.RESULTS_DIR / "thresholds.json")
    thresholds = tuple(threshold_payload["thresholds"])
    exit_costs_cpu = wb.compute_exit_costs()[1]
    oracle = wb.BlackBoxEarlyExitOracle(model, thresholds, exit_costs_cpu.to(device), device=device, cost_noise_std=0.0)
    return model, thresholds, exit_costs_cpu, oracle


def _universal_attack_function(method):
    mapping = {
        "universal_ga_weighted": wb.universal_ga_weighted_attack,
        "universal_pso_jitter_weighted": wb.universal_pso_jitter_weighted_attack,
        "universal_pso_jitter_multiswarm": wb.universal_pso_jitter_multiswarm_attack,
        "universal_apso_weighted": wb.universal_pso_jitter_weighted_attack,
        "universal_apso_multiswarm": wb.universal_pso_jitter_multiswarm_attack,
    }
    return mapping[method]


def summarize_universal_results(result_frame, m_value, queries_used, crafting_ids):
    per_input = result_frame.groupby("input_id", as_index=False).agg({
        "exit_delta": "mean",
        "original_accuracy": "mean",
        "accuracy": "mean",
        "original_proxy": "mean",
        "proxy": "mean",
        "success": "mean",
        "perturbation_norm": "mean",
    })
    per_input["accuracy_delta"] = per_input["original_accuracy"] - per_input["accuracy"]
    per_input["proxy_delta"] = per_input["proxy"] - per_input["original_proxy"]
    summary = pd.DataFrame([{
        "M": int(m_value),
        "crafting_size": int(m_value),
        "crafting_ids": json.dumps([int(x) for x in crafting_ids]),
        "mean_exit_delta": float(per_input["exit_delta"].mean()),
        "mean_accuracy_delta": float(per_input["accuracy_delta"].mean()),
        "mean_proxy_delta": float(per_input["proxy_delta"].mean()),
        "mean_success_rate": float(per_input["success"].mean()),
        "mean_perturbation_norm": float(per_input["perturbation_norm"].mean()),
        "oracle_queries_used": int(queries_used),
        "inputs": int(len(per_input)),
    }])
    return summary


def run_universal_method_m(method, m_value, examples, train_indices, test_indices, baseline_df, model, oracle, thresholds, exit_costs, device, attack_repeats):
    attack_fn = _universal_attack_function(method)
    fixed_params = wb.fixed_method_params(method)
    selected_ids, selected_frame, raw_feature_frame, crafting_metadata = wb.select_diverse_crafting_indices(examples, train_indices.tolist(), m_value)
    crafting_examples = [examples[int(idx)] for idx in selected_ids]
    result_rows = []
    adversarial_images = []
    delta_rows = []
    history_rows = []
    total_queries = 0
    for attack_index in range(1, attack_repeats + 1):
        seed = wb.ATTACK_BASE_SEED + wb.METHOD_SEED_OFFSETS[method] + int(m_value) * 17 + attack_index * 101
        attack_result = attack_fn(
            oracle=oracle,
            examples=crafting_examples,
            epsilon=wb.EPSILON,
            n_particles=wb.ATTACK_PARTICLES,
            n_iterations=wb.ATTACK_ITERATIONS,
            rng_seed=seed,
            hyperparams=fixed_params,
        )
        delta = attack_result["delta"].to(device)
        total_queries += int(attack_result["queries"])
        delta_rows.append(delta.detach().cpu())
        history_rows.append({
            "method": method,
            "M": int(m_value),
            "attack_index": attack_index,
            "seed": int(seed),
            "crafting_ids": [int(x) for x in selected_ids],
            "fitness_history": attack_result.get("fitness_history"),
            "crafting_success_history": attack_result.get("crafting_success_history"),
            "weight_history": attack_result.get("weight_history"),
            "diversity_history": attack_result.get("diversity_history"),
            "queries": int(attack_result.get("queries", 0)),
            "num_swarms": int(attack_result.get("num_swarms", 0)) if attack_result.get("num_swarms") is not None else None,
        })
        for input_id in tqdm(test_indices.tolist(), desc=f"{method} M={m_value} atk={attack_index} eval", leave=False):
            example = examples[int(input_id)]
            base = example["image"].to(device)
            label = int(example["label"])
            baseline_row = baseline_df.loc[int(input_id)]
            original_exit = int(baseline_row["original_exit"])
            original_accuracy = bool(baseline_row["original_accuracy"])
            original_proxy = float(baseline_row["original_proxy"])
            eval_m = wb.evaluate_adversarial_example_for_evaluator(
                model=model,
                base=base,
                delta=delta,
                label=label,
                thresholds=thresholds,
                exit_costs=exit_costs,
            )
            adv_exit = int(eval_m["exit"][0].item())
            adv_accuracy = bool(int(eval_m["final_pred"][0].item()) == label)
            adv_success = bool((adv_exit == len(exit_costs)) and adv_accuracy)
            adv_image = (base + delta).clamp(0.0, 1.0).detach().cpu()
            adversarial_images.append(adv_image)
            result_rows.append({
                "method": method,
                "cap_mode": "universal",
                "input_id": int(input_id),
                "attack_index": attack_index,
                "source_index": int(example["source_index"]),
                "label": label,
                "M": int(m_value),
                "original_exit": original_exit,
                "original_accuracy": original_accuracy,
                "original_proxy": original_proxy,
                "exit": adv_exit,
                "exit_delta": adv_exit - original_exit,
                "accuracy": adv_accuracy,
                "success": adv_success,
                "proxy": float(eval_m["norm_cost"][0].item() * float(exit_costs[-1].item())),
                "adv_observed_cost": float(eval_m["norm_cost"][0].item()),
                "queries_used": int(attack_result["queries"]),
                "elapsed_seconds": np.nan,
                "delta_l2": float(delta.flatten(1).norm(p=2, dim=1)[0].item()),
                "perturbation_norm": float(wb.perturbation_norm(delta, wb.NORM_TYPE)[0].item()),
            })
    result_frame = pd.DataFrame(result_rows)
    long_path = wb.RESULTS_DIR / f"{method}_m{m_value}_long_results.csv"
    summary_path = wb.RESULTS_DIR / f"{method}_m{m_value}_summary.csv"
    feature_path = wb.RESULTS_DIR / f"{method}_m{m_value}_crafting_features.csv"
    metadata_path = wb.RESULTS_DIR / f"{method}_m{m_value}_crafting_metadata.json"
    delta_path = wb.ARTIFACTS_DIR / f"{method}_m{m_value}_delta.pt"
    adv_path = wb.ARTIFACTS_DIR / f"{method}_m{m_value}_adversarial_images.pt"
    history_path = wb.RESULTS_DIR / f"{method}_m{m_value}_optimization_history.json"
    result_frame.to_csv(long_path, index=False)
    summary_df = summarize_universal_results(result_frame, m_value, total_queries, selected_ids)
    summary_df.to_csv(summary_path, index=False)
    selected_frame.to_csv(feature_path, index=False)
    wb.save_json(metadata_path, crafting_metadata)
    torch.save(torch.cat(delta_rows, dim=0) if delta_rows else torch.empty(0), delta_path)
    torch.save(torch.cat(adversarial_images, dim=0) if adversarial_images else torch.empty(0), adv_path)
    wb.save_json(history_path, history_rows)
    return {
        "method": method,
        "M": int(m_value),
        "long_path": long_path,
        "summary_path": summary_path,
        "delta_path": delta_path,
        "adv_path": adv_path,
        "feature_path": feature_path,
        "metadata_path": metadata_path,
        "summary": summary_df.iloc[0].to_dict(),
        "result_frame": result_frame,
        "raw_feature_frame": raw_feature_frame,
    }


def select_best_universal_run(method, run_payloads):
    ranking_rows = []
    for payload in run_payloads:
        summary = payload["summary"]
        ranking_rows.append({
            "M": int(payload["M"]),
            "mean_success_rate": float(summary["mean_success_rate"]),
            "mean_exit_delta": float(summary["mean_exit_delta"]),
            "mean_proxy_delta": float(summary["mean_proxy_delta"]),
        })
    ranking_frame = pd.DataFrame(ranking_rows).sort_values(
        by=["mean_success_rate", "mean_exit_delta", "mean_proxy_delta", "M"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_m = int(ranking_frame.iloc[0]["M"])
    best_payload = next(payload for payload in run_payloads if int(payload["M"]) == best_m)
    wb.save_json(wb.RESULTS_DIR / f"{method}_best_M.json", {
        "method": method,
        "best_M": best_m,
        "ranking": ranking_rows,
        "best_long_results": wb.repo_relative_path(best_payload["long_path"]),
        "best_summary": wb.repo_relative_path(best_payload["summary_path"]),
        "best_delta": wb.repo_relative_path(best_payload["delta_path"]),
        "best_adversarial_images": wb.repo_relative_path(best_payload["adv_path"]),
        "best_crafting_features": wb.repo_relative_path(best_payload["feature_path"]),
        "best_crafting_metadata": wb.repo_relative_path(best_payload["metadata_path"]),
    })
    shutil.copyfile(best_payload["long_path"], wb.RESULTS_DIR / f"{method}_bestM_long_results.csv")
    shutil.copyfile(best_payload["summary_path"], wb.RESULTS_DIR / f"{method}_bestM_summary.csv")
    shutil.copyfile(best_payload["feature_path"], wb.RESULTS_DIR / f"{method}_bestM_crafting_features.csv")
    shutil.copyfile(best_payload["metadata_path"], wb.RESULTS_DIR / f"{method}_bestM_crafting_metadata.json")
    shutil.copyfile(best_payload["delta_path"], wb.ARTIFACTS_DIR / f"{method}_bestM_delta.pt")
    shutil.copyfile(best_payload["adv_path"], wb.ARTIFACTS_DIR / f"{method}_bestM_adversarial_images.pt")
    return best_payload, ranking_frame


def run_single_method_cap(method, cap_mode, examples, test_indices, baseline_df, model, oracle, thresholds, exit_costs, query_budget, wall_clock_budget_seconds, device, attack_repeats):
    attack_fn = wb.get_attack_function(method)
    fixed_params = wb.fixed_method_params(method)

    result_rows = []
    adversarial_images = []
    diversity_rows = []
    first_attack_saved = False

    for input_id in tqdm(test_indices.tolist(), desc=f"{method} {cap_mode} inputs", leave=False):
        example = examples[int(input_id)]
        base = example["image"].to(device)
        label = int(example["label"])
        clean_pred = int(example["clean_blackbox_pred"])
        baseline_row = baseline_df.loc[int(input_id)]
        original_exit = int(baseline_row["original_exit"])
        original_accuracy = bool(baseline_row["original_accuracy"])
        original_proxy = float(baseline_row["original_proxy"])

        for attack_index in tqdm(range(1, attack_repeats + 1), desc=f"{method} {cap_mode} attacks", leave=False):
            seed = wb.ATTACK_BASE_SEED + int(input_id) * 10_007 + attack_index * 101 + wb.METHOD_SEED_OFFSETS[method]
            initial_positions, initial_velocities = wb.make_initial_swarm_state(base, wb.EPSILON, wb.ATTACK_PARTICLES, seed)
            budget = wb.AttackBudget(
                mode=cap_mode,
                query_budget=int(query_budget),
                wall_clock_budget_s=wall_clock_budget_seconds,
            ).start()
            first_success_queries = None

            def progress_callback(best_delta=None, best_metrics=None, queries_used=None, elapsed_seconds=None):
                nonlocal first_success_queries
                if first_success_queries is not None or best_delta is None:
                    return
                eval_progress = wb.evaluate_adversarial_example_for_evaluator(
                    model=model,
                    base=base,
                    delta=best_delta,
                    label=label,
                    thresholds=thresholds,
                    exit_costs=exit_costs,
                )
                is_success = (int(eval_progress["exit"][0].item()) == len(exit_costs)) and (int(eval_progress["final_pred"][0].item()) == label)
                if is_success:
                    first_success_queries = int(queries_used if queries_used is not None else 0)

            attack_result = attack_fn(
                oracle=oracle,
                base=base,
                clean_blackbox_pred=clean_pred,
                epsilon=wb.EPSILON,
                n_particles=wb.ATTACK_PARTICLES,
                n_iterations=wb.ATTACK_ITERATIONS,
                initial_positions=initial_positions,
                initial_velocities=initial_velocities,
                rng_seed=seed,
                hyperparams=fixed_params,
                budget=budget,
                progress_callback=progress_callback,
            )
            eval_m = wb.evaluate_adversarial_example_for_evaluator(
                model=model,
                base=base,
                delta=attack_result["delta"],
                label=label,
                thresholds=thresholds,
                exit_costs=exit_costs,
            )

            metrics = attack_result["metrics"] or {}
            adv_proxy = float(metrics["observed_matmul_proxy"][0].item()) if "observed_matmul_proxy" in metrics else float("nan")
            adv_cost = float(metrics["observed_cost"][0].item()) if "observed_cost" in metrics else float("nan")
            adv_exit = int(eval_m["exit"][0].item())
            adv_accuracy = bool(int(eval_m["final_pred"][0].item()) == label)
            adv_success = bool((adv_exit == len(exit_costs)) and adv_accuracy)
            exit_delta = adv_exit - original_exit
            adv_image = attack_result["image"].detach().cpu() if attack_result["image"] is not None else base.detach().cpu()
            if not first_attack_saved:
                first_attack_path = wb.ARTIFACTS_DIR / f"{method}_{cap_mode}_first_attack_image.pt"
                torch.save(adv_image, first_attack_path)
                wb.save_json(
                    wb.RESULTS_DIR / f"{method}_{cap_mode}_first_attack_metadata.json",
                    {
                        "method": method,
                        "cap_mode": cap_mode,
                        "input_id": int(input_id),
                        "attack_index": attack_index,
                        "source_index": int(example["source_index"]),
                        "label": label,
                        "path": wb.repo_relative_path(first_attack_path),
                    },
                )
                first_attack_saved = True

            if cap_mode == "query":
                for generation_index, diversity_value in enumerate(attack_result.get("diversity_history") or [], start=1):
                    diversity_rows.append({
                        "method": method,
                        "cap_mode": cap_mode,
                        "input_id": int(input_id),
                        "attack_index": attack_index,
                        "generation": generation_index,
                        "diversity": float(diversity_value),
                    })

            delta_l2 = float("nan")
            if attack_result["delta"] is not None:
                delta_l2 = float(attack_result["delta"].flatten(1).norm(p=2, dim=1)[0].item())

            result_rows.append({
                "method": method,
                "cap_mode": cap_mode,
                "input_id": int(input_id),
                "attack_index": attack_index,
                "source_index": int(example["source_index"]),
                "label": label,
                "original_exit": original_exit,
                "original_accuracy": original_accuracy,
                "original_proxy": original_proxy,
                "exit": adv_exit,
                "exit_delta": exit_delta,
                "accuracy": adv_accuracy,
                "success": adv_success,
                "first_success_queries": int(first_success_queries) if first_success_queries is not None else np.nan,
                "proxy": adv_proxy,
                "adv_observed_cost": adv_cost,
                "queries_used": int(attack_result["queries"]),
                "elapsed_seconds": float(attack_result["elapsed_seconds"]),
                "delta_l2": delta_l2,
            })
            adversarial_images.append(adv_image)

    result_frame = pd.DataFrame(result_rows)
    long_path = wb.RESULTS_DIR / f"{method}_{cap_mode}_long_results.csv"
    result_frame.to_csv(long_path, index=False)
    matrices = wb.build_attack_result_matrices(result_rows, repeats=attack_repeats)
    matrices["exit"].to_csv(wb.RESULTS_DIR / f"{method}_{cap_mode}_exit_matrix.csv", index=False)
    matrices["accuracy"].to_csv(wb.RESULTS_DIR / f"{method}_{cap_mode}_accuracy_matrix.csv", index=False)
    matrices["proxy"].to_csv(wb.RESULTS_DIR / f"{method}_{cap_mode}_proxy_matrix.csv", index=False)
    summary_df = wb.summarize_attack_rows(result_frame, repeats=attack_repeats)
    summary_df.to_csv(wb.RESULTS_DIR / f"{method}_{cap_mode}_summary.csv", index=False)
    if cap_mode == "query":
        diversity_frame = pd.DataFrame(diversity_rows)
        diversity_frame.to_csv(wb.RESULTS_DIR / f"{method}_{cap_mode}_diversity_traces.csv", index=False)
    avg_exit_delta = float(result_frame["exit_delta"].mean()) if not result_frame.empty else float("nan")
    print(f"{method} / {cap_mode} avg exit delta: {avg_exit_delta:.4f}")
    if cap_mode == "wall_clock":
        avg_queries = float(result_frame["queries_used"].mean()) if not result_frame.empty else float("nan")
        print(f"{method} / {cap_mode} avg queries used: {avg_queries:.2f}")

    adversarial_tensor = torch.cat(adversarial_images, dim=0) if adversarial_images else torch.empty(0)
    torch.save(adversarial_tensor, wb.ARTIFACTS_DIR / f"{method}_{cap_mode}_adversarial_images.pt")

    bundle = {
        "method": method,
        "cap_mode": cap_mode,
        "input_ids": result_frame["input_id"].tolist(),
        "attack_indices": result_frame["attack_index"].tolist(),
        "source_indices": result_frame["source_index"].tolist(),
        "labels": result_frame["label"].tolist(),
        "original_images_path": wb.repo_relative_path(wb.ARTIFACTS_DIR / "attack_baseline_clean_images.pt"),
        "adversarial_images_path": wb.repo_relative_path(wb.ARTIFACTS_DIR / f"{method}_{cap_mode}_adversarial_images.pt"),
        "results_csv": wb.repo_relative_path(long_path),
    }
    wb.save_json(wb.RESULTS_DIR / f"{method}_{cap_mode}_bundle.json", bundle)
    print(f"saved {method} / {cap_mode} -> {long_path}")
    return result_frame


def main():
    args = parse_args()
    wb.set_global_seed(args.seed)
    device = wb.build_device()
    print("device:", device)

    model, thresholds, exit_costs_cpu, oracle = load_model_and_oracle(device)
    examples, train_indices, test_indices, tune_train_indices, baseline_df = load_attack_examples(device)
    if args.test_input_cap is not None and args.test_input_cap > 0:
        test_indices = test_indices[: min(int(args.test_input_cap), len(test_indices))]
        print("capped test inputs:", len(test_indices))

    tuning_examples = [examples[int(index)] for index in tune_train_indices.tolist()]
    tuning_examples = wb.sample_examples(tuning_examples, wb.ATTACK_TUNE_MAX_TRAIN_SAMPLES, wb.ATTACK_TUNE_RANDOM_SEED)

    if not (wb.RESULTS_DIR / "wall_clock_budget.json").exists():
        wall_clock_budget_seconds = wb.calibrate_wall_clock_budget(
            reference_method=wb.REFERENCE_METHOD_FOR_TIME_CAP,
            tuning_examples=tuning_examples,
            model=model,
            oracle=oracle,
            thresholds=thresholds,
            exit_costs=exit_costs_cpu.to(device),
            device=device,
            query_budget=wb.QUERY_BUDGET_PER_ATTACK,
            repeats=3,
        )
        wb.save_json(wb.RESULTS_DIR / "wall_clock_budget.json", {"wall_clock_budget_seconds": wall_clock_budget_seconds})
    else:
        wall_clock_budget_seconds = float(wb.load_json(wb.RESULTS_DIR / "wall_clock_budget.json")["wall_clock_budget_seconds"])
    print("wall-clock budget (seconds):", wall_clock_budget_seconds)

    universal_plot_rows = []
    for method in args.methods:
        if method in UNIVERSAL_METHODS:
            print(f"{method} fixed params:", wb.fixed_method_params(method))
            run_payloads = []
            for m_value in wb.UNIVERSAL_M_VALUES:
                payload = run_universal_method_m(
                    method=method,
                    m_value=m_value,
                    examples=examples,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    baseline_df=baseline_df,
                    model=model,
                    oracle=oracle,
                    thresholds=thresholds,
                    exit_costs=exit_costs_cpu.to(device),
                    device=device,
                    attack_repeats=args.attack_repeats,
                )
                run_payloads.append(payload)
                universal_plot_rows.append({
                    "method": method,
                    "M": int(m_value),
                    **payload["summary"],
                })
                print(f"{method} / M={m_value} mean success: {float(payload['summary']['mean_success_rate']):.4f}")
            best_payload, ranking_frame = select_best_universal_run(method, run_payloads)
            ranking_frame.to_csv(wb.RESULTS_DIR / f"{method}_bestM_ranking.csv", index=False)
            print(f"{method} best M: {best_payload['M']}")
            continue

        print(f"{method} fixed params:", wb.fixed_method_params(method))

        for cap_mode in args.caps:
            run_single_method_cap(
                method=method,
                cap_mode=cap_mode,
                examples=examples,
                test_indices=test_indices,
                baseline_df=baseline_df,
                model=model,
                oracle=oracle,
                thresholds=thresholds,
                exit_costs=exit_costs_cpu.to(device),
                query_budget=args.query_budget,
                wall_clock_budget_seconds=wall_clock_budget_seconds if cap_mode == "wall_clock" else None,
                device=device,
                attack_repeats=args.attack_repeats,
            )

    if universal_plot_rows:
        pd.DataFrame(universal_plot_rows).to_csv(wb.RESULTS_DIR / "universal_plot_ready_summary.csv", index=False)


if __name__ == "__main__":
    main()
