from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sponge_early_exit_workbench as wb


def parse_args():
    parser = argparse.ArgumentParser(description="Train the seven-exit Fashion-MNIST CNN and save sponge-attack artifacts.")
    parser.add_argument("--data-dir", type=Path, default=wb.DATA_DIR)
    parser.add_argument("--train-subset", type=int, default=wb.TRAIN_SUBSET)
    parser.add_argument("--tuning-subset", type=int, default=wb.TUNING_SUBSET)
    parser.add_argument("--test-subset", type=int, default=wb.TEST_SUBSET)
    parser.add_argument("--batch-size", type=int, default=wb.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=wb.EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=wb.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=wb.WEIGHT_DECAY)
    parser.add_argument("--query-budget", type=int, default=wb.QUERY_BUDGET_PER_ATTACK)
    parser.add_argument("--attack-tuning-samples", type=int, default=wb.ATTACK_TUNE_MAX_TRAIN_SAMPLES)
    parser.add_argument("--seed", type=int, default=wb.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    wb.set_global_seed(args.seed)
    device = wb.build_device()
    print("device:", device)
    print("results:", wb.RESULTS_DIR.resolve())

    loaders = wb.build_loaders(
        data_dir=args.data_dir,
        train_subset=args.train_subset,
        tuning_subset=args.tuning_subset,
        test_subset=args.test_subset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader = loaders["train_loader"]
    tuning_loader = loaders["tuning_loader"]
    test_loader = loaders["test_loader"]
    print("sizes:", len(loaders["train_dataset"]), len(loaders["tuning_dataset"]), len(loaders["test_dataset"]))

    model = wb.BalancedFiveExitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history_rows = []
    for epoch in range(args.epochs):
        train_loss = wb.train_one_epoch(model, train_loader, optimizer, device)
        val_accs, val_losses = wb.evaluate_exits(model, tuning_loader, device)
        row = {"epoch": epoch + 1, "train_loss": train_loss}
        for exit_index, acc in enumerate(val_accs, start=1):
            row[f"val_acc_exit{exit_index}"] = acc
        for exit_index, loss in enumerate(val_losses, start=1):
            row[f"val_loss_exit{exit_index}"] = loss
        history_rows.append(row)
        print(
            f"epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_accs={[f'{acc:.3f}' for acc in val_accs]}"
        )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(wb.RESULTS_DIR / "training_history.csv", index=False)
    torch.save(model.state_dict(), wb.ARTIFACTS_DIR / "balanced_five_exit_cnn_state_dict.pt")
    wb.save_json(wb.RESULTS_DIR / "model_metadata.json", {
        "model_version": wb.MODEL_VERSION,
        "num_exits": wb.NUM_EXITS,
        "architecture": "BalancedFiveExitCNN",
    })

    tuning_logits_cpu, tuning_labels_cpu = wb.collect_logits_and_labels(model, tuning_loader, device)
    thresholds, threshold_grid_df, best_threshold_row = wb.calibrate_thresholds(
        tuning_logits_cpu,
        tuning_labels_cpu,
        tolerance=wb.ACCURACY_TOLERANCE,
    )
    threshold_grid_df.to_csv(wb.RESULTS_DIR / "threshold_grid.csv", index=False)
    pd.DataFrame([best_threshold_row]).to_csv(wb.RESULTS_DIR / "chosen_thresholds.csv", index=False)
    wb.save_json(
        wb.RESULTS_DIR / "thresholds.json",
        {"thresholds": [float(value) for value in thresholds], "best_row": best_threshold_row},
    )
    print("thresholds:", thresholds)

    cost_table_df, exit_costs_cpu = wb.compute_exit_costs()
    cost_table_df.to_csv(wb.RESULTS_DIR / "compute_cost_breakdown.csv", index=False)
    cost_proxy_df = pd.DataFrame({
        "exit": list(range(1, wb.NUM_EXITS + 1)),
        "mac_proxy": exit_costs_cpu.numpy().astype(np.int64),
        "mac_proxy_millions": exit_costs_cpu.numpy() / 1_000_000.0,
        "normalized_cost": (exit_costs_cpu / exit_costs_cpu[-1]).numpy(),
    })
    cost_proxy_df.to_csv(wb.RESULTS_DIR / "compute_cost_proxy.csv", index=False)
    wb.save_json(wb.RESULTS_DIR / "matrix_multiplication_proxy.json", wb.MatrixMultiplicationProxy(exit_costs_cpu).describe())

    oracle = wb.BlackBoxEarlyExitOracle(model, thresholds, exit_costs_cpu.to(device), device=device, cost_noise_std=0.0)

    attack_examples = wb.choose_attack_candidates(oracle, test_loader, wb.TEST_SUBSET, device)
    wb.save_split_csv(attack_examples, wb.RESULTS_DIR / "attack_candidates.csv", "attack_pool")
    wb.save_images_tensor(attack_examples, wb.ARTIFACTS_DIR / "attack_candidate_clean_images.pt")
    print("attack candidates:", len(attack_examples))

    train_indices, test_indices = wb.make_train_test_split(
        len(attack_examples),
        wb.ATTACK_SPLIT_TRAIN_FRACTION,
        wb.ATTACK_SPLIT_RANDOM_SEED,
    )
    tune_train_indices = train_indices
    split_payload = {
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "tune_train_indices": tune_train_indices.tolist(),
        "train_fraction": float(wb.ATTACK_SPLIT_TRAIN_FRACTION),
        "random_seed": int(wb.ATTACK_SPLIT_RANDOM_SEED),
    }
    wb.save_json(wb.RESULTS_DIR / "apso_clpso_split_indices.json", split_payload)

    tuning_examples = [attack_examples[int(index)] for index in tune_train_indices.tolist()]
    tuning_examples = wb.sample_examples(tuning_examples, args.attack_tuning_samples, wb.ATTACK_TUNE_RANDOM_SEED)

    obsolete_paths = [
        wb.RESULTS_DIR / "attack_tuning_split_candidates.csv",
        wb.RESULTS_DIR / "attack_test_split_candidates.csv",
        wb.RESULTS_DIR / "attack_tuning_candidates.csv",
        wb.ARTIFACTS_DIR / "attack_tuning_split_clean_images.pt",
        wb.ARTIFACTS_DIR / "attack_test_split_clean_images.pt",
        wb.ARTIFACTS_DIR / "attack_tuning_clean_images.pt",
    ]
    for obsolete_path in obsolete_paths:
        if obsolete_path.exists():
            obsolete_path.unlink()

    baseline_df, baseline_images, _ = wb.evaluate_baseline_examples(
        model=model,
        oracle=oracle,
        examples=attack_examples,
        thresholds=thresholds,
        exit_costs=exit_costs_cpu.to(device),
        device=device,
    )
    baseline_df.to_csv(wb.RESULTS_DIR / "attack_baseline_metrics.csv", index=False)
    torch.save(baseline_images, wb.ARTIFACTS_DIR / "attack_baseline_clean_images.pt")

    wb.save_json(
        wb.RESULTS_DIR / "budget_meta.json",
        {
            "reference_method": wb.REFERENCE_METHOD_FOR_TIME_CAP,
            "query_budget_per_attack": int(args.query_budget),
            "wall_clock_calibration_query_budget": int(wb.QUERY_BUDGET_PER_ATTACK),
            "attack_repeats": int(wb.ATTACK_REPEATS),
            "attack_particles": int(wb.ATTACK_PARTICLES),
            "attack_iterations": int(wb.ATTACK_ITERATIONS),
            "tuning_samples_used_for_wall_clock": int(len(tuning_examples)),
        },
    )

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
    print("calibrated wall-clock budget (seconds):", wall_clock_budget_seconds)

    print("saved model, thresholds, candidate pool, split indices, baseline metrics, and time budget.")


if __name__ == "__main__":
    main()
