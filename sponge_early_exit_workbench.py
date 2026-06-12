from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results_sponge_workbench"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
PLOTS_DIR = RESULTS_DIR / "plots"
for directory in [RESULTS_DIR, ARTIFACTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SEED = 7
TRAIN_SUBSET = 20_000
TUNING_SUBSET = 2_000
TEST_SUBSET = 2_000
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_EXITS = 5
MODEL_VERSION = "balanced_five_exit_cnn_mps_v1"
TUNING_SPEC_VERSION = "exact_notebook_v1"
NORM_TYPE = "linf"
NORM_BUDGETS = {
    "linf": 0.12,
    "l2": 2.40,
    "l1": 24.0,
    "l0": 80,
}
L0_PIXEL_EPSILON = 0.30
EPSILON = float(NORM_BUDGETS[NORM_TYPE])

ATTACK_REPEATS = 5
ATTACK_PARTICLES = 24
ATTACK_ITERATIONS = 200
QUERY_BUDGET_PER_ATTACK = ATTACK_PARTICLES * ATTACK_ITERATIONS
TUNING_SEED = 54_321
REFERENCE_METHOD_FOR_TIME_CAP = "pso"
ATTACK_BASE_SEED = 12_345
ATTACK_SPLIT_TRAIN_FRACTION = 0.7
ATTACK_SPLIT_RANDOM_SEED = ATTACK_BASE_SEED + 9000
ATTACK_TUNE_MAX_TRAIN_SAMPLES = 20
ATTACK_TUNE_RANDOM_SEED = ATTACK_BASE_SEED + 9999
INITIAL_VELOCITY_FRACTION = 0.05
OBSERVED_FINAL_COST_THRESHOLD = 0.995
ACCURACY_TOLERANCE = 0.015

METHOD_SEED_OFFSETS = {
    "random": 101,
    "pso": 202,
    "pso_jitter": 203,
    "genetic": 505,
    "apso": 606,
    "clpso": 707,
    "universal_ga_weighted": 808,
    "universal_pso_jitter_weighted": 909,
    "universal_pso_jitter_multiswarm": 1001,
    "universal_apso_weighted": 909,
    "universal_apso_multiswarm": 1001,
}

BLACKBOX_FITNESS_WEIGHTS = {
    "observed_cost": 1.60,
    "label_match": 0.18,
    "final_confidence_on_clean_label": 0.12,
    "early_uncertainty": 0.25,
    "perturb_norm_fraction": -0.02,
}

METHOD_ORDER = ["random", "pso", "pso_jitter", "genetic", "apso", "clpso"]
UNIVERSAL_METHOD_ORDER = ["universal_ga_weighted", "universal_pso_jitter_weighted", "universal_pso_jitter_multiswarm"]
ALL_METHOD_ORDER = METHOD_ORDER + UNIVERSAL_METHOD_ORDER
UNIVERSAL_M_VALUES = [2, 4, 6, 8, 10]
UNIVERSAL_CONSENSUS_PERIOD = 5
UNIVERSAL_C3_START = 0.3
UNIVERSAL_C3_END = 1.0

PSO_PARAMS = {
    "inertia": 0.8,
    "cognitive": 1.1,
    "social": 1.6,
    "jitter": 0.0,
    "max_velocity_fraction": 0.35,
    "initial_velocity_fraction": INITIAL_VELOCITY_FRACTION,
}
PSO_JITTER_PARAMS = {
    **PSO_PARAMS,
    "jitter": 0.01,
}

GENETIC_PARAMS = {
    "elite_fraction": 0.18,
    "tournament_size": 3,
    "crossover_rate": 0.55,
    "mutation_rate": 0.18,
    "mutation_scale": 0.75,
    "random_immigrant_fraction": 0.18,
    "local_immigrant_fraction": 0.45,
    "local_mutation_scale": 0.35,
}

APSO_PARAMS = {
    "c_min": 1.2,
    "c_max": 2.4,
    "c_sum_max": 4.0,
    "delta_min": 0.05,
    "delta_max": 0.10,
    "sigma_max": 1.0,
    "sigma_min": 0.10,
    "max_velocity_fraction": 0.15,
    "initial_velocity_fraction": INITIAL_VELOCITY_FRACTION,
}

CLPSO_PARAMS = {
    "w0": 0.90,
    "w1": 0.30,
    "c": 1.2,
    "refreshing_gap": 5,
    "max_velocity_fraction": 0.20,
    "initial_velocity_fraction": INITIAL_VELOCITY_FRACTION,
}


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(key): to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def save_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, sort_keys=True)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_relative_path(path) -> str:
    path_obj = Path(path).resolve()
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(
    data_dir: Path = DATA_DIR,
    train_subset: int = TRAIN_SUBSET,
    tuning_subset: int = TUNING_SUBSET,
    test_subset: int = TEST_SUBSET,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    transform = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    test_perm = torch.randperm(len(test_full), generator=torch.Generator().manual_seed(seed + 1)).tolist()

    train_indices = train_perm[:train_subset]
    tuning_indices = train_perm[train_subset:train_subset + tuning_subset]
    test_indices = test_perm[:test_subset]

    train_dataset = IndexedSubset(train_full, train_indices)
    tuning_dataset = IndexedSubset(train_full, tuning_indices)
    test_dataset = IndexedSubset(test_full, test_indices)

    pin_memory = build_device().type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    tuning_loader = DataLoader(tuning_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    return {
        "train_full": train_full,
        "test_full": test_full,
        "train_indices": train_indices,
        "tuning_indices": tuning_indices,
        "test_indices": test_indices,
        "train_dataset": train_dataset,
        "tuning_dataset": tuning_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "tuning_loader": tuning_loader,
        "test_loader": test_loader,
    }


def make_train_test_split(n_samples, train_fraction, seed):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_samples)
    permuted = rng.permutation(all_indices)
    split_at = int(round(train_fraction * n_samples))
    split_at = min(max(split_at, 1), n_samples - 1)
    train_indices = np.sort(permuted[:split_at])
    test_indices = np.sort(permuted[split_at:])
    return train_indices, test_indices


class IndexedSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        image, label = self.base_dataset[source_index]
        return image, label, source_index


class ExitHead(nn.Module):
    def __init__(self, channels: int, pool_size: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            nn.Flatten(),
            nn.Linear(channels * pool_size * pool_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BalancedFiveExitCNN(nn.Module):
    def __init__(self, model_channels=None, exit_pool_sizes=None, num_classes: int = 10):
        super().__init__()
        if model_channels is None:
            model_channels = {
                "block1": 16,
                "block2": 32,
                "block3": 64,
                "block4": 96,
                "block5": 160,
            }
        if exit_pool_sizes is None:
            exit_pool_sizes = {
                "exit1": 1,
                "exit2": 1,
                "exit3": 1,
                "exit4": 1,
            }
        c1 = model_channels["block1"]
        c2 = model_channels["block2"]
        c3 = model_channels["block3"]
        c4 = model_channels["block4"]
        c5 = model_channels["block5"]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(c2, c3, kernel_size=3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3, c3, kernel_size=3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(c3, c4, kernel_size=3, padding=1),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
                nn.Conv2d(c4, c4, kernel_size=3, padding=1),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(c4, c5, kernel_size=3, padding=1),
                nn.BatchNorm2d(c5),
                nn.ReLU(inplace=True),
                nn.Conv2d(c5, c5, kernel_size=3, padding=1),
                nn.BatchNorm2d(c5),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            ),
        ])
        self.exit_heads = nn.ModuleList([
            ExitHead(c1, exit_pool_sizes["exit1"], num_classes),
            ExitHead(c2, exit_pool_sizes["exit2"], num_classes),
            ExitHead(c3, exit_pool_sizes["exit3"], num_classes),
            ExitHead(c4, exit_pool_sizes["exit4"], num_classes),
        ])
        self.final_head = nn.Linear(c5, num_classes)

    def forward_all(self, x):
        outputs = []
        for block_index, block in enumerate(self.blocks):
            x = block(x)
            if block_index < len(self.exit_heads):
                outputs.append(self.exit_heads[block_index](x))
        outputs.append(self.final_head(x))
        return outputs

    def forward(self, x):
        return self.forward_all(x)[-1]


SevenExitCNN = BalancedFiveExitCNN


def batch_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def exit_loss_weights(num_exits: int):
    return torch.linspace(0.55, 1.0, steps=num_exits).tolist()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits_all = model.forward_all(images)
        weights = exit_loss_weights(len(logits_all))
        loss = sum(weight * F.cross_entropy(logits, labels) for weight, logits in zip(weights, logits_all))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_exits(model, loader, device):
    model.eval()
    correct = None
    losses = None
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits_all = model.forward_all(images)
        if correct is None:
            correct = torch.zeros(len(logits_all), dtype=torch.long)
            losses = torch.zeros(len(logits_all))
        for exit_index, logits in enumerate(logits_all):
            correct[exit_index] += (logits.argmax(dim=1) == labels).sum().cpu()
            losses[exit_index] += F.cross_entropy(logits, labels, reduction="sum").cpu()
        total += images.size(0)
    return (correct.float() / total).tolist(), (losses / total).tolist()


def collect_logits_and_labels(model, loader, device):
    model.eval()
    logits_by_exit = None
    labels_list = []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        logits_all = model.forward_all(images)
        if logits_by_exit is None:
            logits_by_exit = [[] for _ in logits_all]
        for exit_index, logits in enumerate(logits_all):
            logits_by_exit[exit_index].append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
    return [torch.cat(parts, dim=0) for parts in logits_by_exit], torch.cat(labels_list, dim=0)


def dynamic_predictions_from_logits(logits_all, thresholds):
    num_exits = len(logits_all)
    batch_size = logits_all[0].size(0)
    device = logits_all[0].device
    exits = torch.full((batch_size,), num_exits, dtype=torch.long, device=device)
    predictions = torch.empty((batch_size,), dtype=torch.long, device=device)
    confidences = torch.empty((batch_size,), dtype=torch.float32, device=device)
    unresolved = torch.ones((batch_size,), dtype=torch.bool, device=device)
    for exit_index, logits in enumerate(logits_all[:-1], start=1):
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        take = unresolved & (conf >= float(thresholds[exit_index - 1]))
        exits[take] = exit_index
        predictions[take] = pred[take]
        confidences[take] = conf[take]
        unresolved = unresolved & (~take)
    final_probs = F.softmax(logits_all[-1], dim=1)
    final_conf, final_pred = final_probs.max(dim=1)
    exits[unresolved] = num_exits
    predictions[unresolved] = final_pred[unresolved]
    confidences[unresolved] = final_conf[unresolved]
    return exits, predictions, confidences


def metrics_for_thresholds(logits_all, labels, thresholds):
    exits, predictions, _ = dynamic_predictions_from_logits(logits_all, thresholds)
    final_acc = (logits_all[-1].argmax(dim=1).cpu() == labels).float().mean().item()
    dynamic_acc = (predictions.cpu() == labels).float().mean().item()
    mean_exit = exits.float().mean().item()
    row = {"dynamic_acc": dynamic_acc, "final_acc": final_acc, "mean_exit": mean_exit}
    for threshold_index, threshold in enumerate(thresholds, start=1):
        row[f"threshold_exit{threshold_index}"] = float(threshold)
    for exit_index in range(1, len(logits_all) + 1):
        row[f"exit{exit_index}_rate"] = (exits == exit_index).float().mean().item()
    return row


def calibrate_thresholds(logits_all, labels, tolerance=ACCURACY_TOLERANCE):
    final_acc = (logits_all[-1].argmax(dim=1).cpu() == labels).float().mean().item()
    quantiles = np.linspace(0.55, 0.82, len(logits_all) - 1)
    base_thresholds = []
    for exit_index, logits in enumerate(logits_all[:-1], start=1):
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct_conf = conf[(pred.cpu() == labels)]
        if len(correct_conf) == 0:
            correct_conf = conf.cpu()
        threshold = float(np.quantile(correct_conf.numpy(), quantiles[exit_index - 1]))
        base_thresholds.append(float(np.clip(threshold, 0.5, 0.99)))

    scale_grid = np.round(np.linspace(0.85, 1.10, 11), 3)
    offset_grid = np.round(np.linspace(-0.05, 0.03, 9), 3)
    candidate_rows = []
    feasible_rows = []
    for scale in scale_grid:
        for offset in offset_grid:
            thresholds = tuple(float(np.clip(scale * threshold + offset, 0.5, 0.99)) for threshold in base_thresholds)
            row = metrics_for_thresholds(logits_all, labels, thresholds)
            row["scale"] = float(scale)
            row["offset"] = float(offset)
            candidate_rows.append(row)
            if row["dynamic_acc"] >= final_acc - tolerance:
                feasible_rows.append(row)

    if feasible_rows:
        best = sorted(feasible_rows, key=lambda row: (row["mean_exit"], -row["dynamic_acc"]))[0]
    else:
        best = sorted(candidate_rows, key=lambda row: (-(row["dynamic_acc"] - 0.03 * row["mean_exit"]), row["mean_exit"]))[0]

    thresholds = tuple(float(best[f"threshold_exit{exit_index}"]) for exit_index in range(1, len(logits_all)))
    return thresholds, pd.DataFrame(candidate_rows), best


def conv_macs(h, w, in_ch, out_ch, kernel_size):
    return h * w * out_ch * in_ch * kernel_size * kernel_size


def linear_macs(in_features, out_features):
    return in_features * out_features


def compute_exit_costs():
    channels = [16, 32, 64, 96, 160]
    stage_rows = [
        {"stage": "block1_conv1", "mac_proxy": conv_macs(28, 28, 1, channels[0], 3)},
        {"stage": "block1_conv2", "mac_proxy": conv_macs(28, 28, channels[0], channels[0], 3)},
        {"stage": "exit1_linear", "mac_proxy": linear_macs(channels[0] * 1 * 1, 10)},
        {"stage": "block2_conv1", "mac_proxy": conv_macs(14, 14, channels[0], channels[1], 3)},
        {"stage": "block2_conv2", "mac_proxy": conv_macs(14, 14, channels[1], channels[1], 3)},
        {"stage": "exit2_linear", "mac_proxy": linear_macs(channels[1] * 1 * 1, 10)},
        {"stage": "block3_conv1", "mac_proxy": conv_macs(7, 7, channels[1], channels[2], 3)},
        {"stage": "block3_conv2", "mac_proxy": conv_macs(7, 7, channels[2], channels[2], 3)},
        {"stage": "exit3_linear", "mac_proxy": linear_macs(channels[2] * 1 * 1, 10)},
        {"stage": "block4_conv1", "mac_proxy": conv_macs(7, 7, channels[2], channels[3], 3)},
        {"stage": "block4_conv2", "mac_proxy": conv_macs(7, 7, channels[3], channels[3], 3)},
        {"stage": "exit4_linear", "mac_proxy": linear_macs(channels[3] * 1 * 1, 10)},
        {"stage": "block5_conv1", "mac_proxy": conv_macs(3, 3, channels[3], channels[4], 3)},
        {"stage": "block5_conv2", "mac_proxy": conv_macs(3, 3, channels[4], channels[4], 3)},
        {"stage": "exit5_linear", "mac_proxy": linear_macs(channels[4], 10)},
    ]
    cost_df = pd.DataFrame(stage_rows)
    costs = []
    cumulative = 0
    for exit_index in range(1, NUM_EXITS + 1):
        block_prefix = f"block{exit_index}"
        exit_stage = f"exit{exit_index}_linear"
        cumulative += cost_df.loc[cost_df["stage"].str.startswith(block_prefix), "mac_proxy"].sum()
        cumulative += cost_df.loc[cost_df["stage"] == exit_stage, "mac_proxy"].sum()
        costs.append(cumulative)
    exit_costs = torch.tensor(costs, dtype=torch.float32)
    return cost_df, exit_costs


class MatrixMultiplicationProxy:
    def __init__(self, exit_macs):
        self.exit_macs = exit_macs.detach().float().cpu()
        self.final_exit_macs = float(self.exit_macs[-1].item())

    def from_exits(self, exits):
        exits_cpu = exits.detach().cpu().long().clamp(1, len(self.exit_macs))
        return self.exit_macs[exits_cpu - 1]

    def normalized_from_exits(self, exits):
        return self.from_exits(exits) / self.final_exit_macs

    def success_from_proxy(self, mac_proxy, label_preserved=None, threshold=OBSERVED_FINAL_COST_THRESHOLD):
        mac_tensor = torch.as_tensor(mac_proxy, dtype=torch.float32)
        forced_final = mac_tensor >= threshold * self.final_exit_macs
        if label_preserved is None:
            return forced_final
        return forced_final & torch.as_tensor(label_preserved, dtype=torch.bool)

    def describe(self):
        return {
            "proxy_type": "dynamic-path multiply-accumulate estimate",
            "unit": "approximate MACs / matrix multiplications",
            "exit_macs": [float(value) for value in self.exit_macs.tolist()],
            "exit_macs_millions": [float(value) / 1_000_000.0 for value in self.exit_macs.tolist()],
            "final_exit_macs": self.final_exit_macs,
            "final_exit_macs_millions": self.final_exit_macs / 1_000_000.0,
            "success_threshold_fraction_of_final_exit": float(OBSERVED_FINAL_COST_THRESHOLD),
        }


class BlackBoxEarlyExitOracle:
    def __init__(self, model, thresholds, exit_costs, device, cost_noise_std=0.0):
        self.model = model
        self.thresholds = thresholds
        self.exit_costs = exit_costs
        self.device = device
        self.cost_noise_std = float(cost_noise_std)
        self.total_queries = 0

    def reset_query_count(self):
        self.total_queries = 0

    @torch.no_grad()
    def query(self, images, for_evaluation=False, count=True):
        self.model.eval()
        if count:
            self.total_queries += int(images.size(0))
        images = images.to(self.device)
        logits_all = self.model.forward_all(images)
        exits, preds, confidences = dynamic_predictions_from_logits(logits_all, self.thresholds)
        final_probs = F.softmax(logits_all[-1], dim=1)
        final_conf, final_preds = final_probs.max(dim=1)
        observed_cost = self.exit_costs[(exits - 1).long()]
        if self.cost_noise_std > 0 and not for_evaluation:
            observed_cost = observed_cost + torch.randn_like(observed_cost) * self.cost_noise_std
        observed_cost = observed_cost.clamp(0.0, float(self.exit_costs[-1].item()) * 1.05)
        matmul_proxy = observed_cost.clone()
        result = {
            "pred": preds.detach().cpu(),
            "confidence": confidences.detach().cpu(),
            "exit": exits.detach().cpu(),
            "final_pred": final_preds.detach().cpu(),
            "final_confidence": final_conf.detach().cpu(),
            "observed_cost": observed_cost.detach().cpu() / float(self.exit_costs[-1].item()),
            "observed_matmul_proxy": matmul_proxy.detach().cpu(),
        }
        return result


@torch.no_grad()
def evaluate_adversarial_example_for_evaluator(model, base, delta, label, thresholds, exit_costs):
    model.eval()
    image = (base + delta).clamp(0.0, 1.0)
    logits_all = model.forward_all(image)
    exits, dyn_preds, dyn_confs = dynamic_predictions_from_logits(logits_all, thresholds)
    final_probs = F.softmax(logits_all[-1], dim=1)
    final_conf, final_preds = final_probs.max(dim=1)
    norm_cost = exit_costs[(exits - 1).long()] / exit_costs[-1]
    return {
        "exit": exits.detach(),
        "dyn_pred": dyn_preds.detach(),
        "dyn_confidence": dyn_confs.detach(),
        "final_pred": final_preds.detach(),
        "final_confidence": final_conf.detach(),
        "final_label_prob": final_probs[:, int(label)].detach(),
        "norm_cost": norm_cost.detach(),
    }


def canonical_norm_type(norm_type=None):
    norm = (norm_type or NORM_TYPE).lower().replace("_", "")
    aliases = {
        "inf": "linf",
        "infinity": "linf",
        "linfinity": "linf",
        "l0": "l0",
        "l1": "l1",
        "l2": "l2",
        "linf": "linf",
    }
    if norm not in aliases:
        raise ValueError(f"Unsupported norm type {norm_type!r}.")
    return aliases[norm]


def project_l1_ball(flat, epsilon):
    if epsilon <= 0:
        return torch.zeros_like(flat)
    abs_flat = flat.abs()
    l1_norm = abs_flat.sum(dim=1, keepdim=True)
    keep = l1_norm <= epsilon
    if keep.all():
        return flat
    sorted_abs, _ = torch.sort(abs_flat, dim=1, descending=True)
    cssv = sorted_abs.cumsum(dim=1) - epsilon
    indices = torch.arange(1, flat.size(1) + 1, device=flat.device, dtype=flat.dtype).view(1, -1)
    cond = sorted_abs - cssv / indices > 0
    rho = cond.sum(dim=1, keepdim=True).clamp_min(1) - 1
    theta = cssv.gather(1, rho) / (rho.to(flat.dtype) + 1.0)
    projected = flat.sign() * (abs_flat - theta).clamp_min(0.0)
    return torch.where(keep, flat, projected)


def project_l0_ball(delta, k, pixel_epsilon):
    flat = delta.flatten(1)
    k = int(max(0, min(round(float(k)), flat.size(1))))
    if k == 0:
        return torch.zeros_like(delta)
    flat = flat.clamp(-pixel_epsilon, pixel_epsilon)
    if k < flat.size(1):
        _, idx = torch.topk(flat.abs(), k=k, dim=1, largest=True)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        flat = torch.where(mask, flat, torch.zeros_like(flat))
    return flat.view_as(delta)


def project_delta(base, delta, epsilon, norm_type=None):
    norm = canonical_norm_type(norm_type)
    if norm == "linf":
        delta = delta.clamp(-epsilon, epsilon)
    elif norm == "l2":
        flat = delta.flatten(1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(float(epsilon) / norms, max=1.0)
        delta = (flat * scale).view_as(delta)
    elif norm == "l1":
        delta = project_l1_ball(delta.flatten(1), float(epsilon)).view_as(delta)
    elif norm == "l0":
        delta = project_l0_ball(delta, int(round(float(epsilon))), L0_PIXEL_EPSILON)
    else:
        raise AssertionError(f"Unhandled norm {norm}")
    return (base + delta).clamp(0.0, 1.0) - base


def coordinate_step_scale(base, epsilon, norm_type=None):
    norm = canonical_norm_type(norm_type)
    dims = int(np.prod(base.shape[1:]))
    if norm == "linf":
        return float(epsilon)
    if norm == "l2":
        return float(epsilon) / math.sqrt(max(dims, 1))
    if norm == "l1":
        return float(epsilon) / max(dims, 1)
    if norm == "l0":
        return float(L0_PIXEL_EPSILON)
    raise AssertionError(f"Unhandled norm {norm}")


def sample_deltas_like(base, n_particles, epsilon, norm_type=None):
    norm = canonical_norm_type(norm_type)
    shape = (n_particles, *base.shape[1:])
    if norm == "linf":
        deltas = torch.empty(shape, device=base.device).uniform_(-epsilon, epsilon)
    elif norm in {"l2", "l1"}:
        deltas = torch.randn(shape, device=base.device)
        radii = torch.rand((n_particles, 1, 1, 1), device=base.device).clamp_min(1e-6)
        deltas = deltas * radii
    elif norm == "l0":
        deltas = torch.empty(shape, device=base.device).uniform_(-L0_PIXEL_EPSILON, L0_PIXEL_EPSILON)
    else:
        raise AssertionError(f"Unhandled norm {norm}")
    return project_delta(base, deltas, epsilon, norm)


def perturbation_norm(deltas, norm_type=None):
    norm = canonical_norm_type(norm_type)
    flat = deltas.flatten(1)
    if norm == "linf":
        return flat.abs().max(dim=1).values
    if norm == "l2":
        return flat.norm(p=2, dim=1)
    if norm == "l1":
        return flat.abs().sum(dim=1)
    if norm == "l0":
        return (flat.abs() > 1e-8).float().sum(dim=1)
    raise AssertionError(f"Unhandled norm {norm}")


def normalized_perturbation_size(deltas, epsilon, norm_type=None):
    return perturbation_norm(deltas, norm_type) / max(float(epsilon), 1e-8)


def population_diversity(positions, epsilon=None):
    flat = positions.flatten(1)
    if flat.size(0) <= 1:
        return torch.zeros((), device=flat.device)
    diversity = torch.pdist(flat.detach().cpu(), p=2).mean().to(flat.device)
    if epsilon is not None:
        diversity = diversity / max(float(epsilon), 1e-8)
    return diversity


def _image_foreground_mask(image):
    img = image[0] if image.dim() == 4 else image
    channel = img[0] if img.dim() == 3 else img
    threshold = float(channel.mean().item())
    return channel > threshold


def compute_diversity_features_for_example(example):
    image = example["image"].detach().float().cpu()
    img = image[0, 0]
    height, width = img.shape
    mask = _image_foreground_mask(image)
    ys, xs = mask.nonzero(as_tuple=True)
    mass = float(img.mean().item())
    if ys.numel() > 0:
        y_min = float(ys.min().item())
        y_max = float(ys.max().item())
        x_min = float(xs.min().item())
        x_max = float(xs.max().item())
        bbox_height = (y_max - y_min + 1.0) / float(height)
        bbox_width = (x_max - x_min + 1.0) / float(width)
    else:
        bbox_height = 0.0
        bbox_width = 0.0
    total_mass = float(img.sum().item())
    if total_mass > 1e-12:
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=img.dtype),
            torch.arange(width, dtype=img.dtype),
            indexing="ij",
        )
        center_y = float((yy * img).sum().item() / total_mass) / max(height - 1, 1)
        center_x = float((xx * img).sum().item() / total_mass) / max(width - 1, 1)
        yy_c = yy - float(center_y * max(height - 1, 1))
        xx_c = xx - float(center_x * max(width - 1, 1))
        cov_xx = float((img * xx_c.pow(2)).sum().item() / total_mass)
        cov_yy = float((img * yy_c.pow(2)).sum().item() / total_mass)
        cov_xy = float((img * xx_c * yy_c).sum().item() / total_mass)
        trace = cov_xx + cov_yy
        det = max(cov_xx * cov_yy - cov_xy * cov_xy, 0.0)
        disc = max(trace * trace - 4.0 * det, 0.0)
        eig1 = 0.5 * (trace + math.sqrt(disc))
        eig2 = 0.5 * (trace - math.sqrt(disc))
        eccentricity = 0.0 if eig1 <= 1e-12 else float(1.0 - (max(eig2, 0.0) / eig1))
    else:
        center_y = 0.5
        center_x = 0.5
        eccentricity = 0.0
    hflip = torch.flip(img, dims=[1])
    vflip = torch.flip(img, dims=[0])
    horizontal_symmetry = float(1.0 - (img - hflip).abs().mean().item())
    vertical_symmetry = float(1.0 - (img - vflip).abs().mean().item())
    dy = img[1:, :] - img[:-1, :]
    dx = img[:, 1:] - img[:, :-1]
    edge_density = float((dx.abs().mean().item() + dy.abs().mean().item()) / 2.0)
    hist = torch.histc(img, bins=16, min=0.0, max=1.0)
    probs = hist / hist.sum().clamp_min(1e-12)
    entropy = float((-(probs * probs.clamp_min(1e-12).log2())).sum().item())
    label = int(example["label"])
    row = {f"label_{class_idx}": 1.0 if class_idx == label else 0.0 for class_idx in range(10)}
    row.update({
        "foreground_mass": mass,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "center_x": center_x,
        "center_y": center_y,
        "horizontal_symmetry": horizontal_symmetry,
        "vertical_symmetry": vertical_symmetry,
        "edge_density": edge_density,
        "entropy": entropy,
        "eccentricity": eccentricity,
    })
    return row


def build_diversity_feature_frame(examples, pool_indices):
    rows = []
    for input_id in pool_indices:
        example = examples[int(input_id)]
        feature_row = compute_diversity_features_for_example(example)
        feature_row["input_id"] = int(input_id)
        feature_row["source_index"] = int(example["source_index"])
        feature_row["label"] = int(example["label"])
        rows.append(feature_row)
    frame = pd.DataFrame(rows).sort_values("input_id").reset_index(drop=True)
    numeric_cols = [col for col in frame.columns if col not in {"input_id", "source_index", "label"} and not col.startswith("label_")]
    normalized = frame.copy()
    for col in numeric_cols:
        values = normalized[col].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            normalized[col] = 0.0
        else:
            normalized[col] = (values - mean) / std
    feature_cols = [col for col in normalized.columns if col not in {"input_id", "source_index", "label"}]
    return frame, normalized, feature_cols


def select_diverse_crafting_indices(examples, pool_indices, m_value):
    raw_frame, normalized_frame, feature_cols = build_diversity_feature_frame(examples, pool_indices)
    features = normalized_frame[feature_cols].to_numpy(dtype=float)
    if len(features) < int(m_value):
        raise ValueError(f"Cannot select M={m_value} from pool of size {len(features)}")
    medoid_idx = int(np.argmin(np.linalg.norm(features - features.mean(axis=0, keepdims=True), axis=1)))
    selected = [medoid_idx]
    remaining = set(range(len(features)))
    remaining.remove(medoid_idx)
    while len(selected) < int(m_value):
        best_idx = None
        best_distance = -float("inf")
        for candidate in sorted(remaining):
            min_distance = min(float(np.linalg.norm(features[candidate] - features[chosen])) for chosen in selected)
            if (min_distance > best_distance + 1e-12) or (abs(min_distance - best_distance) <= 1e-12 and candidate < (best_idx if best_idx is not None else candidate + 1)):
                best_distance = min_distance
                best_idx = candidate
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    selected_frame = raw_frame.iloc[selected].copy().reset_index(drop=True)
    selected_ids = selected_frame["input_id"].astype(int).tolist()
    metadata = {
        "crafting_pool_size": int(len(pool_indices)),
        "selected_input_ids": selected_ids,
        "selected_source_indices": selected_frame["source_index"].astype(int).tolist(),
        "selected_labels": selected_frame["label"].astype(int).tolist(),
        "selection_method": "greedy_farthest_first_from_medoid",
        "feature_columns": feature_cols,
    }
    return selected_ids, selected_frame, raw_frame, metadata


@torch.no_grad()
def evaluate_universal_candidate_deltas(oracle, examples, deltas, epsilon):
    if deltas is None or deltas.numel() == 0:
        return None
    device = deltas.device
    num_deltas = int(deltas.size(0))
    all_fitness = []
    all_label_match = []
    all_observed_cost = []
    all_proxy = []
    all_confidence = []
    all_final_confidence = []
    all_early_uncertainty = []
    all_perturb_fraction = []
    all_success = []
    perturb_fraction = normalized_perturbation_size(deltas, epsilon).detach().cpu()
    perturb_l2 = deltas.flatten(1).norm(p=2, dim=1).detach().cpu()
    for example in examples:
        base = example["image"].to(device)
        clean_pred = int(example["clean_blackbox_pred"])
        expanded_base = base.expand(num_deltas, -1, -1, -1)
        batch_deltas = project_delta(expanded_base, deltas, epsilon)
        q = oracle.query((expanded_base + batch_deltas).clamp(0.0, 1.0), for_evaluation=False, count=True)
        pred = q["pred"]
        confidence = q["confidence"].to(device)
        observed_cost = q["observed_cost"].to(device).clamp(0.0, 1.05)
        observed_proxy = q["observed_matmul_proxy"].to(device)
        clean_pred_tensor = torch.full_like(pred, clean_pred)
        label_match = (pred == clean_pred_tensor).to(device).float()
        confidence_on_clean_label = confidence * label_match
        final_confidence_on_clean_label = observed_cost.clamp(0.0, 1.0) * confidence_on_clean_label
        early_uncertainty = (1.0 - observed_cost.clamp(0.0, 1.0)) * (1.0 - confidence)
        fitness = (
            BLACKBOX_FITNESS_WEIGHTS["observed_cost"] * observed_cost
            + BLACKBOX_FITNESS_WEIGHTS["label_match"] * label_match
            + BLACKBOX_FITNESS_WEIGHTS["final_confidence_on_clean_label"] * final_confidence_on_clean_label
            + BLACKBOX_FITNESS_WEIGHTS["early_uncertainty"] * early_uncertainty
            + BLACKBOX_FITNESS_WEIGHTS["perturb_norm_fraction"] * perturb_fraction.to(device)
        )
        all_fitness.append(fitness.detach().cpu())
        all_label_match.append(label_match.detach().cpu())
        all_observed_cost.append(observed_cost.detach().cpu())
        all_proxy.append(observed_proxy.detach().cpu())
        all_confidence.append(confidence.detach().cpu())
        all_final_confidence.append(final_confidence_on_clean_label.detach().cpu())
        all_early_uncertainty.append(early_uncertainty.detach().cpu())
        all_perturb_fraction.append(perturb_fraction.clone())
        all_success.append(((observed_cost >= OBSERVED_FINAL_COST_THRESHOLD) & (label_match > 0.5)).detach().cpu())
    fitness_matrix = torch.stack(all_fitness, dim=0)
    return {
        "fitness_matrix": fitness_matrix,
        "label_match_matrix": torch.stack(all_label_match, dim=0),
        "observed_cost_matrix": torch.stack(all_observed_cost, dim=0),
        "proxy_matrix": torch.stack(all_proxy, dim=0),
        "confidence_matrix": torch.stack(all_confidence, dim=0),
        "final_confidence_matrix": torch.stack(all_final_confidence, dim=0),
        "early_uncertainty_matrix": torch.stack(all_early_uncertainty, dim=0),
        "perturb_fraction_matrix": torch.stack(all_perturb_fraction, dim=0),
        "success_matrix": torch.stack(all_success, dim=0),
        "perturb_l2": perturb_l2,
    }


def universal_weights_from_best(fitness_values):
    values = torch.as_tensor(fitness_values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        return values
    minimum = values.min()
    maximum = values.max()
    if float(maximum - minimum) <= 1e-12:
        return torch.full_like(values, 1.0 / float(values.numel()))
    normalized = (values - minimum) / (maximum - minimum)
    weights = 1.0 - normalized
    weight_sum = weights.sum()
    if float(weight_sum.item()) <= 1e-12:
        return torch.full_like(values, 1.0 / float(values.numel()))
    return weights / weight_sum


def aggregate_universal_fitness(universal_metrics, weights):
    weights_tensor = torch.as_tensor(weights, dtype=universal_metrics["fitness_matrix"].dtype).view(-1, 1)
    return (universal_metrics["fitness_matrix"] * weights_tensor).sum(dim=0)


def _slice_universal_metrics(universal_metrics, delta_index):
    return {
        "fitness_per_image": universal_metrics["fitness_matrix"][:, delta_index].detach().cpu(),
        "label_match_per_image": universal_metrics["label_match_matrix"][:, delta_index].detach().cpu(),
        "observed_cost_per_image": universal_metrics["observed_cost_matrix"][:, delta_index].detach().cpu(),
        "proxy_per_image": universal_metrics["proxy_matrix"][:, delta_index].detach().cpu(),
        "confidence_per_image": universal_metrics["confidence_matrix"][:, delta_index].detach().cpu(),
        "final_confidence_per_image": universal_metrics["final_confidence_matrix"][:, delta_index].detach().cpu(),
        "early_uncertainty_per_image": universal_metrics["early_uncertainty_matrix"][:, delta_index].detach().cpu(),
        "perturb_fraction_per_image": universal_metrics["perturb_fraction_matrix"][:, delta_index].detach().cpu(),
        "success_per_image": universal_metrics["success_matrix"][:, delta_index].detach().cpu(),
        "perturb_l2": float(universal_metrics["perturb_l2"][delta_index].item()),
    }


@torch.no_grad()
def universal_ga_weighted_attack(oracle, examples, epsilon, n_particles, n_iterations, rng_seed=None, hyperparams=None):
    set_global_seed(rng_seed or 0)
    params = {**GENETIC_PARAMS, **(hyperparams or {})}
    base = examples[0]["image"].to(oracle.device)
    positions = sample_deltas_like(base, n_particles, epsilon)
    best_score = -float("inf")
    best_delta = positions[:1].detach().clone()
    best_metrics = None
    fitness_history = []
    success_history = []
    weight_history = []
    diversity_history = []
    weights = torch.full((len(examples),), 1.0 / max(len(examples), 1), dtype=torch.float32)
    query_start = int(oracle.total_queries)
    elite_fraction = float(params.get("elite_fraction", 0.18))
    tournament_size = int(params.get("tournament_size", 3))
    crossover_rate = float(params.get("crossover_rate", 0.55))
    mutation_rate = float(params.get("mutation_rate", 0.18))
    mutation_scale = float(params.get("mutation_scale", 0.75))
    random_immigrant_fraction = float(params.get("random_immigrant_fraction", 0.18))
    local_immigrant_fraction = float(params.get("local_immigrant_fraction", 0.45))
    local_mutation_scale = float(params.get("local_mutation_scale", 0.35))

    def tournament_select(scores, count):
        competitors = torch.randint(0, n_particles, (count, tournament_size), device=scores.device)
        competitor_scores = scores[competitors]
        best = competitor_scores.argmax(dim=1, keepdim=True)
        return competitors.gather(1, best).flatten()

    for iteration in range(n_iterations):
        if iteration > 0 and best_metrics is not None:
            weights = universal_weights_from_best(best_metrics["fitness_per_image"])
        metrics = evaluate_universal_candidate_deltas(oracle, examples, positions, epsilon)
        scores = aggregate_universal_fitness(metrics, weights)
        diversity_history.append(float(population_diversity(positions, epsilon=epsilon).item()))
        idx = int(torch.argmax(scores).item())
        score = float(scores[idx].item())
        if score > best_score + 1e-12:
            best_score = score
            best_delta = positions[idx:idx + 1].detach().clone()
            best_metrics = _slice_universal_metrics(metrics, idx)
        fitness_history.append(best_score)
        success_history.append(float(metrics["success_matrix"][:, idx].float().mean().item()))
        weight_history.append(weights.detach().cpu().tolist())
        elite_count = max(1, min(n_particles, int(round(elite_fraction * n_particles))))
        immigrant_count = max(0, min(n_particles - elite_count, int(round(random_immigrant_fraction * n_particles))))
        child_count = n_particles - elite_count - immigrant_count
        elite_idx = torch.topk(scores, k=elite_count, largest=True).indices
        next_positions = [positions[elite_idx].detach().clone()]
        if child_count > 0:
            parent1 = positions[tournament_select(scores, child_count)]
            parent2 = positions[tournament_select(scores, child_count)]
            mask = torch.rand_like(parent1) < crossover_rate
            children = torch.where(mask, parent1, parent2)
            progress = 0.0 if n_iterations <= 1 else iteration / float(n_iterations - 1)
            annealed_mutation = mutation_scale * (1.0 - 0.60 * progress) + 0.05
            noise = torch.randn_like(children) * (coordinate_step_scale(base, epsilon) * annealed_mutation)
            mutation_mask = (torch.rand_like(children) < mutation_rate).float()
            children = project_delta(base.expand(child_count, -1, -1, -1), children + mutation_mask * noise, epsilon)
            next_positions.append(children)
        if immigrant_count > 0:
            local_count = min(int(round(local_immigrant_fraction * immigrant_count)), immigrant_count)
            uniform_count = immigrant_count - local_count
            immigrants = []
            if local_count > 0:
                elite_pick = torch.randint(0, elite_count, (local_count,), device=elite_idx.device)
                source_idx = elite_idx[elite_pick]
                source = positions[source_idx]
                local_noise = torch.randn((local_count, *base.shape[1:]), device=base.device) * (coordinate_step_scale(base, epsilon) * local_mutation_scale)
                immigrants.append(project_delta(base.expand(local_count, -1, -1, -1), source + local_noise, epsilon))
            if uniform_count > 0:
                immigrants.append(sample_deltas_like(base, uniform_count, epsilon))
            next_positions.append(torch.cat(immigrants, dim=0))
        positions = torch.cat(next_positions, dim=0)[:n_particles]
    return {
        "delta": best_delta.detach().cpu(),
        "fitness_history": fitness_history,
        "crafting_success_history": success_history,
        "weight_history": weight_history,
        "diversity_history": diversity_history,
        "best_metrics": best_metrics,
        "queries": int(oracle.total_queries - query_start),
    }


@torch.no_grad()
def universal_pso_jitter_weighted_attack(oracle, examples, epsilon, n_particles, n_iterations, rng_seed=None, hyperparams=None):
    set_global_seed(rng_seed or 0)
    params = {**PSO_JITTER_PARAMS, **(hyperparams or {})}
    base = examples[0]["image"].to(oracle.device)
    positions = sample_deltas_like(base, n_particles, epsilon)
    velocity_scale = coordinate_step_scale(base, epsilon) * float(params.get("initial_velocity_fraction", INITIAL_VELOCITY_FRACTION))
    velocities = torch.empty_like(positions).uniform_(-velocity_scale, velocity_scale)
    pbest_positions = positions.detach().clone()
    pbest_scores = torch.full((n_particles,), -float("inf"), device=base.device)
    gbest_position = positions[:1].detach().clone()
    gbest_score = -float("inf")
    gbest_metrics = None
    fitness_history = []
    success_history = []
    weight_history = []
    diversity_history = []
    weights = torch.full((len(examples),), 1.0 / max(len(examples), 1), dtype=torch.float32)
    query_start = int(oracle.total_queries)
    inertia = float(params.get("inertia", 0.8))
    cognitive = float(params.get("cognitive", 1.1))
    social = float(params.get("social", 1.6))
    jitter = float(params.get("jitter", 0.01))
    vmax = coordinate_step_scale(base, epsilon) * float(params.get("max_velocity_fraction", 0.35))
    for iteration in range(n_iterations):
        if iteration > 0 and gbest_metrics is not None:
            weights = universal_weights_from_best(gbest_metrics["fitness_per_image"])
        metrics = evaluate_universal_candidate_deltas(oracle, examples, positions, epsilon)
        scores = aggregate_universal_fitness(metrics, weights).to(base.device)
        diversity_history.append(float(population_diversity(positions, epsilon=epsilon).item()))
        improved = scores > pbest_scores
        pbest_scores[improved] = scores[improved]
        pbest_positions[improved] = positions[improved].detach().clone()
        idx = int(torch.argmax(pbest_scores).item())
        score = float(pbest_scores[idx].item())
        if score > gbest_score + 1e-12:
            gbest_score = score
            gbest_position = pbest_positions[idx:idx + 1].detach().clone()
            gbest_metrics = _slice_universal_metrics(metrics, idx)
        fitness_history.append(gbest_score)
        success_history.append(float(metrics["success_matrix"][:, idx].float().mean().item()))
        weight_history.append(weights.detach().cpu().tolist())
        r1 = torch.rand_like(positions)
        r2 = torch.rand_like(positions)
        velocities = inertia * velocities + cognitive * r1 * (pbest_positions - positions) + social * r2 * (gbest_position - positions)
        if jitter > 0:
            velocities = velocities + torch.randn_like(velocities) * (coordinate_step_scale(base, epsilon) * jitter)
        velocities = velocities.clamp(-vmax, vmax)
        positions = project_delta(base, positions + velocities, epsilon)
    return {
        "delta": gbest_position.detach().cpu(),
        "fitness_history": fitness_history,
        "crafting_success_history": success_history,
        "weight_history": weight_history,
        "diversity_history": diversity_history,
        "best_metrics": gbest_metrics,
        "queries": int(oracle.total_queries - query_start),
    }


@torch.no_grad()
def universal_pso_jitter_multiswarm_attack(oracle, examples, epsilon, n_particles, n_iterations, rng_seed=None, hyperparams=None):
    set_global_seed(rng_seed or 0)
    params = {**PSO_JITTER_PARAMS, **(hyperparams or {})}
    bases = [example["image"].to(oracle.device) for example in examples]
    num_swarms = len(examples)
    swarms = []
    query_start = int(oracle.total_queries)
    for swarm_idx, (example, base) in enumerate(zip(examples, bases)):
        seed = int(rng_seed or 0) + swarm_idx * 1009
        positions, velocities = make_initial_swarm_state(
            base=base,
            epsilon=epsilon,
            n_particles=n_particles,
            seed=seed,
            initial_velocity_fraction=float(params.get("initial_velocity_fraction", INITIAL_VELOCITY_FRACTION)),
        )
        swarms.append({
            "example": example,
            "base": base,
            "positions": positions,
            "velocities": velocities,
            "pbest_positions": positions.detach().clone(),
            "pbest_scores": torch.full((n_particles,), -float("inf"), device=base.device),
            "gbest_position": positions[:1].detach().clone(),
            "gbest_score": -float("inf"),
            "gbest_metrics": None,
            "prev_state": None,
        })
    delta_universal = torch.stack([swarm["gbest_position"][0] for swarm in swarms], dim=0).median(dim=0).values.unsqueeze(0)
    best_consensus = delta_universal.detach().clone()
    best_consensus_score = -float("inf")
    diversity_history = []
    consensus_history = []
    success_history = []
    inertia = float(params.get("inertia", 0.8))
    cognitive = float(params.get("cognitive", 1.1))
    social = float(params.get("social", 1.6))
    jitter = float(params.get("jitter", 0.01))
    vmax = coordinate_step_scale(bases[0], epsilon) * float(params.get("max_velocity_fraction", 0.35))
    for iteration in range(n_iterations):
        progress = 0.0 if n_iterations <= 1 else iteration / float(n_iterations - 1)
        c3 = UNIVERSAL_C3_START + (UNIVERSAL_C3_END - UNIVERSAL_C3_START) * progress
        generation_diversity = []
        for swarm in swarms:
            base = swarm["base"]
            positions = swarm["positions"]
            velocities = swarm["velocities"]
            pbest_positions = swarm["pbest_positions"]
            pbest_scores = swarm["pbest_scores"]
            metrics = evaluate_blackbox_candidate_deltas(
                oracle=oracle,
                base=base,
                clean_blackbox_pred=int(swarm["example"]["clean_blackbox_pred"]),
                deltas=positions,
                epsilon=epsilon,
                budget=None,
            )
            scores = metrics["fitness"].to(base.device)
            improved = scores > pbest_scores
            pbest_scores[improved] = scores[improved]
            pbest_positions[improved] = positions[improved].detach().clone()
            idx = int(torch.argmax(pbest_scores).item())
            score = float(pbest_scores[idx].item())
            if score > swarm["gbest_score"] + 1e-12:
                swarm["gbest_score"] = score
                swarm["gbest_position"] = pbest_positions[idx:idx + 1].detach().clone()
                swarm["gbest_metrics"] = slice_metric_dict(metrics, idx)
            r1 = torch.rand_like(positions)
            r2 = torch.rand_like(positions)
            r3 = torch.rand_like(positions)
            velocities = (
                inertia * velocities
                + cognitive * r1 * (pbest_positions - positions)
                + social * r2 * (swarm["gbest_position"] - positions)
                + c3 * r3 * (delta_universal.to(base.device) - positions)
            )
            if jitter > 0:
                velocities = velocities + torch.randn_like(velocities) * (coordinate_step_scale(base, epsilon) * jitter)
            velocities = velocities.clamp(-vmax, vmax)
            positions = project_delta(base, positions + velocities, epsilon)
            swarm["positions"] = positions
            swarm["velocities"] = velocities
            generation_diversity.append(float(population_diversity(positions, epsilon=epsilon).item()))
        diversity_history.append(float(np.mean(generation_diversity)) if generation_diversity else 0.0)
        if ((iteration + 1) % UNIVERSAL_CONSENSUS_PERIOD) == 0 or iteration == n_iterations - 1:
            candidate_pool = [swarm["gbest_position"].detach().clone() for swarm in swarms] + [delta_universal.detach().clone()]
            candidate_stack = torch.cat(candidate_pool, dim=0)
            universal_metrics = evaluate_universal_candidate_deltas(oracle, examples, candidate_stack, epsilon)
            total_scores = universal_metrics["fitness_matrix"].sum(dim=0)
            best_idx = int(torch.argmax(total_scores).item())
            delta_universal = candidate_stack[best_idx:best_idx + 1].detach().clone()
            consensus_score = float(total_scores[best_idx].item())
            if consensus_score > best_consensus_score + 1e-12:
                best_consensus_score = consensus_score
                best_consensus = delta_universal.detach().clone()
            consensus_history.append(consensus_score)
            success_history.append(float(universal_metrics["success_matrix"][:, best_idx].float().mean().item()))
    median_candidate = torch.stack([swarm["gbest_position"][0].detach().clone() for swarm in swarms], dim=0).median(dim=0).values.unsqueeze(0)
    final_candidates = torch.cat([median_candidate, best_consensus], dim=0)
    final_metrics = evaluate_universal_candidate_deltas(oracle, examples, final_candidates, epsilon)
    final_scores = final_metrics["fitness_matrix"].sum(dim=0)
    final_idx = int(torch.argmax(final_scores).item())
    chosen = final_candidates[final_idx:final_idx + 1].detach().cpu()
    return {
        "delta": chosen,
        "fitness_history": consensus_history,
        "crafting_success_history": success_history,
        "weight_history": None,
        "diversity_history": diversity_history,
        "best_metrics": _slice_universal_metrics(final_metrics, final_idx),
        "queries": int(oracle.total_queries - query_start),
        "num_swarms": num_swarms,
    }


def universal_apso_weighted_attack(*args, **kwargs):
    return universal_pso_jitter_weighted_attack(*args, **kwargs)


def universal_apso_multiswarm_attack(*args, **kwargs):
    return universal_pso_jitter_multiswarm_attack(*args, **kwargs)


def make_initial_swarm_state(base, epsilon, n_particles, seed, initial_velocity_fraction=INITIAL_VELOCITY_FRACTION):
    set_global_seed(seed)
    positions = sample_deltas_like(base, n_particles, epsilon)
    velocity_scale = coordinate_step_scale(base, epsilon) * initial_velocity_fraction
    velocities = torch.empty_like(positions).uniform_(-velocity_scale, velocity_scale)
    return positions.detach().clone(), velocities.detach().clone()


def slice_metric_dict(metrics, idx):
    sliced = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            sliced[key] = value[idx:idx + 1].detach().clone()
        else:
            sliced[key] = value
    return sliced


class AttackBudget:
    def __init__(self, mode: str, query_budget: int | None = None, wall_clock_budget_s: float | None = None):
        self.mode = mode
        self.query_budget = query_budget
        self.wall_clock_budget_s = wall_clock_budget_s
        self.queries_used = 0
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def elapsed(self):
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    def expired(self):
        if self.mode == "query":
            return self.query_budget is not None and self.queries_used >= self.query_budget
        if self.wall_clock_budget_s is None:
            return False
        return self.start_time is not None and self.elapsed() >= self.wall_clock_budget_s

    def can_consume(self, batch_size=1):
        if self.mode == "query":
            if self.query_budget is None:
                return True
            return self.queries_used + batch_size <= self.query_budget
        return not self.expired()

    def consume(self, batch_size=1):
        self.queries_used += int(batch_size)


@torch.no_grad()
def evaluate_blackbox_candidate_deltas(oracle, base, clean_blackbox_pred, deltas, epsilon, budget: AttackBudget | None = None):
    if budget is not None and not budget.can_consume(deltas.size(0)):
        return None
    images = (base + deltas).clamp(0.0, 1.0)
    q = oracle.query(images, for_evaluation=False, count=True)
    if budget is not None:
        budget.consume(deltas.size(0))

    target_device = deltas.device
    pred = q["pred"]
    confidence = q["confidence"].to(target_device)
    observed_cost = q["observed_cost"].to(target_device).clamp(0.0, 1.05)
    observed_matmul_proxy = q["observed_matmul_proxy"].to(target_device)
    clean_pred_tensor = torch.full_like(pred, int(clean_blackbox_pred))
    label_match = (pred == clean_pred_tensor).to(target_device).float()
    confidence_on_clean_label = confidence * label_match
    final_confidence_on_clean_label = observed_cost.clamp(0.0, 1.0) * confidence_on_clean_label
    early_uncertainty = (1.0 - observed_cost.clamp(0.0, 1.0)) * (1.0 - confidence)
    perturb_fraction = normalized_perturbation_size(deltas, epsilon)
    perturb_l2 = deltas.flatten(1).div(max(float(epsilon), 1e-8)).pow(2).mean(dim=1).sqrt()
    fitness = (
        BLACKBOX_FITNESS_WEIGHTS["observed_cost"] * observed_cost
        + BLACKBOX_FITNESS_WEIGHTS["label_match"] * label_match
        + BLACKBOX_FITNESS_WEIGHTS["final_confidence_on_clean_label"] * final_confidence_on_clean_label
        + BLACKBOX_FITNESS_WEIGHTS["early_uncertainty"] * early_uncertainty
        + BLACKBOX_FITNESS_WEIGHTS["perturb_norm_fraction"] * perturb_fraction
    )
    return {
        "fitness": fitness,
        "pred": pred,
        "confidence": confidence.detach().cpu(),
        "observed_cost": observed_cost.detach().cpu(),
        "observed_matmul_proxy": observed_matmul_proxy.detach().cpu(),
        "label_match": label_match.bool(),
        "confidence_on_clean_label": confidence_on_clean_label.detach().cpu(),
        "final_confidence_on_clean_label": final_confidence_on_clean_label.detach().cpu(),
        "early_uncertainty": early_uncertainty.detach().cpu(),
        "perturb_norm_fraction": perturb_fraction.detach().cpu(),
        "perturb_l2_actual": deltas.flatten(1).norm(p=2, dim=1).detach().cpu(),
        "perturb_l2": perturb_l2.detach().cpu(),
    }


def _attack_return(best_delta, best_metrics, history, budget: AttackBudget):
    return {
        "delta": best_delta,
        "image": (best_delta * 0 + 1),  # overwritten by caller; placeholder keeps structure simple
        "metrics": best_metrics,
        "history": history,
        "queries": int(budget.queries_used),
        "elapsed_seconds": float(budget.elapsed()),
    }


@torch.no_grad()
def random_search_attack(
    oracle,
    base,
    clean_blackbox_pred,
    epsilon,
    n_particles,
    n_iterations,
    initial_positions=None,
    initial_velocities=None,
    rng_seed=None,
    hyperparams=None,
    budget: AttackBudget | None = None,
    progress_callback=None,
):
    set_global_seed(rng_seed or 0)
    history = []
    diversity_history = []
    best_score = -float("inf")
    best_delta = None
    best_metrics = None
    for iteration in range(n_iterations):
        if budget is not None and budget.expired():
            break
        if iteration == 0 and initial_positions is not None:
            deltas = initial_positions.clone()
        else:
            deltas = sample_deltas_like(base, n_particles, epsilon)
        metrics = evaluate_blackbox_candidate_deltas(oracle, base, clean_blackbox_pred, deltas, epsilon, budget=budget)
        if metrics is None:
            break
        diversity_history.append(float(population_diversity(deltas, epsilon=epsilon).item()))
        idx = int(torch.argmax(metrics["fitness"]).item())
        score = float(metrics["fitness"][idx].item())
        if score > best_score:
            best_score = score
            best_delta = deltas[idx:idx + 1].detach().clone()
            best_metrics = slice_metric_dict(metrics, idx)
        history.append(best_score)
        if progress_callback is not None and best_delta is not None:
            progress_callback(best_delta=best_delta, best_metrics=best_metrics, queries_used=int(budget.queries_used if budget is not None else (iteration + 1) * n_particles), elapsed_seconds=float(budget.elapsed() if budget is not None else 0.0))
    return {
        "delta": best_delta,
        "image": (base + best_delta).clamp(0.0, 1.0).detach() if best_delta is not None else base.detach().clone(),
        "metrics": best_metrics,
        "history": history,
        "diversity_history": diversity_history,
        "queries": int(budget.queries_used if budget is not None else n_particles * n_iterations),
        "elapsed_seconds": float(budget.elapsed() if budget is not None else 0.0),
    }


@torch.no_grad()
def pso_search_attack(
    oracle,
    base,
    clean_blackbox_pred,
    epsilon,
    n_particles,
    n_iterations,
    initial_positions=None,
    initial_velocities=None,
    rng_seed=None,
    hyperparams=None,
    budget: AttackBudget | None = None,
    progress_callback=None,
):
    set_global_seed(rng_seed or 0)
    params = {**PSO_PARAMS, **(hyperparams or {})}
    positions = initial_positions.clone() if initial_positions is not None else sample_deltas_like(base, n_particles, epsilon)
    velocity_scale = coordinate_step_scale(base, epsilon) * float(params.get("initial_velocity_fraction", INITIAL_VELOCITY_FRACTION))
    velocities = initial_velocities.clone() if initial_velocities is not None else torch.empty_like(positions).uniform_(-velocity_scale, velocity_scale)
    pbest_positions = positions.detach().clone()
    pbest_scores = torch.full((n_particles,), -float("inf"), device=base.device)
    gbest_position = positions[:1].detach().clone()
    gbest_score = -float("inf")
    gbest_metrics = None
    history = []
    diversity_history = []
    vmax = coordinate_step_scale(base, epsilon) * float(params.get("max_velocity_fraction", 0.30))
    inertia = float(params.get("inertia", 0.62))
    cognitive = float(params.get("cognitive", 1.35))
    social = float(params.get("social", 1.35))
    jitter = float(params.get("jitter", 0.0))
    for iteration in range(n_iterations):
        if budget is not None and budget.expired():
            break
        metrics = evaluate_blackbox_candidate_deltas(oracle, base, clean_blackbox_pred, positions, epsilon, budget=budget)
        if metrics is None:
            break
        diversity_history.append(float(population_diversity(positions, epsilon=epsilon).item()))
        scores = metrics["fitness"]
        improved = scores > pbest_scores
        pbest_scores[improved] = scores[improved]
        pbest_positions[improved] = positions[improved].detach().clone()
        idx = int(torch.argmax(scores).item())
        score = float(scores[idx].item())
        if score > gbest_score:
            gbest_score = score
            gbest_position = positions[idx:idx + 1].detach().clone()
            gbest_metrics = slice_metric_dict(metrics, idx)
        history.append(gbest_score)
        if progress_callback is not None and gbest_position is not None:
            progress_callback(best_delta=gbest_position, best_metrics=gbest_metrics, queries_used=int(budget.queries_used if budget is not None else (iteration + 1) * n_particles), elapsed_seconds=float(budget.elapsed() if budget is not None else 0.0))
        r1 = torch.rand_like(positions)
        r2 = torch.rand_like(positions)
        velocities = inertia * velocities + cognitive * r1 * (pbest_positions - positions) + social * r2 * (gbest_position - positions)
        if jitter > 0:
            velocities = velocities + torch.randn_like(velocities) * (coordinate_step_scale(base, epsilon) * jitter)
        velocities = velocities.clamp(-vmax, vmax)
        positions = project_delta(base, positions + velocities, epsilon)
    return {
        "delta": gbest_position,
        "image": (base + gbest_position).clamp(0.0, 1.0).detach() if gbest_position is not None else base.detach().clone(),
        "metrics": gbest_metrics,
        "history": history,
        "diversity_history": diversity_history,
        "queries": int(budget.queries_used if budget is not None else n_particles * n_iterations),
        "elapsed_seconds": float(budget.elapsed() if budget is not None else 0.0),
    }


def pso_jitter_search_attack(*args, **kwargs):
    hyperparams = {**(kwargs.get("hyperparams") or {}), "jitter": float((kwargs.get("hyperparams") or {}).get("jitter", 0.010))}
    kwargs["hyperparams"] = hyperparams
    return pso_search_attack(*args, **kwargs)


@torch.no_grad()
def genetic_search_attack(
    oracle,
    base,
    clean_blackbox_pred,
    epsilon,
    n_particles,
    n_iterations,
    initial_positions=None,
    initial_velocities=None,
    rng_seed=None,
    hyperparams=None,
    budget: AttackBudget | None = None,
    progress_callback=None,
):
    set_global_seed(rng_seed or 0)
    params = {**GENETIC_PARAMS, **(hyperparams or {})}
    positions = initial_positions.clone() if initial_positions is not None else sample_deltas_like(base, n_particles, epsilon)
    best_score = -float("inf")
    best_delta = None
    best_metrics = None
    history = []
    diversity_history = []
    elite_fraction = float(params.get("elite_fraction", 0.18))
    tournament_size = int(params.get("tournament_size", 3))
    crossover_rate = float(params.get("crossover_rate", 0.55))
    mutation_rate = float(params.get("mutation_rate", 0.18))
    mutation_scale = float(params.get("mutation_scale", 0.75))
    random_immigrant_fraction = float(params.get("random_immigrant_fraction", 0.18))
    local_immigrant_fraction = float(params.get("local_immigrant_fraction", 0.45))
    local_mutation_scale = float(params.get("local_mutation_scale", 0.35))

    def tournament_select(scores, count):
        competitors = torch.randint(0, n_particles, (count, tournament_size), device=base.device)
        competitor_scores = scores[competitors]
        winners = competitor_scores.argmax(dim=1)
        return competitors[torch.arange(count, device=base.device), winners]

    for iteration in range(n_iterations):
        if budget is not None and budget.expired():
            break
        metrics = evaluate_blackbox_candidate_deltas(oracle, base, clean_blackbox_pred, positions, epsilon, budget=budget)
        if metrics is None:
            break
        diversity_history.append(float(population_diversity(positions, epsilon=epsilon).item()))
        scores = metrics["fitness"]
        idx = int(torch.argmax(scores).item())
        score = float(scores[idx].item())
        if score > best_score:
            best_score = score
            best_delta = positions[idx:idx + 1].detach().clone()
            best_metrics = slice_metric_dict(metrics, idx)
        history.append(best_score)
        if progress_callback is not None and best_delta is not None:
            progress_callback(best_delta=best_delta, best_metrics=best_metrics, queries_used=int(budget.queries_used if budget is not None else (iteration + 1) * n_particles), elapsed_seconds=float(budget.elapsed() if budget is not None else 0.0))

        elite_count = max(1, min(n_particles, int(round(elite_fraction * n_particles))))
        immigrant_count = max(0, min(n_particles - elite_count, int(round(random_immigrant_fraction * n_particles))))
        child_count = n_particles - elite_count - immigrant_count
        elite_idx = torch.topk(scores, k=elite_count, largest=True).indices
        next_positions = [positions[elite_idx].detach().clone()]
        if child_count > 0:
            parent1 = positions[tournament_select(scores, child_count)]
            parent2 = positions[tournament_select(scores, child_count)]
            mask = torch.rand_like(parent1) < crossover_rate
            children = torch.where(mask, parent1, parent2)
            mutation_mask = (torch.rand_like(children) < mutation_rate).float()
            progress = 0.0 if n_iterations <= 1 else iteration / float(n_iterations - 1)
            annealed_mutation = mutation_scale * (1.0 - 0.60 * progress) + 0.05
            noise = torch.randn_like(children) * (coordinate_step_scale(base, epsilon) * annealed_mutation)
            children = project_delta(base, children + mutation_mask * noise, epsilon)
            next_positions.append(children)
        if immigrant_count > 0:
            local_count = int(round(local_immigrant_fraction * immigrant_count))
            local_count = min(local_count, immigrant_count)
            uniform_count = immigrant_count - local_count
            immigrants = []
            if local_count > 0:
                source_idx = elite_idx[torch.randint(0, elite_count, (local_count,), device=base.device)]
                source = positions[source_idx]
                local_noise = torch.randn((local_count, *base.shape[1:]), device=base.device) * (coordinate_step_scale(base, epsilon) * local_mutation_scale)
                immigrants.append(project_delta(base, source + local_noise, epsilon))
            if uniform_count > 0:
                immigrants.append(sample_deltas_like(base, uniform_count, epsilon))
            next_positions.append(torch.cat(immigrants, dim=0))
        positions = torch.cat(next_positions, dim=0)[:n_particles]
    return {
        "delta": best_delta,
        "image": (base + best_delta).clamp(0.0, 1.0).detach() if best_delta is not None else base.detach().clone(),
        "metrics": best_metrics,
        "history": history,
        "diversity_history": diversity_history,
        "queries": int(budget.queries_used if budget is not None else n_particles * n_iterations),
        "elapsed_seconds": float(budget.elapsed() if budget is not None else 0.0),
    }


def _apso_memberships(f_value):
    f = float(max(0.0, min(1.0, f_value)))
    mu_s1 = 0.0 if f <= 0.4 else (5.0 * f - 2.0 if f <= 0.6 else (1.0 if f <= 0.7 else (-10.0 * f + 8.0 if f <= 0.8 else 0.0)))
    mu_s2 = 0.0 if f <= 0.2 else (10.0 * f - 2.0 if f <= 0.3 else (1.0 if f <= 0.4 else (-5.0 * f + 3.0 if f <= 0.6 else 0.0)))
    mu_s3 = 1.0 if f <= 0.1 else (-5.0 * f + 1.5 if f <= 0.3 else 0.0)
    mu_s4 = 0.0 if f <= 0.7 else (5.0 * f - 3.5 if f <= 0.9 else 1.0)
    return {
        "S1": float(max(0.0, min(1.0, mu_s1))),
        "S2": float(max(0.0, min(1.0, mu_s2))),
        "S3": float(max(0.0, min(1.0, mu_s3))),
        "S4": float(max(0.0, min(1.0, mu_s4))),
    }


def _apso_pick_state(f_value, prev_state):
    memberships = _apso_memberships(f_value)
    order = ["S1", "S2", "S3", "S4"]
    max_state = max(order, key=lambda state: memberships[state])
    if prev_state in memberships and memberships[prev_state] == memberships[max_state]:
        return prev_state
    return max_state


@torch.no_grad()
def apso_search_attack(
    oracle,
    base,
    clean_blackbox_pred,
    epsilon,
    n_particles,
    n_iterations,
    initial_positions=None,
    initial_velocities=None,
    rng_seed=None,
    hyperparams=None,
    budget: AttackBudget | None = None,
    progress_callback=None,
):
    set_global_seed(rng_seed or 0)
    params = {**APSO_PARAMS, **(hyperparams or {})}
    positions = initial_positions.clone() if initial_positions is not None else sample_deltas_like(base, n_particles, epsilon)
    velocity_scale = epsilon * float(params.get("initial_velocity_fraction", INITIAL_VELOCITY_FRACTION))
    velocities = initial_velocities.clone() if initial_velocities is not None else torch.empty_like(positions).uniform_(-velocity_scale, velocity_scale)
    pbest_positions = positions.detach().clone()
    pbest_scores = torch.full((n_particles,), -float("inf"), device=base.device)
    gbest_position = positions[:1].detach().clone()
    gbest_score = -float("inf")
    gbest_metrics = None
    history = []
    diversity_history = []
    c_min, c_max = float(params["c_min"]), float(params["c_max"])
    c_sum_max = float(params["c_sum_max"])
    delta_min, delta_max = float(params["delta_min"]), float(params["delta_max"])
    sigma_max, sigma_min = float(params["sigma_max"]), float(params["sigma_min"])
    vmax = (2.0 * epsilon) * float(params["max_velocity_fraction"])
    prev_state = None
    dims = int(positions[0].numel())
    delta_min_bound = torch.max(-base, torch.tensor(-epsilon, device=base.device)).flatten()
    delta_max_bound = torch.min(1.0 - base, torch.tensor(epsilon, device=base.device)).flatten()
    search_range = delta_max_bound - delta_min_bound
    for iteration in range(n_iterations):
        if budget is not None and budget.expired():
            break
        metrics = evaluate_blackbox_candidate_deltas(oracle, base, clean_blackbox_pred, positions, epsilon, budget=budget)
        if metrics is None:
            break
        diversity_history.append(float(population_diversity(positions, epsilon=epsilon).item()))
        scores = metrics["fitness"]
        improved = scores > pbest_scores
        pbest_scores[improved] = scores[improved]
        pbest_positions[improved] = positions[improved].detach().clone()
        idx = int(torch.argmax(pbest_scores).item())
        score = float(pbest_scores[idx].item())
        if score > gbest_score + 1e-12:
            gbest_score = score
            gbest_position = pbest_positions[idx:idx + 1].detach().clone()
            gbest_metrics = slice_metric_dict(metrics, idx)
        history.append(gbest_score)
        if progress_callback is not None and gbest_position is not None:
            progress_callback(best_delta=gbest_position, best_metrics=gbest_metrics, queries_used=int(budget.queries_used if budget is not None else (iteration + 1) * n_particles), elapsed_seconds=float(budget.elapsed() if budget is not None else 0.0))
        flat = positions.flatten(1)
        if n_particles > 1:
            dist_mat = torch.cdist(flat, flat)
            d_i = dist_mat.sum(dim=1) / float(n_particles - 1)
            d_min, d_max, d_g = float(d_i.min().item()), float(d_i.max().item()), float(d_i[idx].item())
            f_value = 0.0 if d_max - d_min <= 1e-12 else (d_g - d_min) / (d_max - d_min)
        else:
            f_value = 0.0
        state = _apso_pick_state(f_value, prev_state)
        prev_state = state
        inertia = 1.0 / (1.0 + 1.5 * math.exp(-2.6 * float(f_value)))
        delta = float(torch.empty((), device=base.device).uniform_(delta_min, delta_max).item())
        c1, c2 = 2.0, 2.0
        if state == "S1":
            c1 += delta
            c2 -= delta
        elif state == "S2":
            c1 += 0.5 * delta
            c2 -= 0.5 * delta
        elif state == "S3":
            c1 += 0.5 * delta
            c2 += 0.5 * delta
        else:
            c1 -= delta
            c2 += delta
        c1 = max(c_min, min(c_max, c1))
        c2 = max(c_min, min(c_max, c2))
        c_sum = c1 + c2
        if c_sum > c_sum_max:
            c1 = (c1 / c_sum) * c_sum_max
            c2 = (c2 / c_sum) * c_sum_max
        if state == "S3":
            progress = 0.0 if n_iterations <= 1 else iteration / float(n_iterations - 1)
            sigma = sigma_max + (sigma_min - sigma_max) * progress
            candidate = gbest_position.detach().clone().flatten(1)
            dim_index = int(torch.randint(0, dims, (1,), device=base.device).item())
            candidate[:, dim_index] = candidate[:, dim_index] + search_range[dim_index] * torch.randn((), device=base.device) * sigma
            candidate = project_delta(base, candidate.view_as(gbest_position), epsilon)
            if budget is None or budget.can_consume(1):
                candidate_metrics = evaluate_blackbox_candidate_deltas(oracle, base, clean_blackbox_pred, candidate, epsilon, budget=budget)
                if candidate_metrics is not None:
                    candidate_score = float(candidate_metrics["fitness"][0].item())
                    if candidate_score > gbest_score + 1e-12:
                        gbest_score = candidate_score
                        gbest_position = candidate.detach().clone()
                        gbest_metrics = slice_metric_dict(candidate_metrics, 0)
                    else:
                        worst_idx = int(torch.argmin(scores).item())
                        positions[worst_idx:worst_idx + 1] = candidate.detach()
                        velocities[worst_idx:worst_idx + 1] = torch.empty_like(velocities[worst_idx:worst_idx + 1]).uniform_(-velocity_scale, velocity_scale)
                        pbest_positions[worst_idx:worst_idx + 1] = candidate.detach()
                        pbest_scores[worst_idx] = candidate_metrics["fitness"][0]
        r1 = torch.rand_like(positions)
        r2 = torch.rand_like(positions)
        velocities = inertia * velocities + c1 * r1 * (pbest_positions - positions) + c2 * r2 * (gbest_position - positions)
        velocities = velocities.clamp(-vmax, vmax)
        positions = project_delta(base, positions + velocities, epsilon)
    return {
        "delta": gbest_position,
        "image": (base + gbest_position).clamp(0.0, 1.0).detach() if gbest_position is not None else base.detach().clone(),
        "metrics": gbest_metrics,
        "history": history,
        "diversity_history": diversity_history,
        "queries": int(budget.queries_used if budget is not None else n_particles * n_iterations),
        "elapsed_seconds": float(budget.elapsed() if budget is not None else 0.0),
    }


def _clpso_compute_pc(n_particles):
    if n_particles <= 1:
        return torch.full((n_particles,), 0.05)
    idx = torch.arange(n_particles, dtype=torch.float32)
    numerator = torch.exp(10.0 * idx / float(n_particles - 1)) - 1.0
    denominator = math.exp(10.0) - 1.0
    return 0.05 + 0.45 * (numerator / denominator)


@torch.no_grad()
def clpso_search_attack(
    oracle,
    base,
    clean_blackbox_pred,
    epsilon,
    n_particles,
    n_iterations,
    initial_positions=None,
    initial_velocities=None,
    rng_seed=None,
    hyperparams=None,
    budget: AttackBudget | None = None,
    progress_callback=None,
):
    set_global_seed(rng_seed or 0)
    params = {**CLPSO_PARAMS, **(hyperparams or {})}
    positions = initial_positions.clone() if initial_positions is not None else sample_deltas_like(base, n_particles, epsilon)
    velocity_scale = epsilon * float(params.get("initial_velocity_fraction", INITIAL_VELOCITY_FRACTION))
    velocities = initial_velocities.clone() if initial_velocities is not None else torch.empty_like(positions).uniform_(-velocity_scale, velocity_scale)
    pbest_positions = positions.detach().clone()
    pbest_scores = torch.full((n_particles,), -float("inf"), device=base.device)
    gbest_position = positions[:1].detach().clone()
    gbest_score = -float("inf")
    gbest_metrics = None
    history = []
    diversity_history = []
    w0, w1, c = float(params["w0"]), float(params["w1"]), float(params["c"])
    refreshing_gap = int(params["refreshing_gap"])
    vmax = (2.0 * epsilon) * float(params["max_velocity_fraction"])
    dims = int(positions[0].numel())
    dim_idx = torch.arange(dims, device=base.device)
    pc = _clpso_compute_pc(n_particles).to(base.device)
    exemplar_idx = torch.zeros((n_particles, dims), dtype=torch.long, device=base.device)
    no_improve_flags = torch.zeros((n_particles,), dtype=torch.long, device=base.device)

    def pick_exemplar_mapping(i):
        if n_particles <= 1:
            return torch.full((dims,), int(i), dtype=torch.long, device=base.device)
        rand_probs = torch.rand((dims,), device=base.device)
        learn_from_others = rand_probs <= pc[i]
        mapping = torch.full((dims,), int(i), dtype=torch.long, device=base.device)
        if learn_from_others.any():
            n_others = dims if int(learn_from_others.sum().item()) == dims else int(learn_from_others.sum().item())
            candidates = torch.cat([torch.arange(0, i, device=base.device), torch.arange(i + 1, n_particles, device=base.device)])
            idx_a = candidates[torch.randint(0, candidates.numel(), (n_others,), device=base.device)]
            idx_b = candidates[torch.randint(0, candidates.numel(), (n_others,), device=base.device)]
            winner_mask = pbest_scores[idx_a] >= pbest_scores[idx_b]
            winners = torch.where(winner_mask, idx_a, idx_b)
            mapping[learn_from_others] = winners
        if (~learn_from_others).all():
            forced_dim = torch.randint(0, dims, (1,), device=base.device).item()
            candidates = torch.cat([torch.arange(0, i, device=base.device), torch.arange(i + 1, n_particles, device=base.device)])
            a = candidates[torch.randint(0, candidates.numel(), (1,), device=base.device).item()]
            b = candidates[torch.randint(0, candidates.numel(), (1,), device=base.device).item()]
            mapping[forced_dim] = a if pbest_scores[a] >= pbest_scores[b] else b
        return mapping

    for i in range(n_particles):
        exemplar_idx[i] = pick_exemplar_mapping(i)

    for iteration in range(n_iterations):
        if budget is not None and budget.expired():
            break
        metrics = evaluate_blackbox_candidate_deltas(oracle, base, clean_blackbox_pred, positions, epsilon, budget=budget)
        if metrics is None:
            break
        diversity_history.append(float(population_diversity(positions, epsilon=epsilon).item()))
        scores = metrics["fitness"]
        improved = scores > (pbest_scores + 1e-12)
        pbest_scores[improved] = scores[improved]
        pbest_positions[improved] = positions[improved].detach().clone()
        no_improve_flags[improved] = 0
        no_improve_flags[~improved] += 1
        idx = int(torch.argmax(pbest_scores).item())
        score = float(pbest_scores[idx].item())
        if score > gbest_score + 1e-12:
            gbest_score = score
            gbest_position = pbest_positions[idx:idx + 1].detach().clone()
            gbest_metrics = slice_metric_dict(metrics, idx)
        history.append(gbest_score)
        if progress_callback is not None and gbest_position is not None:
            progress_callback(best_delta=gbest_position, best_metrics=gbest_metrics, queries_used=int(budget.queries_used if budget is not None else (iteration + 1) * n_particles), elapsed_seconds=float(budget.elapsed() if budget is not None else 0.0))
        for i in range(n_particles):
            if refreshing_gap >= 0 and int(no_improve_flags[i].item()) >= refreshing_gap:
                exemplar_idx[i] = pick_exemplar_mapping(i)
                no_improve_flags[i] = 0
        progress = (iteration + 1) / float(max(1, n_iterations))
        inertia = w0 - (w0 - w1) * progress
        flat_pos = positions.flatten(1)
        flat_vel = velocities.flatten(1)
        flat_pbest = pbest_positions.flatten(1)
        for i in range(n_particles):
            targets = flat_pbest[exemplar_idx[i], dim_idx]
            rand_vec = torch.rand((dims,), device=base.device)
            flat_vel[i] = inertia * flat_vel[i] + c * rand_vec * (targets - flat_pos[i])
        flat_vel = flat_vel.clamp(-vmax, vmax)
        velocities = flat_vel.view_as(velocities)
        positions = project_delta(base, positions + velocities, epsilon)
    return {
        "delta": gbest_position,
        "image": (base + gbest_position).clamp(0.0, 1.0).detach() if gbest_position is not None else base.detach().clone(),
        "metrics": gbest_metrics,
        "history": history,
        "diversity_history": diversity_history,
        "queries": int(budget.queries_used if budget is not None else n_particles * n_iterations),
        "elapsed_seconds": float(budget.elapsed() if budget is not None else 0.0),
    }


def get_attack_function(method: str):
    methods = {
        "random": random_search_attack,
        "pso": pso_search_attack,
        "pso_jitter": pso_jitter_search_attack,
        "genetic": genetic_search_attack,
        "apso": apso_search_attack,
        "clpso": clpso_search_attack,
    }
    return methods[method.lower()]


@torch.no_grad()
def choose_attack_candidates(oracle, loader, limit, device):
    examples = []
    for images, labels, source_indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        q = oracle.query(images, for_evaluation=True, count=False)
        mask = (q["pred"] == labels.cpu()) & (q["final_pred"] == labels.cpu()) & (q["exit"] < len(oracle.exit_costs))
        idxs = mask.nonzero(as_tuple=False).flatten()
        for idx in idxs:
            examples.append({
                "source_index": int(source_indices[idx].item()),
                "image": images[idx:idx + 1].detach().cpu(),
                "label": int(labels[idx].item()),
                "clean_exit": int(q["exit"][idx].item()),
                "clean_blackbox_pred": int(q["pred"][idx].item()),
                "clean_blackbox_conf": float(q["confidence"][idx].item()),
                "clean_observed_cost": float(q["observed_cost"][idx].item()),
                "clean_matmul_proxy": float(q["observed_matmul_proxy"][idx].item()),
                "clean_final_pred": int(q["final_pred"][idx].item()),
                "clean_final_conf": float(q["final_confidence"][idx].item()),
            })
            if len(examples) >= limit:
                return examples
    return examples


def fixed_method_params(method):
    method = method.lower()
    fixed = {
        "random": {},
        "pso": {**PSO_PARAMS},
        "pso_jitter": {**PSO_JITTER_PARAMS},
        "genetic": {**GENETIC_PARAMS},
        "apso": {**APSO_PARAMS},
        "clpso": {**CLPSO_PARAMS},
        "universal_ga_weighted": {**GENETIC_PARAMS},
        "universal_pso_jitter_weighted": {**PSO_JITTER_PARAMS},
        "universal_pso_jitter_multiswarm": {**PSO_JITTER_PARAMS},
        "universal_apso_weighted": {**PSO_JITTER_PARAMS},
        "universal_apso_multiswarm": {**PSO_JITTER_PARAMS},
    }
    return fixed[method]


def save_split_csv(examples, path, split_name):
    frame = pd.DataFrame([{
        "split": split_name,
        "row_index": idx,
        "source_index": example["source_index"],
        "label": example["label"],
        "clean_exit": example["clean_exit"],
        "clean_blackbox_pred": example["clean_blackbox_pred"],
        "clean_blackbox_conf": example["clean_blackbox_conf"],
        "clean_observed_cost": example["clean_observed_cost"],
        "clean_matmul_proxy": example["clean_matmul_proxy"],
        "clean_final_pred": example["clean_final_pred"],
        "clean_final_conf": example["clean_final_conf"],
    } for idx, example in enumerate(examples)])
    frame.to_csv(path, index=False)
    return frame


def save_images_tensor(examples, path):
    tensor = torch.cat([example["image"] for example in examples], dim=0) if examples else torch.empty(0)
    torch.save(tensor, path)
    return tensor


def sample_examples(examples, limit, seed):
    if limit is None or limit >= len(examples):
        return list(examples)
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(examples), size=int(limit), replace=False))
    return [examples[int(index)] for index in indices]


@torch.no_grad()
def evaluate_baseline_examples(model, oracle, examples, thresholds, exit_costs, device):
    rows = []
    images = []
    zero_deltas = []
    for example_index, example in enumerate(examples):
        base = example["image"].to(device)
        label = int(example["label"])
        q = oracle.query(base, for_evaluation=True, count=False)
        eval_m = evaluate_adversarial_example_for_evaluator(
            model=model,
            base=base,
            delta=torch.zeros_like(base),
            label=label,
            thresholds=thresholds,
            exit_costs=exit_costs,
        )
        rows.append({
            "input_id": example_index,
            "source_index": int(example["source_index"]),
            "label": label,
            "original_exit": int(eval_m["exit"][0].item()),
            "original_accuracy": bool(int(eval_m["final_pred"][0].item()) == label),
            "original_proxy": float(q["observed_matmul_proxy"][0].item()),
            "original_observed_cost": float(q["observed_cost"][0].item()),
            "original_blackbox_pred": int(q["pred"][0].item()),
            "original_blackbox_conf": float(q["confidence"][0].item()),
            "original_final_pred": int(q["final_pred"][0].item()),
            "original_final_conf": float(q["final_confidence"][0].item()),
        })
        images.append(base.detach().cpu())
        zero_deltas.append(torch.zeros_like(base).detach().cpu())
    baseline_frame = pd.DataFrame(rows)
    image_tensor = torch.cat(images, dim=0) if images else torch.empty(0)
    delta_tensor = torch.cat(zero_deltas, dim=0) if zero_deltas else torch.empty(0)
    return baseline_frame, image_tensor, delta_tensor


def build_attack_result_matrices(results_rows, repeats=ATTACK_REPEATS):
    result_frame = pd.DataFrame(results_rows)
    metric_frames = {}
    for metric_name in ["exit", "accuracy", "proxy"]:
        rows = []
        for input_id in sorted(result_frame["input_id"].unique()):
            subset = result_frame[result_frame["input_id"] == input_id].sort_values("attack_index")
            row = {"input_id": input_id, "label": int(subset["label"].iloc[0])}
            for attack_index in range(1, repeats + 1):
                row[f"attack_{attack_index}"] = subset[metric_name].iloc[attack_index - 1]
            rows.append(row)
        metric_frames[metric_name] = pd.DataFrame(rows)
    return metric_frames


def summarize_attack_rows(result_frame, repeats=ATTACK_REPEATS):
    summary_rows = []
    for method in sorted(result_frame["method"].unique()):
        method_frame = result_frame[result_frame["method"] == method]
        summary_rows.append({
            "method": method,
            "inputs": method_frame["input_id"].nunique(),
            "attack_repeats": repeats,
            "mean_exit": method_frame["exit"].mean(),
            "mean_accuracy": method_frame["accuracy"].mean(),
            "mean_proxy": method_frame["proxy"].mean(),
            "std_exit": method_frame["exit"].std(ddof=1),
            "std_accuracy": method_frame["accuracy"].astype(float).std(ddof=1),
            "std_proxy": method_frame["proxy"].std(ddof=1),
        })
    return pd.DataFrame(summary_rows)


def calibrate_wall_clock_budget(reference_method, tuning_examples, model, oracle, thresholds, exit_costs, device, query_budget=QUERY_BUDGET_PER_ATTACK, repeats=3):
    attack_fn = get_attack_function(reference_method)
    measured_times = []
    for repeat_index in range(min(repeats, len(tuning_examples))):
        example = tuning_examples[repeat_index]
        base = example["image"].to(device)
        clean_pred = int(example["clean_blackbox_pred"])
        initial_positions, initial_velocities = make_initial_swarm_state(base, EPSILON, ATTACK_PARTICLES, TUNING_SEED + repeat_index * 100)
        budget = AttackBudget(mode="query", query_budget=query_budget).start()
        start = time.perf_counter()
        attack_fn(
            oracle=oracle,
            base=base,
            clean_blackbox_pred=clean_pred,
            epsilon=EPSILON,
            n_particles=ATTACK_PARTICLES,
            n_iterations=ATTACK_ITERATIONS,
            initial_positions=initial_positions,
            initial_velocities=initial_velocities,
            rng_seed=TUNING_SEED + repeat_index * 1000,
            hyperparams=fixed_method_params(reference_method),
            budget=budget,
        )
        measured_times.append(time.perf_counter() - start)
    return float(np.median(measured_times) * 1.05 if measured_times else 30.0)
