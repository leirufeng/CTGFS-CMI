from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression

from codes.data import (
    encode_kmer_features,
    encode_sequences,
    load_fold,
    load_pooled_word2vec,
    load_root_fold,
)
from codes.model import CIRC_PU, NGCFLite, info_nce, non_negative_pu_risk


DATASETS = ["CMI-9589", "CMI-9905", "CMI-20208"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_pair_dataset(
    positive_pairs: np.ndarray,
    negative_pairs: np.ndarray,
    structure_features: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = np.concatenate([positive_pairs, negative_pairs], axis=0)
    labels = np.concatenate(
        [
            np.ones(positive_pairs.shape[0], dtype=np.float32),
            np.zeros(negative_pairs.shape[0], dtype=np.float32),
        ]
    )
    mir = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device)
    circ = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    structure = torch.as_tensor(
        structure_features,
        dtype=torch.float32,
        device=device,
    )
    return mir, circ, target, structure


def evaluate(
    model: CIRC_PU,
    mir_index: torch.Tensor,
    circ_index: torch.Tensor,
    target: torch.Tensor,
    structure_features: torch.Tensor | None,
    norm_adj: torch.Tensor,
    batch_size: int = 4096,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    score_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, mir_index.numel(), batch_size):
            end = start + batch_size
            scores = model(
                mir_index[start:end],
                circ_index[start:end],
                norm_adj,
                structure_features=(
                    structure_features[start:end]
                    if structure_features is not None
                    else None
                ),
                return_alignment=False,
            )
            score_chunks.append(torch.sigmoid(scores).cpu().numpy())
    probabilities = np.concatenate(score_chunks)
    labels = target.cpu().numpy()
    return compute_metrics(labels, probabilities), labels, probabilities


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    labels_int = labels.astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels_int, predictions, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "auc": float(roc_auc_score(labels_int, probabilities)),
        "aupr": float(average_precision_score(labels_int, probabilities)),
        "accuracy": float(accuracy_score(labels_int, predictions)),
        "precision": float(precision_score(labels_int, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels_int, predictions, zero_division=0)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(labels_int, predictions)),
    }


def best_mcc_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    thresholds = np.unique(np.r_[0.5, np.quantile(probabilities, np.linspace(0.01, 0.99, 199))])
    best_threshold = 0.5
    best_mcc = float(matthews_corrcoef(labels.astype(np.int64), probabilities >= 0.5))
    for threshold in thresholds:
        mcc = matthews_corrcoef(
            labels.astype(np.int64),
            (probabilities >= threshold).astype(np.int64),
        )
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_threshold = float(threshold)
    return best_threshold


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels.astype(np.int64), predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def evaluate_predictions(
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
) -> dict:
    """Select a threshold on final validation probabilities, never test labels."""
    threshold = best_mcc_threshold(validation_labels, validation_probabilities)
    return {
        "validation": compute_metrics(validation_labels, validation_probabilities),
        "validation_at_threshold": metrics_at_threshold(validation_labels, validation_probabilities, threshold),
        "test": compute_metrics(test_labels, test_probabilities),
        "test_at_threshold": metrics_at_threshold(test_labels, test_probabilities, threshold),
    }


def hard_negative_weights(
    scores: torch.Tensor,
    target: torch.Tensor,
    hard_fraction: float = 0.25,
    hard_weight: float = 2.0,
) -> torch.Tensor:
    weights = torch.ones_like(scores)
    negative_positions = torch.nonzero(target < 0.5, as_tuple=False).flatten()
    if negative_positions.numel() == 0:
        return weights
    count = max(1, int(negative_positions.numel() * hard_fraction))
    negative_scores = scores[negative_positions].detach()
    hardest = torch.topk(negative_scores, k=count, largest=True).indices
    weights[negative_positions[hardest]] = hard_weight
    return weights


def train_one_fold(
    dataset: str,
    fold: int,
    root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    alignment_weight: float,
    pu_weight: float,
    patience: int,
    seed: int,
    d_model: int,
    sequence_layers: int,
    graph_layers: int,
    sequence_encoder: str,
    model_name: str,
    split_name: str,
    validation_fraction: float,
    use_structure_features: bool,
    ensemble_size: int,
    add_word2vec: bool,
    stack_validation: bool,
    eval_batch_size: int,
    circ_segment_length: int,
    circ_segments: int,
    artifact_dir: Path | None = None,
) -> dict:
    set_seed(seed + fold)
    if split_name == "root":
        fold_data = load_root_fold(
            dataset,
            fold,
            root,
            device,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    else:
        fold_data = load_fold(dataset, fold, root, device)
    structure_dim = (
        fold_data.train_structure.shape[1] if use_structure_features else 0
    )
    if model_name == "ngcf":
        mir_tokens = mir_mask = circ_tokens = circ_mask = None
        mir_features = circ_features = None
    elif sequence_encoder == "transformer":
        mir_tokens, mir_mask, circ_tokens, circ_mask = encode_sequences(
            fold_data.mir_sequences,
            fold_data.circ_sequences,
            device,
            circ_segment_length=circ_segment_length,
            circ_segment_count=circ_segments,
        )
        mir_features = circ_features = None
    elif sequence_encoder == "kmer":
        mir_features, circ_features = encode_kmer_features(
            fold_data.mir_sequences,
            fold_data.circ_sequences,
            device,
        )
        if add_word2vec:
            w2v_mir, w2v_circ = load_pooled_word2vec(
                dataset,
                root,
                device,
            )
            mir_features = torch.cat([mir_features, w2v_mir], dim=-1)
            circ_features = torch.cat([circ_features, w2v_circ], dim=-1)
        mir_tokens = mir_mask = circ_tokens = circ_mask = None
    else:
        mir_tokens, mir_mask, circ_tokens, circ_mask = encode_sequences(
            fold_data.mir_sequences,
            fold_data.circ_sequences,
            device,
            circ_segment_length=circ_segment_length,
            circ_segment_count=circ_segments,
        )
        mir_features, circ_features = encode_kmer_features(
            fold_data.mir_sequences,
            fold_data.circ_sequences,
            device,
        )
        if add_word2vec:
            w2v_mir, w2v_circ = load_pooled_word2vec(
                dataset,
                root,
                device,
            )
            mir_features = torch.cat([mir_features, w2v_mir], dim=-1)
            circ_features = torch.cat([circ_features, w2v_circ], dim=-1)
    train_mir, train_circ, train_target, train_structure = make_pair_dataset(
        fold_data.train_positive,
        fold_data.train_negative,
        fold_data.train_structure,
        device,
    )
    (
        validation_mir,
        validation_circ,
        validation_target,
        validation_structure,
    ) = make_pair_dataset(
        fold_data.validation_positive,
        fold_data.validation_negative,
        fold_data.validation_structure,
        device,
    )
    test_mir, test_circ, test_target, test_structure = make_pair_dataset(
        fold_data.test_positive,
        fold_data.test_negative,
        fold_data.test_structure,
        device,
    )

    if model_name == "ngcf":
        model = NGCFLite(
            n_mir=fold_data.n_mir,
            n_circ=fold_data.n_circ,
            n_nodes=fold_data.n_mir + fold_data.n_circ,
            d_model=d_model,
            graph_layers=graph_layers,
            dropout=0.1,
            structure_dim=structure_dim,
        ).to(device)
    else:
        model = CIRC_PU(
            mir_tokens=mir_tokens,
            mir_mask=mir_mask,
            circ_tokens=circ_tokens,
            circ_mask=circ_mask,
            n_nodes=fold_data.n_mir + fold_data.n_circ,
            mir_features=mir_features,
            circ_features=circ_features,
            sequence_encoder=sequence_encoder,
            circ_segments=circ_segments,
            d_model=d_model,
            sequence_layers=sequence_layers,
            graph_layers=graph_layers,
            dropout=0.1,
            structure_dim=structure_dim,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=learning_rate * 0.05,
    )

    best_score = -float("inf")
    best_epoch = -1
    best_validation: dict[str, float] = {}
    stale_epochs = 0
    logs: list[dict[str, float]] = []
    top_states: list[tuple[float, int, dict, np.ndarray]] = []

    n_train = train_target.numel()
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_train, device=device)
        losses: list[float] = []
        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            if index.numel() < 2:
                continue
            scores, mir_alignment, circ_alignment = model(
                train_mir[index],
                train_circ[index],
                fold_data.norm_adj,
                structure_features=(
                    train_structure[index] if use_structure_features else None
                ),
                return_alignment=True,
            )
            target = train_target[index]
            weights = hard_negative_weights(scores, target)
            supervised = F.binary_cross_entropy_with_logits(
                scores,
                target,
                weight=weights,
            )
            if mir_alignment is None or circ_alignment is None:
                alignment = scores.new_zeros(())
            else:
                alignment = 0.5 * (
                    info_nce(mir_alignment) + info_nce(circ_alignment)
                )
            positive_scores = scores[target > 0.5]
            unlabeled_scores = scores[target < 0.5]
            pu_risk = non_negative_pu_risk(
                positive_scores,
                unlabeled_scores,
                fold_data.positive_prior,
            )
            loss = (
                supervised
                + alignment_weight * alignment
                + pu_weight * pu_risk
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        validation_metrics, validation_labels, validation_probabilities = evaluate(
            model,
            validation_mir,
            validation_circ,
            validation_target,
            validation_structure if use_structure_features else None,
            fold_data.norm_adj,
            batch_size=eval_batch_size,
        )
        validation_score = 0.5 * (
            validation_metrics["auc"] + validation_metrics["aupr"]
        )
        logs.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
        )
        print(
            f"[{dataset} fold {fold}] epoch {epoch + 1:02d}/{epochs} "
            f"loss={logs[-1]['loss']:.4f} "
            f"val_auc={validation_metrics['auc']:.4f} "
            f"val_aupr={validation_metrics['aupr']:.4f}"
        )

        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_validation = validation_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        top_states.append(
            (
                validation_score,
                epoch,
                copy.deepcopy(model.state_dict()),
                validation_probabilities,
            )
        )
        top_states.sort(key=lambda item: item[0], reverse=True)
        top_states = top_states[: max(1, ensemble_size)]
        if stale_epochs >= patience:
            print(f"[{dataset} fold {fold}] early stop at epoch {epoch + 1}")
            break

    ensemble_validation_probabilities = np.mean(
        [item[3] for item in top_states],
        axis=0,
    )
    test_probability_snapshots: list[np.ndarray] = []
    test_labels: np.ndarray | None = None
    for _, _, state, _ in top_states:
        model.load_state_dict(state)
        _, current_test_labels, current_test_probabilities = evaluate(
            model,
            test_mir,
            test_circ,
            test_target,
            test_structure if use_structure_features else None,
            fold_data.norm_adj,
            batch_size=eval_batch_size,
        )
        test_labels = current_test_labels
        test_probability_snapshots.append(current_test_probabilities)
    test_probabilities = np.mean(test_probability_snapshots, axis=0)
    test_metrics = compute_metrics(
        test_labels,
        test_probabilities,
    )
    base_validation_probabilities = ensemble_validation_probabilities.copy()
    base_test_probabilities = test_probabilities.copy()
    base_metrics = evaluate_predictions(validation_labels, base_validation_probabilities,
                                        test_labels, base_test_probabilities)
    if stack_validation:
        validation_meta = np.column_stack(
            [
                np.log(
                    np.clip(ensemble_validation_probabilities, 1e-6, 1 - 1e-6)
                    / np.clip(
                        1.0 - ensemble_validation_probabilities,
                        1e-6,
                        1 - 1e-6,
                    )
                ),
                validation_structure.cpu().numpy(),
            ]
        )
        test_meta = np.column_stack(
            [
                np.log(
                    np.clip(test_probabilities, 1e-6, 1 - 1e-6)
                    / np.clip(1.0 - test_probabilities, 1e-6, 1 - 1e-6)
                ),
                test_structure.cpu().numpy(),
            ]
        )
        stacker = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed + fold,
        )
        stacker.fit(validation_meta, validation_labels)
        ensemble_validation_probabilities = stacker.predict_proba(
            validation_meta
        )[:, 1]
        test_probabilities = stacker.predict_proba(test_meta)[:, 1]
        validation_metrics = compute_metrics(
            validation_labels,
            ensemble_validation_probabilities,
        )
        test_metrics = compute_metrics(test_labels, test_probabilities)
    final_metrics = evaluate_predictions(
        validation_labels, ensemble_validation_probabilities, test_labels, test_probabilities,
    )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_dir / "predictions.npz",
            validation_pairs=np.column_stack([validation_mir.cpu().numpy(), validation_circ.cpu().numpy()]),
            test_pairs=np.column_stack([test_mir.cpu().numpy(), test_circ.cpu().numpy()]),
            validation_labels=validation_labels, test_labels=test_labels,
            base_validation_probabilities=base_validation_probabilities,
            base_test_probabilities=base_test_probabilities,
            validation_probabilities=ensemble_validation_probabilities, test_probabilities=test_probabilities,
            validation_structure=validation_structure.cpu().numpy(), test_structure=test_structure.cpu().numpy(),
        )
        if stack_validation:
            parameters = {
                "feature_order": ["base_logit", "log1p_path3", "log1p_mir_degree", "log1p_circ_degree", "log1p_degree_product", "normalized_path3"],
                "coef": stacker.coef_.tolist(), "intercept": stacker.intercept_.tolist(),
                "classes": stacker.classes_.tolist(), "fit_split": "validation",
            }
            (artifact_dir / "stacker.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")

    return {
        "dataset": dataset,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "ensemble_size": len(top_states),
        "stack_validation": stack_validation,
        "validation_role": "model_selection_and_stacker_fit" if stack_validation else "model_selection",
        "snapshot_epochs": [item[1] for item in top_states],
        **final_metrics,
        **{f"base_{key}": value for key, value in base_metrics.items()},
        "logs": logs,
    }


def summarize(results: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    metric_groups = ["validation", "test", "test_at_threshold", "base_test"]
    for dataset in DATASETS:
        dataset_results = [row for row in results if row["dataset"] == dataset]
        if not dataset_results:
            continue
        summary[dataset] = {}
        for group in metric_groups:
            if group not in dataset_results[0]:
                continue
            keys = dataset_results[0][group].keys()
            for key in keys:
                values = np.asarray(
                    [row[group][key] for row in dataset_results],
                    dtype=np.float64,
                )
                summary[dataset][f"{group}_{key}_mean"] = float(values.mean())
                summary[dataset][f"{group}_{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return summary


def write_outputs(output_dir: Path, results: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fold_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / "fold_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "fold",
                "best_epoch",
                "auc",
                "aupr",
                "accuracy",
                "precision",
                "sensitivity",
                "specificity",
                "mcc",
                "threshold",
                "threshold_accuracy",
                "threshold_precision",
                "threshold_sensitivity",
                "threshold_specificity",
                "threshold_mcc",
            ]
        )
        for row in results:
            test = row["test"]
            threshold = row["test_at_threshold"]
            writer.writerow(
                [
                    row["dataset"],
                    row["fold"],
                    row["best_epoch"],
                    test["auc"],
                    test["aupr"],
                    test["accuracy"],
                    test["precision"],
                    test["sensitivity"],
                    test["specificity"],
                    test["mcc"],
                    threshold["threshold"],
                    threshold["accuracy"],
                    threshold["precision"],
                    threshold["sensitivity"],
                    threshold["specificity"],
                    threshold["mcc"],
                ]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIRC-PU model.")
    parser.add_argument("--preset", choices=["baseline", "stacked", "transformer"], default="baseline",
                        help="stacked uses k-mer + stacking; transformer replaces k-mer with the full token encoder")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alignment-weight", type=float, default=0.02)
    parser.add_argument("--pu-weight", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--sequence-layers", type=int, default=1)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument(
        "--sequence-encoder",
        choices=["transformer", "kmer", "hybrid"],
        default="kmer",
    )
    parser.add_argument(
        "--model",
        choices=["circpu", "ngcf"],
        default="circpu",
    )
    parser.add_argument(
        "--split",
        choices=["fivefold", "root"],
        default="fivefold",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--use-structure-features", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--add-word2vec", action="store_true")
    parser.add_argument("--stack-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--circ-segment-length", type=int, default=64)
    parser.add_argument("--circ-segments", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result") / "CIRC-PU",
    )
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.preset in ("stacked", "transformer"):
        parser.set_defaults(epochs=30, batch_size=1024, patience=8, d_model=192,
                            stack_validation=True, output_dir=Path("result") / "CIRC-PU-stacked-run")
    if preliminary.preset == "transformer":
        parser.set_defaults(sequence_encoder="transformer",
                            output_dir=Path("result") / "CIRC-PU-transformer-run")
    args = parser.parse_args(argv)
    for name in ("folds", "epochs", "batch_size", "eval_batch_size", "patience", "d_model",
                 "sequence_layers", "ensemble_size", "circ_segment_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 2 or args.d_model < 2:
        parser.error("batch-size and d-model must be at least 2")
    if args.folds > 5 or args.graph_layers < 0 or not 1 <= args.circ_segments <= 4:
        parser.error("folds must be <= 5, graph-layers >= 0, and circ-segments between 1 and 4")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between 0 and 1")
    if args.lr <= 0 or min(args.weight_decay, args.alignment_weight, args.pu_weight) < 0:
        parser.error("lr must be positive and regularization weights nonnegative")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "fold_results.json").exists():
        raise FileExistsError(f"Results already exist in {args.output_dir}; choose a new --output-dir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "command": sys.argv, "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")},
        "validation_role": "model selection and stacker fitting" if args.stack_validation else "model selection",
        "test_role": "held-out pairs; not an external dataset or node cold-start test",
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Preset: {args.preset}; encoder: {args.sequence_encoder}; stacking: {args.stack_validation}")

    results: list[dict] = []
    for dataset in args.datasets:
        for fold in range(args.folds):
            result = train_one_fold(
                dataset=dataset,
                fold=fold,
                root=Path(".").resolve(),
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                alignment_weight=args.alignment_weight,
                pu_weight=args.pu_weight,
                patience=args.patience,
                seed=args.seed,
                d_model=args.d_model,
                sequence_layers=args.sequence_layers,
                graph_layers=args.graph_layers,
                sequence_encoder=args.sequence_encoder,
                model_name=args.model,
                split_name=args.split,
                validation_fraction=args.validation_fraction,
                use_structure_features=args.use_structure_features,
                ensemble_size=args.ensemble_size,
                add_word2vec=args.add_word2vec,
                stack_validation=args.stack_validation,
                eval_batch_size=args.eval_batch_size,
                circ_segment_length=args.circ_segment_length,
                circ_segments=args.circ_segments,
                artifact_dir=args.output_dir / dataset / f"fold{fold}",
            )
            results.append(result)
            partial_summary = summarize(results)
            write_outputs(args.output_dir, results, partial_summary)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize(results)
    write_outputs(args.output_dir, results, summary)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_folds"] = len(results)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
