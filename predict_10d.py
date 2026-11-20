"""Load the exported five snapshots and the fitted 10D logistic stacker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch

from circpu.data import encode_sequences, load_fold
from circpu.model import CIRC_PU
from circpu.run import compute_metrics, evaluate, make_pair_dataset, set_seed

ROOT = Path(__file__).resolve().parent
DATASETS = ("CMI-9589", "CMI-9905", "CMI-20208")
FEATURES = [f"snapshot_rank{i}_logit" for i in range(1, 6)] + [
    "log1p_path3", "log1p_mir_degree", "log1p_circ_degree",
    "log1p_degree_product", "normalized_path3",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stacking_probability(snapshot_probabilities, structure, parameters):
    """p: [5, N]; structure: [N, 5]; output: [N], without averaging p."""
    p, z = np.asarray(snapshot_probabilities), np.asarray(structure)
    if p.ndim != 2 or p.shape[0] != 5 or z.shape != (p.shape[1], 5):
        raise ValueError("Expected snapshot probabilities [5,N] and structure [N,5]")
    if parameters["feature_order"] != FEATURES:
        raise ValueError("Incorrect LR feature order")
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                    np.clip(1.0 - p, 1e-6, 1 - 1e-6))
    features = np.column_stack([logits.T, z])
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite features")
    return expit(features @ np.asarray(parameters["coef"])[0] + parameters["intercept"][0])


def verify_bundle():
    manifest = read_json(ROOT / "bundle_manifest.json")
    for filename, item in manifest["files"].items():
        path = ROOT / filename
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"File integrity check failed: {filename}")
    checked = []
    for dataset in DATASETS:
        for fold in range(1, 6):
            folder = ROOT / "models" / dataset / f"fold{fold}"
            params = read_json(folder / "stacker_10d.json")
            expected = read_json(folder / "metadata.json")["test_metrics"]
            with np.load(folder / "predictions.npz") as data:
                for part in ("validation", "test"):
                    calculated = stacking_probability(
                        data[f"snapshot_{part}_probabilities"], data[f"{part}_structure"], params)
                    np.testing.assert_allclose(calculated, data[f"stacking_10d_{part}_probabilities"],
                                               atol=1e-12, rtol=0)
                metrics = compute_metrics(data["test_labels"], calculated)
            for metric, value in metrics.items():
                if abs(value - expected[metric]) > 1e-12:
                    raise RuntimeError(f"Metric mismatch: {dataset} fold{fold} {metric}")
            checked.append({"dataset": dataset, "fold": fold, "test": metrics})
    save_json(ROOT / "verification_export.json", {
        "status": "PASS", "checked_files": len(manifest["files"]),
        "snapshot_pt_files": manifest["snapshot_pt_files"],
        "stackers": len(checked), "all_saved_test_metrics_reproduced": True,
    })
    print(f"PASS: {len(manifest['files'])} files, 75 snapshot .pt files, 15 fitted 10D stackers")


def evaluate_saved(dataset, fold, device):
    """Recompute test predictions from actual .pt weights; no fitting occurs."""
    folder = ROOT / "models" / dataset / f"fold{fold}"
    metadata = read_json(folder / "metadata.json")
    config = metadata["settings"]
    set_seed(config["seed"] + fold - 1)
    data = load_fold(dataset, fold - 1, ROOT, device)
    mir_tokens, mir_mask, circ_tokens, circ_mask = encode_sequences(
        data.mir_sequences, data.circ_sequences, device,
        circ_segment_length=config["circ_segment_length"], circ_segment_count=config["circ_segments"])
    model = CIRC_PU(
        mir_tokens=mir_tokens, mir_mask=mir_mask, circ_tokens=circ_tokens, circ_mask=circ_mask,
        n_nodes=data.n_mir + data.n_circ, sequence_encoder=config["sequence_encoder"],
        structure_dim=0, circ_segments=config["circ_segments"], d_model=config["d_model"],
        sequence_layers=config["sequence_layers"], graph_layers=config["graph_layers"], dropout=0.1,
    ).to(device)
    mir, circ, labels, _ = make_pair_dataset(data.test_positive, data.test_negative, data.test_structure, device)
    snapshots = []
    for rank in range(1, 6):
        checkpoint = torch.load(folder / f"snapshot_rank{rank}.pt", map_location="cpu", weights_only=True)
        if checkpoint["dataset"] != dataset or checkpoint["fold"] != fold - 1 or checkpoint["rank"] != rank:
            raise RuntimeError("Checkpoint dataset/fold/rank mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        _, y, p = evaluate(model, mir, circ, labels, None, data.norm_adj, batch_size=config["eval_batch_size"])
        snapshots.append(p)
    snapshots = np.stack(snapshots)
    probability = stacking_probability(snapshots, data.test_structure, read_json(folder / "stacker_10d.json"))
    measured = compute_metrics(y, probability)
    with np.load(folder / "predictions.npz") as reference:
        np.testing.assert_array_equal(y, reference["test_labels"])
        np.testing.assert_array_equal(np.column_stack([mir.cpu().numpy(), circ.cpu().numpy()]), reference["test_pairs"])
        max_difference = float(np.max(np.abs(probability - reference["stacking_10d_test_probabilities"])))
    output = ROOT / "reevaluation" / dataset / f"fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", test_labels=y, test_probabilities=probability,
                        snapshot_test_probabilities=snapshots)
    result = {"dataset": dataset, "fold": fold, "device": str(device), "threshold": 0.5,
              "test": measured, "maximum_probability_difference_from_saved": max_difference,
              "metric_differences_from_saved": {k: measured[k] - metadata["test_metrics"][k] for k in measured}}
    save_json(output / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "evaluate"], nargs="?", default="verify")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == "verify":
        verify_bundle()
    else:
        for dataset in args.datasets:
            for fold in args.folds:
                evaluate_saved(dataset, fold, torch.device(args.device))


if __name__ == "__main__":
    main()
