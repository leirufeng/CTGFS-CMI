"""Reproducible ablations with disjoint snapshot-selection and meta-fit splits.

API: run_experiment(fold_data, output_dir, config=None, selection_indices=None,
                    meta_indices=None, training_node_mask=None, active_id_mask=None).
Indices address [validation_positive; validation_negative] in FoldData.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from codes.data import FoldData, encode_sequences, load_fold
from codes.experiments_model import ExperimentalCIRCPU, VARIANTS
from codes.model import info_nce, non_negative_pu_risk
from codes.run import set_seed, hard_negative_weights, compute_metrics

@dataclass
class ExperimentConfig:
    variant: str = 'full'
    epochs: int = 30
    patience: int = 8
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    alignment_weight: float = 0.02
    pu_weight: float = 0.03
    seed: int = 42
    split_seed: int = 1042
    d_model: int = 192
    sequence_layers: int = 1
    graph_layers: int = 3
    ensemble_size: int = 5
    circ_segment_length: int = 64
    circ_segments: int = 4
    save_weights: bool = True


def logit_features(probabilities, structure=None):
    p = np.asarray(probabilities)
    logit = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1.0-p, 1e-6, 1-1e-6))
    return logit[:, None] if structure is None else np.column_stack([logit, structure])


def split_validation(labels, seed):
    indices = np.arange(len(labels))
    selection, meta = train_test_split(indices, test_size=0.5, random_state=seed, stratify=labels)
    return np.sort(selection), np.sort(meta)


def _pairs(positive, negative):
    return np.concatenate([positive, negative]), np.r_[np.ones(len(positive)), np.zeros(len(negative))].astype(np.float32)


def _save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.inference_mode()
def predict(model, pairs, norm_adj, batch_size=8192):
    model.eval()
    mir, circ, _, _ = model.encode_nodes(norm_adj)
    chunks = []
    for start in range(0, len(pairs), batch_size):
        block = pairs[start:start+batch_size]
        scores = model.decoder(circ[block[:, 1]], mir[block[:, 0]])
        chunks.append(torch.sigmoid(scores).cpu().numpy())
    return np.concatenate(chunks)


def run_experiment(fold_data: FoldData, output_dir: Path | str,
                   config: ExperimentConfig | None = None,
                   selection_indices=None, meta_indices=None,
                   training_node_mask=None, active_id_mask=None):
    config = config or ExperimentConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding='utf-8'))
        if old.get('config') != asdict(config):
            raise ValueError(f'Existing result config mismatch: {output_dir}')
        return old
    device = fold_data.norm_adj.device
    set_seed(config.seed + fold_data.fold)
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    train_pairs, train_y = _pairs(fold_data.train_positive, fold_data.train_negative)
    val_pairs, val_y = _pairs(fold_data.validation_positive, fold_data.validation_negative)
    test_pairs, test_y = _pairs(fold_data.test_positive, fold_data.test_negative)
    if selection_indices is None or meta_indices is None:
        if selection_indices is not None or meta_indices is not None:
            raise ValueError('Provide both selection_indices and meta_indices')
        selection_indices, meta_indices = split_validation(val_y, config.split_seed + fold_data.fold)
    selection_indices, meta_indices = np.asarray(selection_indices), np.asarray(meta_indices)
    if len(np.intersect1d(selection_indices, meta_indices)) or len(np.unique(np.r_[selection_indices, meta_indices])) != len(val_y):
        raise ValueError('Selection/meta must be a disjoint partition of validation')
    for indices in (selection_indices, meta_indices):
        if len(np.unique(val_y[indices])) != 2:
            raise ValueError('Both classes required in selection and meta splits')
    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=device)
    if training_node_mask is not None:
        training_node_mask = tensor(training_node_mask, torch.bool)
    if active_id_mask is not None:
        active_id_mask = tensor(active_id_mask, torch.bool)
    pairs_t = tensor(train_pairs, torch.long)
    train_y_t = tensor(train_y, torch.float32)
    val_t = tensor(val_pairs, torch.long)
    test_t = tensor(test_pairs, torch.long)
    tokens = encode_sequences(fold_data.mir_sequences, fold_data.circ_sequences, device,
                             circ_segment_length=config.circ_segment_length,
                             circ_segment_count=config.circ_segments)
    model = ExperimentalCIRCPU(
        mir_tokens=tokens[0], mir_mask=tokens[1], circ_tokens=tokens[2], circ_mask=tokens[3],
        n_nodes=fold_data.n_mir+fold_data.n_circ, variant=config.variant,
        training_node_mask=training_node_mask, active_id_mask=active_id_mask,
        d_model=config.d_model, sequence_layers=config.sequence_layers,
        graph_layers=config.graph_layers, circ_segments=config.circ_segments,
        sequence_encoder='transformer', structure_dim=0, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1), eta_min=config.learning_rate*.05)
    align_weight = 0.0 if config.variant in ('no_alignment', 'sequence_only', 'graph_only') else config.alignment_weight
    pu_weight = 0.0 if config.variant == 'no_pu' else config.pu_weight
    manifest = {'dataset': fold_data.dataset, 'fold': fold_data.fold, 'config': asdict(config),
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'device': str(device), 'training_node_mask': training_node_mask is not None,
                'active_id_mask': active_id_mask is not None,
                'split_counts': {'train': len(train_y), 'selection': len(selection_indices), 'meta': len(meta_indices), 'test': len(test_y)},
                'positive_prior': fold_data.positive_prior,
                'snapshot_selection': 'mean(AUC, AP) on selection only',
                'meta_fit': 'LogisticRegression C=0.5 balanced on disjoint meta only',
                'threshold': 0.5, 'test_role': 'evaluation only; protocol designed after earlier project test results were available',
                'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name('experiments_model.py'), Path(__file__).with_name('model.py'), Path(__file__).with_name('data.py'))}}
    _save_json(output_dir / 'manifest.json', manifest)
    np.savez_compressed(output_dir/'split_manifest.npz', train_pairs=train_pairs, train_labels=train_y,
                        selection_indices=selection_indices, meta_indices=meta_indices,
                        validation_pairs=val_pairs, validation_labels=val_y,
                        test_pairs=test_pairs, test_labels=test_y,
                        training_node_mask=training_node_mask.cpu().numpy() if training_node_mask is not None else np.array([], dtype=bool),
                        active_id_mask=active_id_mask.cpu().numpy() if active_id_mask is not None else np.array([], dtype=bool))
    start_time = time.perf_counter()
    best_score, stale, logs, states = -float('inf'), 0, [], []
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(train_y), device=device)
        losses = []
        for offset in range(0, len(train_y), config.batch_size):
            idx = permutation[offset:offset+config.batch_size]
            if len(idx) < 2:
                continue
            scores, ma, ca = model(pairs_t[idx,0], pairs_t[idx,1], fold_data.norm_adj, return_alignment=True)
            y = train_y_t[idx]
            supervised = F.binary_cross_entropy_with_logits(scores, y, weight=hard_negative_weights(scores, y))
            alignment = scores.new_zeros(()) if ma is None or ca is None else 0.5*(info_nce(ma)+info_nce(ca))
            pu = non_negative_pu_risk(scores[y>.5], scores[y<.5], fold_data.positive_prior) if pu_weight else scores.new_zeros(())
            loss = supervised + align_weight*alignment + pu_weight*pu
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        # Meta labels and test labels are never queried during neural fitting.
        selection_p = predict(model, val_t[selection_indices], fold_data.norm_adj)
        metrics = compute_metrics(val_y[selection_indices], selection_p)
        score = 0.5*(metrics['auc']+metrics['aupr'])
        log = {'epoch': epoch, 'loss': float(np.mean(losses)), 'wall_seconds': time.perf_counter()-epoch_start,
               **{f'selection_{k}': v for k,v in metrics.items()}}
        logs.append(log)
        print(f"[{fold_data.dataset} {config.variant} fold{fold_data.fold}] epoch {epoch+1}/{config.epochs} loss={log['loss']:.4f} AUC={metrics['auc']:.4f} AP={metrics['aupr']:.4f} seconds={log['wall_seconds']:.1f}", flush=True)
        if score > best_score:
            best_score, stale = score, 0
        else:
            stale += 1
        if len(states) < config.ensemble_size or score > states[-1][0]:
            states.append((score, epoch, _cpu_state(model)))
            states.sort(key=lambda s: s[0], reverse=True)
            states = states[:config.ensemble_size]
        _save_json(output_dir/'learning_curve.json', logs)
        if stale >= config.patience:
            break
    val_snapshots, test_snapshots = [], []
    for rank, (score, epoch, state) in enumerate(states):
        model.load_state_dict(state)
        val_snapshots.append(predict(model, val_t, fold_data.norm_adj))
        test_snapshots.append(predict(model, test_t, fold_data.norm_adj))
        if config.save_weights:
            torch.save({'state_dict': state, 'config': asdict(config), 'epoch': epoch,
                        'selection_score': score, 'manifest': manifest}, output_dir/f'snapshot_rank{rank+1}.pt')
    val_snapshots, test_snapshots = np.asarray(val_snapshots), np.asarray(test_snapshots)
    all_predictions = {'validation_pairs': val_pairs, 'validation_labels': val_y,
                       'test_pairs': test_pairs, 'test_labels': test_y,
                       'selection_indices': selection_indices, 'meta_indices': meta_indices,
                       'validation_structure': fold_data.validation_structure, 'test_structure': fold_data.test_structure,
                       'snapshot_validation_probabilities': val_snapshots, 'snapshot_test_probabilities': test_snapshots,
                       'snapshot_epochs': np.array([s[1] for s in states])}
    test_metrics, stackers = {}, {}
    for name, vp, tp in [('single', val_snapshots[0], test_snapshots[0]),
                         ('ensemble', val_snapshots.mean(axis=0), test_snapshots.mean(axis=0))]:
        all_predictions[f'{name}_base_validation_probabilities'] = vp
        all_predictions[f'{name}_base_test_probabilities'] = tp
        test_metrics[f'{name}_base'] = compute_metrics(test_y, tp)
        stack = LogisticRegression(C=.5, class_weight='balanced', max_iter=2000, random_state=config.seed+fold_data.fold)
        stack.fit(logit_features(vp[meta_indices], fold_data.validation_structure[meta_indices]), val_y[meta_indices])
        stacked_vp = stack.predict_proba(logit_features(vp, fold_data.validation_structure))[:,1]
        stacked_tp = stack.predict_proba(logit_features(tp, fold_data.test_structure))[:,1]
        all_predictions[f'{name}_stacked_validation_probabilities'] = stacked_vp
        all_predictions[f'{name}_stacked_test_probabilities'] = stacked_tp
        test_metrics[f'{name}_stacked'] = compute_metrics(test_y, stacked_tp)
        stackers[name] = {'coef': stack.coef_.tolist(), 'intercept': stack.intercept_.tolist(), 'fit_split': 'meta',
                          'feature_order': ['base_logit','log1p_path3','log1p_mir_degree','log1p_circ_degree','log1p_degree_product','normalized_path3']}
    # Shared low-cost control for every model; no tuning on test labels.
    control_inputs = {'structure_only': (fold_data.validation_structure, fold_data.test_structure),
                      'score_only': (logit_features(val_snapshots.mean(axis=0)), logit_features(test_snapshots.mean(axis=0)))}
    for name, (vx, tx) in control_inputs.items():
        stack = LogisticRegression(C=.5, class_weight='balanced', max_iter=2000, random_state=config.seed+fold_data.fold)
        stack.fit(vx[meta_indices], val_y[meta_indices])
        probabilities = stack.predict_proba(tx)[:,1]
        all_predictions[f'{name}_test_probabilities'] = probabilities
        test_metrics[name] = compute_metrics(test_y, probabilities)
        stackers[name] = {'coef': stack.coef_.tolist(), 'intercept': stack.intercept_.tolist(), 'fit_split': 'meta'}
    np.savez_compressed(output_dir/'predictions.npz', **all_predictions)
    _save_json(output_dir/'stackers.json', stackers)
    result = {'dataset': fold_data.dataset, 'fold': fold_data.fold, 'config': asdict(config),
              'test_metrics': test_metrics, 'snapshot_epochs': [s[1] for s in states],
              'snapshot_selection_scores': [s[0] for s in states], 'epochs_run': len(logs),
              'wall_seconds': time.perf_counter()-start_time,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    _save_json(result_path, result)
    print(f"COMPLETE {fold_data.dataset} {config.variant} fold{fold_data.fold} MCC={test_metrics['ensemble_stacked']['mcc']:.5f} time={result['wall_seconds']:.1f}s", flush=True)
    del model, optimizer, states
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['CMI-9589','CMI-9905','CMI-20208'])
    parser.add_argument('--variants', nargs='+', default=['full'], choices=VARIANTS)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/experimental_extension_20261012/ablation'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    results=[]
    for variant in args.variants:
        for dataset in args.datasets:
            for fold in args.folds:
                data=load_fold(dataset, fold, Path('.').resolve(), torch.device(args.device))
                config=ExperimentConfig(variant=variant, epochs=args.epochs, seed=args.seed)
                results.append(run_experiment(data, args.output_dir/variant/dataset/f'fold{fold}', config))
                _save_json(args.output_dir/'completed_results.json', results)
if __name__ == '__main__':
    main()
