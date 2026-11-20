"""Compare five separate snapshot scores + topology with probability-mean stacking.

The primary variant uses five logits, retaining the historical logit transform.
The raw-probability variant is a prespecified secondary control. Neural model,
snapshot selection, structural features and LR hyperparameters are unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from codes.data import load_root_fold
from codes.experiments_ablation import ExperimentConfig, run_experiment
from codes.run import compute_metrics, best_mcc_threshold, metrics_at_threshold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/snapshot_stacking_20261103'
CACHE=ROOT/'outputs/experimental_extension_20261012/ablation/full'
DATASETS=['CMI-9589','CMI-9905','CMI-20208']
METRICS=['specificity','precision','sensitivity','mcc','accuracy','auc','aupr']
STRUCTURE=['log1p_path3','log1p_mir_degree','log1p_circ_degree',
           'log1p_degree_product','normalized_path3']
VARIANTS=['mean_logit_structure_6d','snapshot_logits_structure_10d',
          'snapshot_probabilities_structure_10d']


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(p):
    return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1.0-p,1e-6,1-1e-6))


def make_meta(snapshots,structure,variant):
    snapshots,structure=np.asarray(snapshots),np.asarray(structure)
    if snapshots.ndim!=2 or snapshots.shape[0]!=5:
        raise ValueError('Exactly five separately retained snapshots are required')
    if structure.shape!=(snapshots.shape[1],5):
        raise ValueError('Expected five training-graph features per query')
    if variant==VARIANTS[0]:
        scores=logit(snapshots.mean(axis=0))[:,None]
        names=['mean_probability_logit']
    elif variant==VARIANTS[1]:
        scores=logit(snapshots).T
        names=[f'snapshot_rank{i}_logit' for i in range(1,6)]
    elif variant==VARIANTS[2]:
        scores=snapshots.T
        names=[f'snapshot_rank{i}_probability' for i in range(1,6)]
    else:raise ValueError(variant)
    matrix=np.column_stack([scores,structure])
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite meta input')
    return matrix,names+STRUCTURE


def check_cache(d,split):
    for key in ['validation_pairs','validation_labels','test_pairs','test_labels',
                'selection_indices','meta_indices']:
        np.testing.assert_array_equal(d[key],split[key])
    selection,meta=d['selection_indices'],d['meta_indices']
    assert not len(np.intersect1d(selection,meta))
    assert np.array_equal(np.sort(np.r_[selection,meta]),np.arange(len(d['validation_labels'])))
    for p,label in [('train_pairs',split['train_labels']),
                    ('validation_pairs',split['validation_labels']),('test_pairs',split['test_labels'])]:
        assert len(split[p])==len(label)
        assert len(set(map(tuple,split[p])))==len(split[p])
    train=set(map(tuple,split['train_pairs']))
    val=set(map(tuple,split['validation_pairs']))
    test=set(map(tuple,split['test_pairs']))
    assert not train.intersection(val) and not train.intersection(test) and not val.intersection(test)
    for key in ['snapshot_validation_probabilities','snapshot_test_probabilities']:
        assert d[key].shape[0]==5 and np.isfinite(d[key]).all()
        assert d[key].min()>=0 and d[key].max()<=1


def fit_and_score(source,destination,dataset,fold,split_name):
    destination.mkdir(parents=True,exist_ok=True)
    with np.load(source/'predictions.npz') as saved:
        d={k:saved[k].copy() for k in saved.files}
    with np.load(source/'split_manifest.npz') as saved:
        split={k:saved[k].copy() for k in saved.files}
    check_cache(d,split)
    previous=json.loads((source/'result.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert previous['config']['variant']=='full' and previous['config']['d_model']==192
    for name in ['model.py','data.py']:
        assert manifest['sources_sha256'][name]==sha(ROOT/'circpu'/name)
    meta=d['meta_indices'];vy=d['validation_labels'];ty=d['test_labels']
    results={}; parameters={}; predictions={}
    for variant in VARIANTS:
        vx,names=make_meta(d['snapshot_validation_probabilities'],d['validation_structure'],variant)
        tx,test_names=make_meta(d['snapshot_test_probabilities'],d['test_structure'],variant)
        assert names==test_names
        stack=LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,random_state=42+fold)
        stack.fit(vx[meta],vy[meta])
        assert stack.n_iter_.max()<stack.max_iter
        vp=stack.predict_proba(vx)[:,1]
        tp=stack.predict_proba(tx)[:,1]
        threshold=best_mcc_threshold(vy[meta],vp[meta])
        results[variant]={'test':compute_metrics(ty,tp),
            'meta_fit':compute_metrics(vy[meta],vp[meta]),
            'threshold_selected_on_meta':metrics_at_threshold(ty,tp,threshold),
            'input_shape_meta':[len(meta),vx.shape[1]],'input_shape_test':list(tx.shape)}
        parameters[variant]={'feature_order':names,'coef':stack.coef_.tolist(),
            'intercept':stack.intercept_.tolist(),'classes':stack.classes_.tolist(),
            'fit_split':'disjoint meta validation subset only',
            'C':.5,'class_weight':'balanced','max_iter':2000,'n_iter':stack.n_iter_.tolist()}
        predictions[variant+'_validation']=vp
        predictions[variant+'_test']=tp
        if variant==VARIANTS[0]:
            # Exact paired baseline: same five snapshots, same rows and LR settings.
            np.testing.assert_allclose(tp,d['ensemble_stacked_test_probabilities'],atol=1e-10,rtol=0)
            for metric,value in previous['test_metrics']['ensemble_stacked'].items():
                assert abs(value-results[variant]['test'][metric])<1e-12
    np.savez_compressed(destination/'predictions.npz',
        **{k:d[k] for k in ['validation_pairs','validation_labels','test_pairs','test_labels',
            'selection_indices','meta_indices','validation_structure','test_structure',
            'snapshot_validation_probabilities','snapshot_test_probabilities','snapshot_epochs']},
        **predictions)
    write_json(destination/'stackers.json',parameters)
    result={'dataset':dataset,'fold':fold,'split':split_name,'source':str(source),
        'source_predictions_sha256':sha(source/'predictions.npz'),
        'source_manifest_sha256':sha(source/'manifest.json'),
        'script_sha256':sha(Path(__file__)),'snapshot_epochs_zero_based':d['snapshot_epochs'].tolist(),
        'snapshot_order':'descending selection (AUC+AP)/2, never test performance',
        'snapshot_test_correlation':np.corrcoef(d['snapshot_test_probabilities']).tolist(),
        'protocol':{'selection_meta_disjoint':True,'train_validation_test_pairs_disjoint':True,
            'test_used_for_fit_or_threshold':False,'primary_variant':VARIANTS[1],
            'secondary_variant':VARIANTS[2],'primary_threshold':.5,
            'positive_prior':'training-positive graph density; unchanged',
            'test_role':'held-out pairs within the source dataset, not external biological validation'},
        'counts':{'train':len(split['train_labels']),'selection':len(d['selection_indices']),
            'meta':len(meta),'test':len(ty),'test_positive':int(ty.sum())},
        'variants':results,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(destination/'result.json',result)
    print('STACK_COMPLETE',split_name,dataset,fold,
        {k:round(v['test']['mcc'],5) for k,v in results.items()},flush=True)
    return result


def train_root(args):
    torch.set_num_threads(4)
    for dataset in args.datasets:
        for fold in args.folds:
            destination=args.output_dir/'root'/dataset/f'fold{fold}'
            source=destination/'neural'
            data=load_root_fold(dataset,fold,ROOT,torch.device(args.device),seed=42,validation_fraction=.1)
            config=ExperimentConfig()
            files=[ROOT/'Dataset'/dataset/name for name in ['miRNA_order.csv','circRNA_order.csv',
                f'Positive_Sample_Train{fold}.csv',f'Negative_Sample_Train{fold}.csv',
                f'Positive_Sample_Test{fold}.csv',f'Negative_Sample_Test{fold}.csv']]
            write_json(destination/'input_manifest.json',{'split':'root','fold':fold,'seed':42,
                'validation_fraction_of_original_train':.1,'device':args.device,
                'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})
            print('TRAIN_ROOT',dataset,fold,args.device,flush=True)
            run_experiment(data,source,config)
            fit_and_score(source,destination,dataset,fold,'root')
            del data
            if torch.cuda.is_available():torch.cuda.empty_cache()


def summarize(output):
    rows=[]
    for p in sorted(output.glob('*/CMI-*/fold*/result.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        for variant,v in r['variants'].items():
            rows.append({'split':r['split'],'dataset':r['dataset'],'fold_zero_based':r['fold'],
                'variant':variant,**v['test'],'test_samples':r['counts']['test']})
    if not rows:raise ValueError('No completed results')
    fields=['split','dataset','fold_zero_based','variant']+METRICS+['test_samples']
    with (output/'per_fold_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    aggregated=[]
    for split in sorted(set(r['split'] for r in rows)):
        for dataset in DATASETS:
            for variant in VARIANTS:
                group=[r for r in rows if r['split']==split and r['dataset']==dataset and r['variant']==variant]
                if not group:continue
                row={'split':split,'dataset':dataset,'variant':variant,'n_splits':len(group)}
                for metric in METRICS:
                    a=np.array([r[metric] for r in group])
                    row[metric+'_mean']=float(a.mean())
                    row[metric+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None
                aggregated.append(row)
    write_json(output/'summary.json',aggregated)
    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregated[0]));w.writeheader();w.writerows(aggregated)
    for row in aggregated:print(json.dumps(row,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['cached','train-root','summarize'])
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--folds',nargs='+',type=int,choices=range(5),default=[0])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.task=='cached':
        for dataset in args.datasets:
            for fold in args.folds:
                fit_and_score(CACHE/dataset/f'fold{fold}',
                    args.output_dir/'fivefold'/dataset/f'fold{fold}',dataset,fold,'fivefold')
    elif args.task=='train-root':train_root(args)
    else:summarize(args.output_dir)


if __name__=='__main__':main()
