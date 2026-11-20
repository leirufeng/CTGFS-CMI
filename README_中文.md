# 十维 stacking 模型完整保存包

本文件夹保存2026-11-03按历史协议重新训练的十维stacking版本，拷贝整理现有结果，没有重新训练。

## 文件

- `predict_10d.py`：十维专用入口，可校验导出文件，或加载真实PT权重重新预测测试集。
- `models/CMI-9589/fold1`至`fold5`、`models/CMI-9905/fold1`至`fold5`、`models/CMI-20208/fold1`至`fold5`：共15组。
- 每组`snapshot_rank1.pt`至`snapshot_rank5.pt`：共75个独立神经网络快照，按验证评分从高到低排列。
- 每组`stacker_10d.json`：该组已拟合LR的10个系数、截距和特征顺序，直接用于预测，无须重新拟合。
- 每组`metadata.json`：训练配置、快照对应epoch和七项测试指标。
- 每组`predictions.npz`：5个快照各自的验证/测试预测、结构特征、配对、标签和最终十维预测。
- `metrics/`：仅十维的三个数据集汇总和15折全部指标。
- `Dataset/`：本次运行使用的99个必要输入文件，包含原始5fold_CV划分及节点序列。整个文件夹可一并搬到其他位置。
- `circpu/`：实际训练的原始模型、数据处理和训练代码，以及原封不动保存的历史训练入口及其依赖。
- `bundle_manifest.json`：导出文件SHA256清单；`verification_export.json`：导出后验证结果。

## 模型如何使用

每组是“五个神经网络快照 + 一个10维LR”，75个PT不是75个独立stacking模型。
LR输入依次为五个快照的logit分数，再拼接五项训练图结构特征，不先对五个概率求平均。
五项结构特征顺序为log1p_path3、log1p_mir_degree、log1p_circ_degree、log1p_degree_product、normalized_path3。
`.pt`保存的是PyTorch state_dict和配置信息；最终LR另存为`stacker_10d.json`，使用时两者配套。
文件夹fold1~fold5对应原代码fold0~fold4，PT内部fold字段保留原来的0~4，未修改权重内容。
序列编码和图结构依赖Dataset中对应的数据及节点索引，不能把其他节点顺序直接替换进来。

## 校验和预测

先进入本文件夹，在具有PyTorch、NumPy、SciPy和scikit-learn的Python环境运行：

```powershell
python predict_10d.py verify
python predict_10d.py evaluate --datasets CMI-9589 --folds 1 --device cuda:0
python predict_10d.py evaluate --device cuda:0
```

第一条校验全部文件、75个PT和15组已保存十维预测的指标；第二条从实际PT重新预测一折；第三条从PT重新预测全部15折。
CPU也支持，把设备改为`--device cpu`，但会更慢。不同硬件的浮点计算可能有微小差异。
重新预测结果写入`reevaluation/`，已保存的模型与原始指标保持原样。
环境版本记录在`environment.json`和`requirements.txt`中；GPU使用时安装与设备对应的PyTorch/CUDA版本。

## 训练代码的来源

`circpu/experiments_historical_stacking.py`是产生这批PT的原始训练入口，完整保留六维配对对照分支用于追溯。
本包模型、LR参数、导出预测、指标文件和`predict_10d.py`主入口均对应十维版本。
如需从头重训，可运行：

```powershell
python -u -m circpu.experiments_historical_stacking worker --device cuda:0
python -m circpu.experiments_historical_stacking summarize
```

它读取本文件夹内的数据和历史manifest，保持原训练协议，另写到本文件夹的`outputs/`，不替换`models/`已保存权重。
原训练协议每组约80%训练、10%验证、10%测试；完整验证集同时用于快照选择和LR拟合；测试标签只用于评估。
分类阈值为0.5；旧代码aupr字段实际计算的是average_precision_score（AP）。
