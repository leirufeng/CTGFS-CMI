# 按历史方案重新比较六维和十维 stacking

本次明确沿用用户原表对应的20261014 Transformer复现方案：读取Dataset/数据集/5fold_CV原文件，采用约80%训练、10%验证、10%测试的五组预定义划分。同一完整验证集既用于早停与筛选五个快照，也用于拟合LR；不再切换根目录文件或采用selection/meta拆分。

每组重新训练一次原始CIRC-PU模型，两种stacking共享该组同一批五个快照。六维为logit(mean(p1,...,p5))加五项结构特征；十维为logit(p1),...,logit(p5)加相同五项结构特征。两种LR都固定C=0.5、class_weight=balanced、max_iter=2000；主表分类阈值0.5。AP对应旧代码aupr字段。

所有神经训练调用原始circpu.run.train_one_fold函数，主函数、模型和数据处理源文件均不修改。仅以evaluate包装器记录原函数已经计算的逐轮验证与最终五个测试快照预测，不增加随机数抽取或神经网络前向计算。最终测试的五个模型状态均保存，以避免再次缺失单快照记录。

参数直接读取并核对历史run_manifest：d_model=192，Transformer1层4头，图传播3层，circRNA4×64nt，miRNA最多96nt，epochs=30，patience=8，batch_size=1024，lr=0.001，weight_decay=1e-5，alignment_weight=0.02，pu_weight=0.03，ensemble_size=5，seed=42+fold，eval_batch_size=2048。逐项核对历史数据文件和源代码SHA256一致。

比较分为两部分：本次同快照六维与十维的配对比较；本次六维与用户历史表六维的复现偏差。CUDA/cuDNN历史设置允许非确定计算，因此不预先保证重新训练与历史每个小数完全相同，不按测试分数挑选重跑。

同一验证集承担模型选择和元模型拟合是用户指定的历史协议，应如实报告，不宣称selection/meta互斥或OOF-stacking。测试标签不参与训练、快照选择或LR拟合。只比较已有数据集内部的测试配对，不宣称外部独立验证。

最终要求：三个数据集、每个五组，共15组神经网络和30组stacking指标全部完成；报告七项指标均值±样本标准差（ddof=1），逐组结果，同快照差值，历史表复现偏差，快照权重、逐快照预测、LR参数、完整来源审查。
