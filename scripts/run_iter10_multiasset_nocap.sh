#!/bin/bash
# 自动断点续训脚本：无需降低精度（无采样上限），自动从上一个 Epoch 恢复
# 如果沙箱销毁，重新运行此脚本即可从上次中断的 Epoch 继续训练，绝不丢失已完成的 Epoch 进度。

TRAIN_CMD="python /workspace/Finance/02_核心代码/源代码/khaos/模型训练/train.py \
  --data_dir /workspace/Finance/01_数据中心/03_研究数据/research_processed \
  --training_subdir iter10_multiasset \
  --save_dir /workspace/Finance/02_核心代码/模型权重备份/iter10_multiasset/iter10_multiasset_no1d_full30_es_nocap_v1 \
  --market legacy_multiasset \
  --assets BTCUSD,ESUSD,ETHUSD,EURUSD,SPXUSD,UDXUSD,WTIUSD,XAUUSD \
  --timeframes 5m,15m,60m,240m \
  --arch_version iterA4_multiscale \
  --constraint_profile teacher_feasible_discovery_v1 \
  --score_profile short_t_discovery_guarded_focus \
  --epochs 30 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.002 \
  --batch_size 256 \
  --fast_full \
  --resume"

echo "开始训练（无采样上限，全量数据）。"
echo "已启用 --resume 参数。每个 Epoch 结束后都会自动保存进度到 /workspace 下。"
echo "若遇到沙箱销毁，只需再次执行此脚本，即可无缝续训。"

$TRAIN_CMD
