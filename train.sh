#!/usr/bin/env bash
# 裁切原始数据，划分本地验证集
python3 ./data/cut_data.py
python3 ./data/select_val_dataset.py

# 简单赛道
# python3 train.py \
# --name simple_1

# python3 test_model.py \
# --name simple_1 \
# --test_model_path ./log_complex/simple_1/ckpt/best_ckpt.pth

# 复制赛道
python3 train_complex.py \
--name complex_1

python3 /home/sjw/Desktop/SpongeBobbb+3+神经元复杂赛道/sjw/test_model_complex.py \
--name complex_1 \
--test_model_path ./log_complex/complex_1/ckpt/best_ckpt.pth