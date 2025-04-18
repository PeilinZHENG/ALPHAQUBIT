#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sycamore 实验数据预处理脚本
此脚本用于将 Google Sycamore 公开实验数据（JSONL 格式）转换为
适合机器学习训练的样本，输出为 Python pickle 文件。

用法：
    python preprocess_sycamore.py \
        --input_dir /path/to/jsonl_files \
        --output_file sycamore_dataset.pkl

输出样本格式（list of dict）：
    [
        {
            'code_distance': int,        # 码距离（3 或 5）
            'basis': str,               # 测量基态 ('X' 或 'Z')
            'rounds': int,              # 纠错轮数，如 1,3,...,25
            'detection_events': np.ndarray(shape=(rounds-1, num_stabilizers), dtype=int),
            'label': int                # 逻辑错误标签：0 = 正确，1 = 错误
        },
        ...
    ]
"""

import os
import json
import argparse
import pickle
import numpy as np

def load_jsonl_records(input_dir):
    """
    遍历 input_dir 下所有 .jsonl 文件，逐行读取 JSON 记录。
    """
    for fname in os.listdir(input_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, 'r') as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # 跳过解析错误的行
                    continue

def compute_detection_events(stab_meas):
    """
    计算 detection events：即相邻两轮 stabilizer 测量值的 XOR。
    输入 stab_meas: list 或 np.ndarray，形状 (rounds, num_stabilizers)
    返回 numpy 数组，形状 (rounds-1, num_stabilizers)
    """
    arr = np.array(stab_meas, dtype=int)
    # 相邻行异或，True->1, False->0
    return np.bitwise_xor(arr[1:], arr[:-1])

def get_logical_error_label(initial, final):
    """
    根据初始态和最终测量态判断逻辑错误：
        相同 -> 0 (无错误)
        不同 -> 1 (有错误)
    """
    return int(initial != final)

def build_sample(record):
    """
    从一条原始记录构建训练样本字典。
    失败时返回 None。
    """
    # 检查必要字段
    if 'stabilizer_measurements' not in record \
       or 'initial_state' not in record \
       or 'final_measurement' not in record:
        return None
    
    rounds = record.get('rounds')
    stab_meas = record['stabilizer_measurements']
    # 保证数据长度与 rounds 匹配
    if not isinstance(stab_meas, list) or len(stab_meas) != rounds:
        return None
    
    # 计算 detection events
    det_events = compute_detection_events(stab_meas)
    
    # 计算逻辑错误标签
    label = get_logical_error_label(record['initial_state'], record['final_measurement'])
    
    return {
        'code_distance': record.get('code_distance'),
        'basis': record.get('basis'),
        'rounds': rounds,
        'detection_events': det_events,
        'label': label
    }

def main():
    parser = argparse.ArgumentParser(description="预处理 Sycamore 实验数据")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="目录，包含 .jsonl 数据文件")
    parser.add_argument('--output_file', type=str, required=True,
                        help="输出 pickle 文件路径，例如 sycamore_dataset.pkl")
    args = parser.parse_args()
    
    samples = []
    for rec in load_jsonl_records(args.input_dir):
        sample = build_sample(rec)
        if sample is not None:
            samples.append(sample)
    
    # 保存结果
    with open(args.output_file, 'wb') as fout:
        pickle.dump(samples, fout)
    
    print(f"已处理 {len(samples)} 个样本，保存至 {args.output_file}")

if __name__ == '__main__':
    main()

