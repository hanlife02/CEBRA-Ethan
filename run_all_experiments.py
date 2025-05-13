"""
运行所有CEBRA实验的主脚本。

这个脚本按顺序运行论文"Learnable latent embeddings for joint behavioural and neural analysis"中的所有实验。
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import set_seed


def run_experiment(experiment_name: str):
    """运行指定的实验。
    
    Args:
        experiment_name: 实验名称
    """
    print(f"\n{'='*80}")
    print(f"运行实验: {experiment_name}")
    print(f"{'='*80}\n")
    
    # 导入并运行实验
    if experiment_name == 'synthetic':
        from experiments.synthetic.run_experiment import main
    elif experiment_name == 'hippocampus':
        from experiments.hippocampus.run_experiment import main
    elif experiment_name == 'primate':
        from experiments.primate.run_experiment import main
    elif experiment_name == 'allen':
        from experiments.allen.run_experiment import main
    elif experiment_name == 'video_decoding':
        from experiments.video_decoding.run_experiment import main
    elif experiment_name == 'multi_session':
        from experiments.multi_session.run_experiment import main
    else:
        raise ValueError(f"未知的实验: {experiment_name}")
    
    # 运行实验
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"\n实验 {experiment_name} 完成！")
    print(f"运行时间: {end_time - start_time:.2f}秒")


def main():
    """运行所有实验。"""
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 定义要运行的实验
    experiments = [
        'synthetic',
        'hippocampus',
        'primate',
        'allen',
        'video_decoding',
        'multi_session'
    ]
    
    # 运行所有实验
    for experiment in experiments:
        run_experiment(experiment)
    
    print("\n所有实验完成！")


if __name__ == "__main__":
    main()
