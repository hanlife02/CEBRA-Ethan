"""
实验4：跨模态一致嵌入分析

这个实验使用Allen Brain Observatory数据集的钙成像和Neuropixels记录，
将视频特征作为"行为"标签用于训练CEBRA模型，
验证不同记录方法（钙成像vs电生理）是否产生类似的潜在表示，
测试联合训练对跨模态一致性的影响，
分析不同视觉区域内部和区域间的一致性。
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cebra
from config import AllenConfig
from data_loader import load_allen_data
from utils import evaluate_consistency, set_seed, visualize_embeddings


def run_cebra_single_modality(
    data: Dict[str, Dict[str, np.ndarray]],
    config: AllenConfig,
    modality: str
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA模型，使用单一模态数据。
    
    Args:
        data: 包含钙成像和Neuropixels数据的字典
        config: 实验配置
        modality: 模态，可以是'calcium'或'neuropixels'
        
    Returns:
        训练好的CEBRA模型、嵌入和训练时间
    """
    # 创建CEBRA模型
    model = cebra.CEBRA(
        output_dimension=config.output_dimension,
        model_architecture=config.model_architecture,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        device=config.device,
        verbose=config.verbose,
        time_offsets=config.time_offsets,
        conditional=config.conditional,
        delta=config.delta,
        distance=config.distance
    )
    
    # 准备数据
    modality_data = data[modality]
    
    # 训练模型
    start_time = time.time()
    model.fit(modality_data['train_neural'], modality_data['train_dino'])
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(modality_data['test_neural'])
    
    return model, embeddings, {'training_time': training_time}


def run_cebra_joint(
    data: Dict[str, Dict[str, np.ndarray]],
    config: AllenConfig
) -> Tuple[cebra.CEBRA, Dict[str, np.ndarray], Dict[str, float]]:
    """运行CEBRA模型，联合训练钙成像和Neuropixels数据。
    
    Args:
        data: 包含钙成像和Neuropixels数据的字典
        config: 实验配置
        
    Returns:
        训练好的CEBRA模型、嵌入字典和训练时间
    """
    # 创建CEBRA模型
    model = cebra.CEBRA(
        output_dimension=config.output_dimension,
        model_architecture=config.model_architecture,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        device=config.device,
        verbose=config.verbose,
        time_offsets=config.time_offsets,
        conditional=config.conditional,
        delta=config.delta,
        distance=config.distance
    )
    
    # 准备数据
    calcium_data = data['calcium']
    neuropixels_data = data['neuropixels']
    
    # 联合训练模型
    start_time = time.time()
    model.fit(
        [calcium_data['train_neural'], neuropixels_data['train_neural']],
        [calcium_data['train_dino'], neuropixels_data['train_dino']]
    )
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = {
        'calcium': model.transform(calcium_data['test_neural']),
        'neuropixels': model.transform(neuropixels_data['test_neural'])
    }
    
    return model, embeddings, {'training_time': training_time}


def evaluate_cross_modality_consistency(
    data: Dict[str, Dict[str, np.ndarray]],
    config: AllenConfig,
    training_type: str = 'separate'
) -> Dict[str, float]:
    """评估跨模态一致性。
    
    Args:
        data: 包含钙成像和Neuropixels数据的字典
        config: 实验配置
        training_type: 训练类型，可以是'separate'或'joint'
        
    Returns:
        一致性指标
    """
    if training_type == 'separate':
        # 单独训练每个模态
        print("单独训练钙成像模型...")
        model_calcium, embeddings_calcium, _ = run_cebra_single_modality(data, config, 'calcium')
        
        print("单独训练Neuropixels模型...")
        model_neuropixels, embeddings_neuropixels, _ = run_cebra_single_modality(data, config, 'neuropixels')
        
        # 使用钙成像模型转换Neuropixels数据
        neuropixels_in_calcium_space = model_calcium.transform(data['neuropixels']['test_neural'])
        
        # 评估一致性
        consistency = evaluate_consistency(
            embeddings1=embeddings_calcium,
            embeddings2=neuropixels_in_calcium_space,
            metric='r2',
            test_size=0.0,  # 已经是测试集
            random_seed=config.random_seed
        )
        
    else:  # training_type == 'joint'
        # 联合训练
        print("联合训练钙成像和Neuropixels模型...")
        _, embeddings, _ = run_cebra_joint(data, config)
        
        # 评估一致性
        consistency = evaluate_consistency(
            embeddings1=embeddings['calcium'],
            embeddings2=embeddings['neuropixels'],
            metric='r2',
            test_size=0.0,  # 已经是测试集
            random_seed=config.random_seed
        )
    
    return consistency


def compare_training_strategies(
    data: Dict[str, Dict[str, np.ndarray]],
    config: AllenConfig
) -> Dict[str, Dict[str, float]]:
    """比较不同训练策略的跨模态一致性。
    
    Args:
        data: 包含钙成像和Neuropixels数据的字典
        config: 实验配置
        
    Returns:
        包含不同训练策略一致性指标的字典
    """
    results = {}
    
    # 单独训练
    print("评估单独训练的跨模态一致性...")
    separate_consistency = evaluate_cross_modality_consistency(data, config, training_type='separate')
    results['separate'] = separate_consistency
    print(f"  R²分数: {separate_consistency['r2_score']:.4f}")
    
    # 联合训练
    print("评估联合训练的跨模态一致性...")
    joint_consistency = evaluate_cross_modality_consistency(data, config, training_type='joint')
    results['joint'] = joint_consistency
    print(f"  R²分数: {joint_consistency['r2_score']:.4f}")
    
    return results


def visualize_cross_modality_embeddings(
    data: Dict[str, Dict[str, np.ndarray]],
    config: AllenConfig,
    save_dir: str
):
    """可视化跨模态嵌入。
    
    Args:
        data: 包含钙成像和Neuropixels数据的字典
        config: 实验配置
        save_dir: 保存目录
    """
    # 联合训练
    print("联合训练钙成像和Neuropixels模型...")
    _, embeddings, _ = run_cebra_joint(data, config)
    
    # 可视化钙成像嵌入
    print("可视化钙成像嵌入...")
    visualize_embeddings(
        embeddings=embeddings['calcium'],
        labels=np.arange(len(embeddings['calcium'])),  # 使用时间作为颜色
        dimension=3,
        title="Calcium Imaging Embeddings",
        save_path=os.path.join(save_dir, 'calcium_embeddings.png'),
        show=False,
        cmap='viridis'
    )
    
    # 可视化Neuropixels嵌入
    print("可视化Neuropixels嵌入...")
    visualize_embeddings(
        embeddings=embeddings['neuropixels'],
        labels=np.arange(len(embeddings['neuropixels'])),  # 使用时间作为颜色
        dimension=3,
        title="Neuropixels Embeddings",
        save_path=os.path.join(save_dir, 'neuropixels_embeddings.png'),
        show=False,
        cmap='viridis'
    )


def visualize_consistency_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "跨模态一致性比较",
    save_path: Optional[str] = None,
    show: bool = True
):
    """可视化一致性比较。
    
    Args:
        results: 包含不同训练策略一致性指标的字典
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
    """
    strategies = list(results.keys())
    r2_scores = [results[strategy]['r2_score'] for strategy in strategies]
    
    plt.figure(figsize=(8, 6))
    plt.bar(strategies, r2_scores)
    plt.ylabel('R² Score')
    plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """运行跨模态一致嵌入分析实验。"""
    # 加载配置
    config = AllenConfig()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 创建输出目录
    os.makedirs(os.path.join(config.output_dir, 'allen'), exist_ok=True)
    
    # 加载数据
    print("加载Allen Brain Observatory数据...")
    data = load_allen_data(
        data_dir=config.data_dir,
        calcium_dataset=config.calcium_dataset,
        neuropixels_dataset=config.neuropixels_dataset,
        preprocessing='zscore',
        test_size=0.2,
        random_seed=config.random_seed
    )
    
    # 比较训练策略
    print("比较训练策略...")
    consistency_results = compare_training_strategies(data, config)
    
    # 可视化一致性比较
    print("可视化一致性比较...")
    visualize_consistency_comparison(
        results=consistency_results,
        title="不同训练策略的跨模态一致性(R²)比较",
        save_path=os.path.join(config.output_dir, 'allen', 'consistency_comparison.png'),
        show=False
    )
    
    # 可视化跨模态嵌入
    print("可视化跨模态嵌入...")
    visualize_cross_modality_embeddings(
        data=data,
        config=config,
        save_dir=os.path.join(config.output_dir, 'allen')
    )
    
    print("实验完成！")


if __name__ == "__main__":
    main()
