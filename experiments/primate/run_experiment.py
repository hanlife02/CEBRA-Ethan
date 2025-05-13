"""
实验3：灵长类运动任务潜在动态分析

这个实验使用灵长类体感皮质(S1)八方向中心外运动任务的电生理记录数据，
分析主动和被动运动对神经群体活动的影响，
测试位置、方向和主动/被动状态的解码性能，
结果显示S1神经元对位置和主动/被动状态的编码更为明显。
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cebra
from config import PrimateConfig
from data_loader import load_primate_data
from utils import evaluate_embeddings, set_seed, visualize_embeddings


def run_cebra_position(
    data: Dict[str, Dict[str, np.ndarray]],
    config: PrimateConfig,
    condition: str
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA模型，使用位置作为标签。
    
    Args:
        data: 包含主动和被动运动数据的字典
        config: 实验配置
        condition: 条件，可以是'active'或'passive'
        
    Returns:
        训练好的CEBRA模型、嵌入和评估指标
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
    condition_data = data[condition]
    
    # 训练模型
    start_time = time.time()
    model.fit(condition_data['train_neural'], condition_data['train_position'])
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(condition_data['test_neural'])
    
    # 评估位置解码性能
    position_metrics = evaluate_embeddings(
        embeddings=embeddings,
        target_data=condition_data['test_position'],
        task_type='regression',
        k=config.position_decoding_k,
        test_size=0.0,  # 已经是测试集
        random_seed=config.random_seed
    )
    
    position_metrics['training_time'] = training_time
    
    return model, embeddings, position_metrics


def run_cebra_direction(
    data: Dict[str, Dict[str, np.ndarray]],
    config: PrimateConfig,
    condition: str
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA模型，使用方向作为标签。
    
    Args:
        data: 包含主动和被动运动数据的字典
        config: 实验配置
        condition: 条件，可以是'active'或'passive'
        
    Returns:
        训练好的CEBRA模型、嵌入和评估指标
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
    condition_data = data[condition]
    
    # 训练模型
    start_time = time.time()
    model.fit(condition_data['train_neural'], condition_data['train_direction'])
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(condition_data['test_neural'])
    
    # 评估方向解码性能
    direction_metrics = evaluate_embeddings(
        embeddings=embeddings,
        target_data=condition_data['test_direction'],
        task_type='regression',
        k=config.position_decoding_k,
        test_size=0.0,  # 已经是测试集
        random_seed=config.random_seed
    )
    
    direction_metrics['training_time'] = training_time
    
    return model, embeddings, direction_metrics


def decode_active_passive(
    data: Dict[str, Dict[str, np.ndarray]],
    config: PrimateConfig
) -> Dict[str, float]:
    """解码主动/被动状态。
    
    Args:
        data: 包含主动和被动运动数据的字典
        config: 实验配置
        
    Returns:
        解码性能指标
    """
    # 准备数据
    active_data = data['active']
    passive_data = data['passive']
    
    # 合并训练数据
    X_train = np.vstack([active_data['train_neural'], passive_data['train_neural']])
    y_train = np.hstack([np.ones(len(active_data['train_neural'])), np.zeros(len(passive_data['train_neural']))])
    
    # 合并测试数据
    X_test = np.vstack([active_data['test_neural'], passive_data['test_neural']])
    y_test = np.hstack([np.ones(len(active_data['test_neural'])), np.zeros(len(passive_data['test_neural']))])
    
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
    
    # 训练模型
    start_time = time.time()
    model.fit(X_train)
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings_train = model.transform(X_train)
    embeddings_test = model.transform(X_test)
    
    # 使用KNN分类器解码主动/被动状态
    classifier = KNeighborsClassifier(n_neighbors=config.active_passive_k)
    classifier.fit(embeddings_train, y_train)
    
    # 预测并评估
    y_pred = classifier.predict(embeddings_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'training_time': training_time
    }


def compare_decoding_performance(
    data: Dict[str, Dict[str, np.ndarray]],
    config: PrimateConfig
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """比较不同任务的解码性能。
    
    Args:
        data: 包含主动和被动运动数据的字典
        config: 实验配置
        
    Returns:
        包含不同任务和条件的解码性能指标的字典
    """
    results = {
        'position': {},
        'direction': {},
        'active_passive': {}
    }
    
    # 位置解码
    for condition in ['active', 'passive']:
        print(f"评估{condition}条件下的位置解码性能...")
        _, _, metrics = run_cebra_position(data, config, condition)
        results['position'][condition] = metrics
        print(f"  R²分数: {metrics['r2_score']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    # 方向解码
    for condition in ['active', 'passive']:
        print(f"评估{condition}条件下的方向解码性能...")
        _, _, metrics = run_cebra_direction(data, config, condition)
        results['direction'][condition] = metrics
        print(f"  R²分数: {metrics['r2_score']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    # 主动/被动状态解码
    print("评估主动/被动状态解码性能...")
    metrics = decode_active_passive(data, config)
    results['active_passive']['combined'] = metrics
    print(f"  准确率: {metrics['accuracy']:.4f}")
    
    return results


def visualize_decoding_performance(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'r2_score',
    title: str = "解码性能比较",
    save_path: Optional[str] = None,
    show: bool = True
):
    """可视化解码性能比较。
    
    Args:
        results: 包含不同任务和条件的解码性能指标的字典
        metric: 要可视化的指标
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
    """
    # 准备数据
    tasks = []
    conditions = []
    values = []
    
    for task, task_results in results.items():
        if task == 'active_passive':
            if metric == 'accuracy':
                tasks.append(task)
                conditions.append('combined')
                values.append(task_results['combined']['accuracy'])
        else:
            for condition, condition_results in task_results.items():
                tasks.append(task)
                conditions.append(condition)
                values.append(condition_results[metric])
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 创建分组条形图
    x = np.arange(len(set(tasks)))
    width = 0.35
    
    active_values = [values[i] for i in range(len(values)) if conditions[i] == 'active']
    passive_values = [values[i] for i in range(len(values)) if conditions[i] == 'passive']
    combined_values = [values[i] for i in range(len(values)) if conditions[i] == 'combined']
    
    unique_tasks = list(set(tasks))
    
    if active_values:
        plt.bar(x - width/2, active_values, width, label='Active')
    if passive_values:
        plt.bar(x + width/2, passive_values, width, label='Passive')
    if combined_values:
        plt.bar(x + width*1.5, combined_values, width, label='Combined')
    
    plt.xlabel('Task')
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(x, unique_tasks)
    plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """运行灵长类运动任务潜在动态分析实验。"""
    # 加载配置
    config = PrimateConfig()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 创建输出目录
    os.makedirs(os.path.join(config.output_dir, 'primate'), exist_ok=True)
    
    # 加载数据
    print("加载灵长类运动任务数据...")
    data = load_primate_data(
        data_dir=config.data_dir,
        dataset_name=config.dataset_name,
        bin_width_ms=config.bin_width_ms,
        gaussian_smoothing_sigma_ms=config.gaussian_smoothing_sigma_ms,
        preprocessing='zscore',
        test_size=0.2,
        random_seed=config.random_seed
    )
    
    # 运行CEBRA-Position（主动条件）
    print("运行CEBRA-Position（主动条件）...")
    model_position_active, embeddings_position_active, metrics_position_active = run_cebra_position(
        data, config, 'active'
    )
    
    # 可视化CEBRA-Position（主动条件）嵌入
    print("可视化CEBRA-Position（主动条件）嵌入...")
    visualize_embeddings(
        embeddings=embeddings_position_active,
        labels=data['active']['test_position'][:, 0],  # 使用第一个位置维度作为颜色
        dimension=3,
        title="CEBRA-Position (Active) Embeddings",
        save_path=os.path.join(config.output_dir, 'primate', 'cebra_position_active_embeddings.png'),
        show=False
    )
    
    # 运行CEBRA-Position（被动条件）
    print("运行CEBRA-Position（被动条件）...")
    model_position_passive, embeddings_position_passive, metrics_position_passive = run_cebra_position(
        data, config, 'passive'
    )
    
    # 可视化CEBRA-Position（被动条件）嵌入
    print("可视化CEBRA-Position（被动条件）嵌入...")
    visualize_embeddings(
        embeddings=embeddings_position_passive,
        labels=data['passive']['test_position'][:, 0],  # 使用第一个位置维度作为颜色
        dimension=3,
        title="CEBRA-Position (Passive) Embeddings",
        save_path=os.path.join(config.output_dir, 'primate', 'cebra_position_passive_embeddings.png'),
        show=False
    )
    
    # 比较解码性能
    print("比较解码性能...")
    decoding_results = compare_decoding_performance(data, config)
    
    # 可视化解码性能比较
    print("可视化解码性能比较...")
    visualize_decoding_performance(
        results=decoding_results,
        metric='r2_score',
        title="不同任务和条件的解码性能(R²)比较",
        save_path=os.path.join(config.output_dir, 'primate', 'decoding_performance_r2.png'),
        show=False
    )
    
    # 可视化主动/被动状态解码性能
    if 'accuracy' in decoding_results['active_passive']['combined']:
        visualize_decoding_performance(
            results=decoding_results,
            metric='accuracy',
            title="主动/被动状态解码准确率",
            save_path=os.path.join(config.output_dir, 'primate', 'active_passive_decoding_accuracy.png'),
            show=False
        )
    
    print("实验完成！")


if __name__ == "__main__":
    main()
