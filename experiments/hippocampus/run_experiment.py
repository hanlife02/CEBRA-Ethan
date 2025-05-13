"""
实验2：大鼠海马体数据分析

这个实验使用Grosmark与Buzsáki 2016年收集的大鼠线性轨道实验数据，
评估不同算法生成的潜在空间在不同大鼠之间的一致性，
进行假设驱动分析（位置、方向标签）和发现驱动分析（仅使用时间信息），
使用拓扑数据分析验证嵌入的拓扑结构稳健性，
比较位置解码性能。
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cebra
from config import HippocampusConfig
from data_loader import load_hippocampus_data
from utils import evaluate_consistency, evaluate_embeddings, set_seed, visualize_embeddings


def run_cebra_behaviour(
    data: Dict[str, Dict[str, np.ndarray]],
    config: HippocampusConfig,
    rat_id: str
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA-Behaviour模型。
    
    Args:
        data: 包含不同大鼠数据的字典
        config: 实验配置
        rat_id: 大鼠ID
        
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
    
    # 准备行为标签（位置和方向）
    rat_data = data[rat_id]
    position = rat_data['train_position']
    direction = rat_data['train_direction']
    behavior_labels = np.hstack([position, direction])
    
    # 训练模型
    start_time = time.time()
    model.fit(rat_data['train_neural'], behavior_labels)
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(rat_data['test_neural'])
    
    # 评估位置解码性能
    position_metrics = evaluate_embeddings(
        embeddings=embeddings,
        target_data=rat_data['test_position'],
        task_type='regression',
        k=config.position_decoding_k,
        test_size=0.0,  # 已经是测试集
        random_seed=config.random_seed
    )
    
    position_metrics['training_time'] = training_time
    
    return model, embeddings, position_metrics


def run_cebra_time(
    data: Dict[str, Dict[str, np.ndarray]],
    config: HippocampusConfig,
    rat_id: str
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA-Time模型。
    
    Args:
        data: 包含不同大鼠数据的字典
        config: 实验配置
        rat_id: 大鼠ID
        
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
        conditional='time',  # 使用时间条件
        distance=config.distance
    )
    
    # 训练模型
    rat_data = data[rat_id]
    start_time = time.time()
    model.fit(rat_data['train_neural'])
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(rat_data['test_neural'])
    
    # 评估位置解码性能
    position_metrics = evaluate_embeddings(
        embeddings=embeddings,
        target_data=rat_data['test_position'],
        task_type='regression',
        k=config.position_decoding_k,
        test_size=0.0,  # 已经是测试集
        random_seed=config.random_seed
    )
    
    position_metrics['training_time'] = training_time
    
    return model, embeddings, position_metrics


def evaluate_cross_rat_consistency(
    data: Dict[str, Dict[str, np.ndarray]],
    config: HippocampusConfig,
    model_type: str = 'behaviour'
) -> Dict[str, Dict[str, float]]:
    """评估不同大鼠之间的一致性。
    
    Args:
        data: 包含不同大鼠数据的字典
        config: 实验配置
        model_type: 模型类型，可以是'behaviour'或'time'
        
    Returns:
        包含不同大鼠对之间一致性指标的字典
    """
    # 获取所有大鼠ID
    rat_ids = list(data.keys())
    
    # 为每只大鼠训练模型并生成嵌入
    rat_models = {}
    rat_embeddings = {}
    
    for rat_id in rat_ids:
        print(f"训练{rat_id}的模型...")
        
        if model_type == 'behaviour':
            model, embeddings, _ = run_cebra_behaviour(data, config, rat_id)
        else:  # model_type == 'time'
            model, embeddings, _ = run_cebra_time(data, config, rat_id)
        
        rat_models[rat_id] = model
        rat_embeddings[rat_id] = embeddings
    
    # 评估不同大鼠对之间的一致性
    consistency_results = {}
    
    for i, rat1 in enumerate(rat_ids):
        for j, rat2 in enumerate(rat_ids):
            if i >= j:  # 只评估上三角矩阵
                continue
            
            pair_name = f"{rat1}_vs_{rat2}"
            print(f"评估{pair_name}的一致性...")
            
            # 使用rat1的模型转换rat2的数据
            rat2_in_rat1_space = rat_models[rat1].transform(data[rat2]['test_neural'])
            
            # 评估一致性
            consistency = evaluate_consistency(
                embeddings1=rat_embeddings[rat1],
                embeddings2=rat2_in_rat1_space,
                metric='r2',
                test_size=0.0,  # 已经是测试集
                random_seed=config.random_seed
            )
            
            consistency_results[pair_name] = consistency
    
    return consistency_results


def compare_position_decoding(
    data: Dict[str, Dict[str, np.ndarray]],
    config: HippocampusConfig
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """比较不同模型的位置解码性能。
    
    Args:
        data: 包含不同大鼠数据的字典
        config: 实验配置
        
    Returns:
        包含不同模型和大鼠的位置解码指标的字典
    """
    # 获取所有大鼠ID
    rat_ids = list(data.keys())
    
    # 比较不同模型
    model_types = ['behaviour', 'time']
    results = {model_type: {} for model_type in model_types}
    
    for model_type in model_types:
        print(f"评估{model_type}模型的位置解码性能...")
        
        for rat_id in rat_ids:
            print(f"  {rat_id}...")
            
            if model_type == 'behaviour':
                _, _, metrics = run_cebra_behaviour(data, config, rat_id)
            else:  # model_type == 'time'
                _, _, metrics = run_cebra_time(data, config, rat_id)
            
            results[model_type][rat_id] = metrics
            print(f"    R²分数: {metrics['r2_score']:.4f}")
            print(f"    MAE: {metrics['mae']:.4f}")
    
    return results


def visualize_embeddings_with_position(
    embeddings: np.ndarray,
    position: np.ndarray,
    title: str = "CEBRA Embeddings with Position",
    save_path: Optional[str] = None,
    show: bool = True
):
    """可视化嵌入，使用位置作为颜色。
    
    Args:
        embeddings: 嵌入，形状为(n_samples, n_dimensions)
        position: 位置，形状为(n_samples, 2)
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
    """
    # 使用位置的第一个维度作为颜色
    visualize_embeddings(
        embeddings=embeddings,
        labels=position[:, 0],
        dimension=3,
        title=title,
        save_path=save_path,
        show=show,
        cmap='viridis'
    )


def main():
    """运行大鼠海马体数据分析实验。"""
    # 加载配置
    config = HippocampusConfig()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 创建输出目录
    os.makedirs(os.path.join(config.output_dir, 'hippocampus'), exist_ok=True)
    
    # 加载数据
    print("加载大鼠海马体数据...")
    data = load_hippocampus_data(
        data_dir=config.data_dir,
        dataset_name=config.dataset_name,
        bin_width_ms=config.bin_width_ms,
        preprocessing='zscore',
        test_size=0.2,
        random_seed=config.random_seed
    )
    
    # 运行CEBRA-Behaviour
    print("运行CEBRA-Behaviour...")
    rat_id = list(data.keys())[0]  # 使用第一只大鼠
    model_behaviour, embeddings_behaviour, metrics_behaviour = run_cebra_behaviour(data, config, rat_id)
    
    # 可视化CEBRA-Behaviour嵌入
    print("可视化CEBRA-Behaviour嵌入...")
    visualize_embeddings_with_position(
        embeddings=embeddings_behaviour,
        position=data[rat_id]['test_position'],
        title="CEBRA-Behaviour Embeddings",
        save_path=os.path.join(config.output_dir, 'hippocampus', 'cebra_behaviour_embeddings.png'),
        show=False
    )
    
    # 运行CEBRA-Time
    print("运行CEBRA-Time...")
    model_time, embeddings_time, metrics_time = run_cebra_time(data, config, rat_id)
    
    # 可视化CEBRA-Time嵌入
    print("可视化CEBRA-Time嵌入...")
    visualize_embeddings_with_position(
        embeddings=embeddings_time,
        position=data[rat_id]['test_position'],
        title="CEBRA-Time Embeddings",
        save_path=os.path.join(config.output_dir, 'hippocampus', 'cebra_time_embeddings.png'),
        show=False
    )
    
    # 评估跨大鼠一致性
    print("评估跨大鼠一致性...")
    consistency_behaviour = evaluate_cross_rat_consistency(data, config, model_type='behaviour')
    consistency_time = evaluate_cross_rat_consistency(data, config, model_type='time')
    
    # 比较位置解码性能
    print("比较位置解码性能...")
    decoding_results = compare_position_decoding(data, config)
    
    # 打印结果
    print("\n结果摘要:")
    print("CEBRA-Behaviour位置解码R²分数:", metrics_behaviour['r2_score'])
    print("CEBRA-Time位置解码R²分数:", metrics_time['r2_score'])
    
    print("\n跨大鼠一致性(R²):")
    for pair, metrics in consistency_behaviour.items():
        print(f"  CEBRA-Behaviour {pair}: {metrics['r2_score']:.4f}")
    for pair, metrics in consistency_time.items():
        print(f"  CEBRA-Time {pair}: {metrics['r2_score']:.4f}")
    
    print("实验完成！")


if __name__ == "__main__":
    main()
