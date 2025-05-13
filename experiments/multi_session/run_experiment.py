"""
实验6：多会话、多动物CEBRA训练

这个实验联合训练跨会话和不同动物的数据，
研究联合训练对嵌入一致性的提升，
测试预训练模型在新动物数据上的快速适应能力。
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
from config import MultiSessionConfig
from data_loader import load_hippocampus_data
from utils import evaluate_consistency, set_seed, visualize_embeddings


def run_cebra_single_animal(
    data: Dict[str, Dict[str, np.ndarray]],
    config: MultiSessionConfig,
    animal_id: str
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA模型，使用单一动物数据。
    
    Args:
        data: 包含不同动物数据的字典
        config: 实验配置
        animal_id: 动物ID
        
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
    animal_data = data[animal_id]
    
    # 训练模型
    start_time = time.time()
    model.fit(animal_data['train_neural'], animal_data['train_position'])
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(animal_data['test_neural'])
    
    return model, embeddings, {'training_time': training_time}


def run_cebra_joint(
    data: Dict[str, Dict[str, np.ndarray]],
    config: MultiSessionConfig
) -> Tuple[cebra.CEBRA, Dict[str, np.ndarray], Dict[str, float]]:
    """运行CEBRA模型，联合训练多个动物数据。
    
    Args:
        data: 包含不同动物数据的字典
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
    animal_ids = list(data.keys())
    train_neural_list = [data[animal_id]['train_neural'] for animal_id in animal_ids]
    train_position_list = [data[animal_id]['train_position'] for animal_id in animal_ids]
    
    # 联合训练模型
    start_time = time.time()
    model.fit(train_neural_list, train_position_list)
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = {
        animal_id: model.transform(data[animal_id]['test_neural'])
        for animal_id in animal_ids
    }
    
    return model, embeddings, {'training_time': training_time}


def evaluate_transfer_learning(
    data: Dict[str, Dict[str, np.ndarray]],
    config: MultiSessionConfig,
    source_animal_id: str,
    target_animal_id: str,
    fine_tune_iterations: int = 1000
) -> Dict[str, float]:
    """评估迁移学习性能。
    
    Args:
        data: 包含不同动物数据的字典
        config: 实验配置
        source_animal_id: 源动物ID
        target_animal_id: 目标动物ID
        fine_tune_iterations: 微调迭代次数
        
    Returns:
        迁移学习性能指标
    """
    # 在源动物上训练模型
    print(f"在{source_animal_id}上训练模型...")
    source_model, _, _ = run_cebra_single_animal(data, config, source_animal_id)
    
    # 在目标动物上从头训练模型
    print(f"在{target_animal_id}上从头训练模型...")
    scratch_config = MultiSessionConfig(
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
    scratch_model, scratch_embeddings, _ = run_cebra_single_animal(data, scratch_config, target_animal_id)
    
    # 在目标动物上微调源模型
    print(f"在{target_animal_id}上微调{source_animal_id}的模型...")
    fine_tune_config = MultiSessionConfig(
        output_dimension=config.output_dimension,
        model_architecture=config.model_architecture,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate * 0.1,  # 降低学习率
        max_iterations=fine_tune_iterations,
        temperature=config.temperature,
        device=config.device,
        verbose=config.verbose,
        time_offsets=config.time_offsets,
        conditional=config.conditional,
        delta=config.delta,
        distance=config.distance
    )
    
    # 复制源模型
    fine_tune_model = cebra.CEBRA(
        output_dimension=config.output_dimension,
        model_architecture=config.model_architecture,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate * 0.1,  # 降低学习率
        max_iterations=fine_tune_iterations,
        temperature=config.temperature,
        device=config.device,
        verbose=config.verbose,
        time_offsets=config.time_offsets,
        conditional=config.conditional,
        delta=config.delta,
        distance=config.distance
    )
    
    # 加载源模型的权重
    # 注意：这里假设CEBRA模型有一个load_state_dict方法
    # 实际实现可能需要根据CEBRA的API进行调整
    fine_tune_model.model = source_model.model
    
    # 微调模型
    start_time = time.time()
    fine_tune_model.fit(data[target_animal_id]['train_neural'], data[target_animal_id]['train_position'])
    fine_tune_time = time.time() - start_time
    
    # 生成嵌入
    fine_tune_embeddings = fine_tune_model.transform(data[target_animal_id]['test_neural'])
    
    # 评估位置解码性能
    from sklearn.neighbors import KNeighborsRegressor
    
    # 从头训练模型的性能
    scratch_decoder = KNeighborsRegressor(n_neighbors=5)
    scratch_decoder.fit(scratch_embeddings, data[target_animal_id]['test_position'])
    scratch_pred = scratch_decoder.predict(scratch_embeddings)
    scratch_r2 = r2_score(data[target_animal_id]['test_position'], scratch_pred)
    
    # 微调模型的性能
    fine_tune_decoder = KNeighborsRegressor(n_neighbors=5)
    fine_tune_decoder.fit(fine_tune_embeddings, data[target_animal_id]['test_position'])
    fine_tune_pred = fine_tune_decoder.predict(fine_tune_embeddings)
    fine_tune_r2 = r2_score(data[target_animal_id]['test_position'], fine_tune_pred)
    
    return {
        'scratch_r2': scratch_r2,
        'fine_tune_r2': fine_tune_r2,
        'fine_tune_time': fine_tune_time
    }


def compare_training_strategies(
    data: Dict[str, Dict[str, np.ndarray]],
    config: MultiSessionConfig
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """比较不同训练策略的性能。
    
    Args:
        data: 包含不同动物数据的字典
        config: 实验配置
        
    Returns:
        包含不同训练策略性能指标的字典
    """
    animal_ids = list(data.keys())
    results = {
        'single': {},
        'joint': {},
        'transfer': {}
    }
    
    # 单独训练
    for animal_id in animal_ids:
        print(f"单独训练{animal_id}...")
        _, embeddings, metrics = run_cebra_single_animal(data, config, animal_id)
        
        # 评估位置解码性能
        from sklearn.neighbors import KNeighborsRegressor
        
        decoder = KNeighborsRegressor(n_neighbors=5)
        decoder.fit(embeddings, data[animal_id]['test_position'])
        pred = decoder.predict(embeddings)
        r2 = r2_score(data[animal_id]['test_position'], pred)
        
        results['single'][animal_id] = {
            'r2_score': r2,
            'training_time': metrics['training_time']
        }
        
        print(f"  R²分数: {r2:.4f}")
        print(f"  训练时间: {metrics['training_time']:.2f}秒")
    
    # 联合训练
    print("联合训练所有动物...")
    _, embeddings_dict, metrics = run_cebra_joint(data, config)
    
    for animal_id in animal_ids:
        # 评估位置解码性能
        from sklearn.neighbors import KNeighborsRegressor
        
        decoder = KNeighborsRegressor(n_neighbors=5)
        decoder.fit(embeddings_dict[animal_id], data[animal_id]['test_position'])
        pred = decoder.predict(embeddings_dict[animal_id])
        r2 = r2_score(data[animal_id]['test_position'], pred)
        
        results['joint'][animal_id] = {
            'r2_score': r2,
            'training_time': metrics['training_time'] / len(animal_ids)  # 平均每个动物的训练时间
        }
        
        print(f"  {animal_id} R²分数: {r2:.4f}")
    
    print(f"  总训练时间: {metrics['training_time']:.2f}秒")
    
    # 迁移学习
    if len(animal_ids) >= 2:
        for i, source_id in enumerate(animal_ids):
            for j, target_id in enumerate(animal_ids):
                if i == j:
                    continue
                
                print(f"从{source_id}迁移到{target_id}...")
                transfer_metrics = evaluate_transfer_learning(
                    data, config, source_id, target_id
                )
                
                results['transfer'][f"{source_id}_to_{target_id}"] = transfer_metrics
                
                print(f"  从头训练R²分数: {transfer_metrics['scratch_r2']:.4f}")
                print(f"  微调R²分数: {transfer_metrics['fine_tune_r2']:.4f}")
                print(f"  微调时间: {transfer_metrics['fine_tune_time']:.2f}秒")
    
    return results


def visualize_training_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'r2_score',
    title: str = "训练策略比较",
    save_path: Optional[str] = None,
    show: bool = True
):
    """可视化不同训练策略的比较。
    
    Args:
        results: 包含不同训练策略性能指标的字典
        metric: 要可视化的指标
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
    """
    # 准备数据
    animal_ids = list(results['single'].keys())
    strategies = ['single', 'joint']
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 创建分组条形图
    x = np.arange(len(animal_ids))
    width = 0.35
    
    single_values = [results['single'][animal_id][metric] for animal_id in animal_ids]
    joint_values = [results['joint'][animal_id][metric] for animal_id in animal_ids]
    
    plt.bar(x - width/2, single_values, width, label='Single')
    plt.bar(x + width/2, joint_values, width, label='Joint')
    
    plt.xlabel('Animal')
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(x, animal_ids)
    plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """运行多会话、多动物CEBRA训练实验。"""
    # 加载配置
    config = MultiSessionConfig()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 创建输出目录
    os.makedirs(os.path.join(config.output_dir, 'multi_session'), exist_ok=True)
    
    # 加载数据
    print("加载多动物数据...")
    data = load_hippocampus_data(
        data_dir=config.data_dir,
        dataset_name=config.dataset_names[0],  # 使用第一个数据集
        bin_width_ms=25,
        preprocessing='zscore',
        test_size=0.2,
        random_seed=config.random_seed
    )
    
    # 比较训练策略
    print("比较训练策略...")
    results = compare_training_strategies(data, config)
    
    # 可视化训练策略比较
    print("可视化训练策略比较...")
    visualize_training_comparison(
        results=results,
        metric='r2_score',
        title="不同训练策略的位置解码性能(R²)比较",
        save_path=os.path.join(config.output_dir, 'multi_session', 'training_comparison.png'),
        show=False
    )
    
    # 如果有迁移学习结果，可视化迁移学习比较
    if 'transfer' in results and results['transfer']:
        # 准备数据
        transfer_pairs = list(results['transfer'].keys())
        scratch_values = [results['transfer'][pair]['scratch_r2'] for pair in transfer_pairs]
        fine_tune_values = [results['transfer'][pair]['fine_tune_r2'] for pair in transfer_pairs]
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(transfer_pairs))
        width = 0.35
        
        plt.bar(x - width/2, scratch_values, width, label='From Scratch')
        plt.bar(x + width/2, fine_tune_values, width, label='Fine-tuned')
        
        plt.xlabel('Transfer Pair')
        plt.ylabel('R² Score')
        plt.title("Transfer Learning Performance Comparison")
        plt.xticks(x, transfer_pairs, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_dir, 'multi_session', 'transfer_comparison.png'), dpi=300)
        plt.close()
    
    print("实验完成！")


if __name__ == "__main__":
    main()
