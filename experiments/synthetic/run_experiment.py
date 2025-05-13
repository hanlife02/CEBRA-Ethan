"""
实验1：合成数据验证实验

这个实验使用人工合成的神经元活动数据测试CEBRA重构真实潜在空间的能力，
并与t-SNE、UMAP、pi-VAE和autoLFADS等方法进行比较。
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from umap import UMAP

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cebra
from config import SyntheticConfig
from data_loader import load_synthetic_data
from utils import evaluate_embeddings, set_seed, visualize_embeddings


def run_cebra(
    data: Dict[str, np.ndarray],
    config: SyntheticConfig
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA模型。
    
    Args:
        data: 包含训练和测试数据的字典
        config: 实验配置
        
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
    
    # 训练模型
    start_time = time.time()
    model.fit(data['train_neural'])
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(data['test_neural'])
    
    # 评估嵌入
    metrics = evaluate_embeddings(
        embeddings=embeddings,
        target_data=data['test_latent'],
        task_type='regression',
        test_size=0.0,  # 已经是测试集
        random_seed=config.random_seed
    )
    
    metrics['training_time'] = training_time
    
    return model, embeddings, metrics


def run_baseline(
    data: Dict[str, np.ndarray],
    method: str,
    n_components: int = 3,
    random_seed: int = 42
) -> Tuple[np.ndarray, Dict[str, float]]:
    """运行基线方法。
    
    Args:
        data: 包含训练和测试数据的字典
        method: 方法名称，可以是'pca'、'tsne'、'umap'或'isomap'
        n_components: 组件数量
        random_seed: 随机种子
        
    Returns:
        嵌入和评估指标
    """
    # 设置随机种子
    set_seed(random_seed)
    
    # 选择方法
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=random_seed)
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=random_seed)
    elif method == 'umap':
        model = UMAP(n_components=n_components, random_state=random_seed)
    elif method == 'isomap':
        model = Isomap(n_components=n_components)
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    # 训练模型并生成嵌入
    start_time = time.time()
    embeddings = model.fit_transform(data['test_neural'])
    training_time = time.time() - start_time
    
    # 评估嵌入
    metrics = evaluate_embeddings(
        embeddings=embeddings,
        target_data=data['test_latent'],
        task_type='regression',
        test_size=0.0,  # 已经是测试集
        random_seed=random_seed
    )
    
    metrics['training_time'] = training_time
    
    return embeddings, metrics


def compare_methods(
    data: Dict[str, np.ndarray],
    config: SyntheticConfig,
    methods: List[str] = ['cebra', 'pca', 'tsne', 'umap', 'isomap']
) -> Dict[str, Dict[str, float]]:
    """比较不同方法的性能。
    
    Args:
        data: 包含训练和测试数据的字典
        config: 实验配置
        methods: 要比较的方法列表
        
    Returns:
        包含每种方法评估指标的字典
    """
    results = {}
    
    for method in methods:
        print(f"运行方法: {method}")
        
        if method == 'cebra':
            _, _, metrics = run_cebra(data, config)
        else:
            _, metrics = run_baseline(
                data,
                method=method,
                n_components=config.output_dimension,
                random_seed=config.random_seed
            )
        
        results[method] = metrics
        print(f"  R²分数: {metrics['r2_score']:.4f}")
        print(f"  训练时间: {metrics['training_time']:.2f}秒")
    
    return results


def visualize_results(
    results: Dict[str, Dict[str, float]],
    metric: str = 'r2_score',
    title: str = "方法比较",
    save_path: Optional[str] = None,
    show: bool = True
):
    """可视化比较结果。
    
    Args:
        results: 包含每种方法评估指标的字典
        metric: 要可视化的指标
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
    """
    methods = list(results.keys())
    values = [results[method][metric] for method in methods]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, values)
    plt.ylabel(metric)
    plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """运行合成数据验证实验。"""
    # 加载配置
    config = SyntheticConfig()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 创建输出目录
    os.makedirs(os.path.join(config.output_dir, 'synthetic'), exist_ok=True)
    
    # 加载数据
    print("加载合成数据...")
    data = load_synthetic_data(
        num_samples=config.num_samples,
        num_neurons=config.num_neurons,
        num_latent=config.num_latent,
        noise_level=config.noise_level,
        preprocessing='zscore',
        test_size=config.test_size,
        random_seed=config.random_seed
    )
    
    # 运行CEBRA
    print("运行CEBRA...")
    model, embeddings, metrics = run_cebra(data, config)
    
    # 可视化嵌入
    print("可视化嵌入...")
    visualize_embeddings(
        embeddings=embeddings,
        labels=data['test_latent'][:, 0],  # 使用第一个潜在变量作为颜色
        dimension=3,
        title="CEBRA Embeddings",
        save_path=os.path.join(config.output_dir, 'synthetic', 'cebra_embeddings.png'),
        show=False
    )
    
    # 比较不同方法
    print("比较不同方法...")
    results = compare_methods(data, config)
    
    # 可视化比较结果
    print("可视化比较结果...")
    visualize_results(
        results=results,
        metric='r2_score',
        title="不同方法的重构分数(R²)比较",
        save_path=os.path.join(config.output_dir, 'synthetic', 'method_comparison.png'),
        show=False
    )
    
    print("实验完成！")


if __name__ == "__main__":
    main()
