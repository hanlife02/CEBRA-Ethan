"""
工具函数，用于数据处理和评估。
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def set_seed(seed: int = 42):
    """设置随机种子以确保可重复性。
    
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_neural_data(neural_data: np.ndarray, preprocessing_type: str = 'zscore') -> np.ndarray:
    """使用与论文相同的方法预处理神经数据。
    
    Args:
        neural_data: 原始神经数据，形状为(time_steps, neurons)
        preprocessing_type: 预处理类型，可以是'zscore'、'minmax'或'none'
        
    Returns:
        预处理后的神经数据
    """
    if preprocessing_type == 'zscore':
        # Z-score标准化，与论文一致
        mean = np.mean(neural_data, axis=0, keepdims=True)
        std = np.std(neural_data, axis=0, keepdims=True)
        return (neural_data - mean) / (std + 1e-8)
    
    elif preprocessing_type == 'minmax':
        # Min-max标准化
        min_val = np.min(neural_data, axis=0, keepdims=True)
        max_val = np.max(neural_data, axis=0, keepdims=True)
        return (neural_data - min_val) / (max_val - min_val + 1e-8)
    
    elif preprocessing_type == 'none':
        return neural_data
    
    else:
        raise ValueError(f"不支持的预处理类型: {preprocessing_type}")


def generate_synthetic_data(
    num_samples: int = 10000,
    num_neurons: int = 100,
    num_latent: int = 3,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成神经数据用于实验。
    
    Args:
        num_samples: 样本数量
        num_neurons: 神经元数量
        num_latent: 潜在变量维度
        noise_level: 噪声水平
        random_seed: 随机种子
        
    Returns:
        神经数据和潜在变量
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # 生成潜在变量
    latent = np.random.randn(num_samples, num_latent)
    
    # 生成随机投影矩阵
    projection = np.random.randn(num_latent, num_neurons)
    
    # 生成神经数据
    neural_data = np.dot(latent, projection)
    
    # 添加噪声
    neural_data += np.random.randn(num_samples, num_neurons) * noise_level
    
    return neural_data, latent


def evaluate_embeddings(
    embeddings: np.ndarray,
    target_data: np.ndarray,
    task_type: str = 'regression',
    k: int = 5,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, float]:
    """评估嵌入质量。
    
    Args:
        embeddings: 模型生成的嵌入
        target_data: 目标数据，用于评估
        task_type: 评估任务类型，可以是'regression'、'classification'或'cross_validated'
        k: KNN的邻居数
        cv_folds: 交叉验证的折数
        test_size: 测试集比例
        random_seed: 随机种子
        
    Returns:
        评估指标
    """
    np.random.seed(random_seed)
    
    if task_type == 'regression':
        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, target_data, test_size=test_size, random_state=random_seed
        )
        
        # 使用岭回归，与论文一致
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # 计算MAE
        mae = np.mean(np.abs(y_test - y_pred))
        
        return {'r2_score': r2, 'mae': mae}
    
    elif task_type == 'classification':
        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, target_data, test_size=test_size, random_state=random_seed
        )
        
        # 使用K近邻分类器，与论文一致
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {'accuracy': accuracy}
    
    elif task_type == 'cross_validated':
        # 使用交叉验证
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        r2_scores = []
        mae_scores = []
        
        for train_idx, test_idx in kf.split(embeddings):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = target_data[train_idx], target_data[test_idx]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            r2_scores.append(r2)
            mae_scores.append(mae)
            
        return {
            'mean_r2_score': np.mean(r2_scores),
            'std_r2_score': np.std(r2_scores),
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores)
        }
    
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")


def evaluate_consistency(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = 'r2',
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, float]:
    """评估两组嵌入之间的一致性。
    
    Args:
        embeddings1: 第一组嵌入
        embeddings2: 第二组嵌入
        metric: 一致性度量，可以是'r2'或'correlation'
        test_size: 测试集比例
        random_seed: 随机种子
        
    Returns:
        一致性指标
    """
    np.random.seed(random_seed)
    
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings1, embeddings2, test_size=test_size, random_state=random_seed
    )
    
    if metric == 'r2':
        # 使用线性回归评估一致性
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return {'r2_score': r2}
    
    elif metric == 'correlation':
        # 使用相关系数评估一致性
        corr = np.mean([np.corrcoef(X_test[:, i], y_test[:, i])[0, 1] for i in range(X_test.shape[1])])
        
        return {'correlation': corr}
    
    else:
        raise ValueError(f"不支持的一致性度量: {metric}")


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    dimension: int = 3,
    title: str = "CEBRA Embeddings",
    save_path: Optional[str] = None,
    show: bool = True,
    cmap: str = 'viridis',
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (10, 8)
):
    """可视化嵌入。
    
    Args:
        embeddings: 嵌入，形状为(n_samples, n_dimensions)
        labels: 标签，用于着色，形状为(n_samples,)
        dimension: 可视化维度，可以是2或3
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
        cmap: 颜色映射
        alpha: 透明度
        figsize: 图表大小
    """
    if embeddings.shape[1] > dimension:
        # 如果嵌入维度大于可视化维度，使用PCA降维
        pca = PCA(n_components=dimension)
        embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    
    if dimension == 2:
        if labels is not None:
            scatter = plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap=cmap,
                alpha=alpha
            )
            plt.colorbar(scatter, label='Labels')
        else:
            plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                alpha=alpha
            )
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
    elif dimension == 3:
        ax = plt.figure(figsize=figsize).add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                embeddings[:, 2],
                c=labels,
                cmap=cmap,
                alpha=alpha
            )
            plt.colorbar(scatter, label='Labels')
        else:
            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                embeddings[:, 2],
                alpha=alpha
            )
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
    
    plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
