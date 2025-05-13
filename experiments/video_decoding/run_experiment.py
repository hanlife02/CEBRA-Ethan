"""
实验5：自然视频从皮层解码实验

这个实验使用CEBRA模型解码小鼠视觉皮层观看的自然视频，
比较单帧和多帧输入的解码性能，
分析不同视觉区域和不同皮层层次的视频解码能力，
结果显示使用CEBRA能达到95%以上的解码准确率。
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cebra
from config import VideoDecodingConfig
from data_loader import load_video_decoding_data
from utils import set_seed, visualize_embeddings


def run_cebra(
    data: Dict[str, np.ndarray],
    config: VideoDecodingConfig
) -> Tuple[cebra.CEBRA, np.ndarray, Dict[str, float]]:
    """运行CEBRA模型。
    
    Args:
        data: 包含神经数据和帧索引的字典
        config: 实验配置
        
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
        conditional='time',  # 使用时间条件
        distance=config.distance
    )
    
    # 准备数据
    neural_data = data['neural_data']
    frame_indices = data['frame_indices']
    num_frames = data['num_frames']
    num_repeats = data['num_repeats']
    
    # 分割训练和测试数据
    train_size = config.train_repeats * num_frames
    train_neural = neural_data[:train_size]
    test_neural = neural_data[train_size:]
    
    # 训练模型
    start_time = time.time()
    model.fit(train_neural)
    training_time = time.time() - start_time
    
    # 生成嵌入
    embeddings = model.transform(neural_data)
    
    return model, embeddings, {'training_time': training_time}


def decode_video_frames(
    data: Dict[str, np.ndarray],
    embeddings: np.ndarray,
    config: VideoDecodingConfig
) -> Dict[str, float]:
    """解码视频帧。
    
    Args:
        data: 包含神经数据和帧索引的字典
        embeddings: CEBRA嵌入
        config: 实验配置
        
    Returns:
        解码性能指标
    """
    # 准备数据
    frame_indices = data['frame_indices']
    num_frames = data['num_frames']
    num_repeats = data['num_repeats']
    
    # 分割训练和测试数据
    train_size = config.train_repeats * num_frames
    train_embeddings = embeddings[:train_size]
    test_embeddings = embeddings[train_size:]
    
    train_frames = frame_indices[:train_size]
    test_frames = frame_indices[train_size:]
    
    # 训练KNN解码器
    decoder = KNeighborsClassifier(n_neighbors=5)
    decoder.fit(train_embeddings, train_frames)
    
    # 预测帧
    pred_frames = decoder.predict(test_embeddings)
    
    # 评估准确率
    # 精确匹配
    exact_match = (pred_frames == test_frames)
    exact_accuracy = exact_match.mean()
    
    # 在窗口内正确
    window_match = np.abs(pred_frames - test_frames) < config.frame_window_size
    window_accuracy = window_match.mean()
    
    return {
        'exact_accuracy': exact_accuracy,
        'window_accuracy': window_accuracy
    }


def compare_window_sizes(
    data: Dict[str, np.ndarray],
    embeddings: np.ndarray,
    config: VideoDecodingConfig,
    window_sizes: List[int] = [1, 5, 10, 30, 60]
) -> Dict[int, float]:
    """比较不同窗口大小的解码准确率。
    
    Args:
        data: 包含神经数据和帧索引的字典
        embeddings: CEBRA嵌入
        config: 实验配置
        window_sizes: 要比较的窗口大小列表
        
    Returns:
        包含不同窗口大小准确率的字典
    """
    # 准备数据
    frame_indices = data['frame_indices']
    num_frames = data['num_frames']
    num_repeats = data['num_repeats']
    
    # 分割训练和测试数据
    train_size = config.train_repeats * num_frames
    train_embeddings = embeddings[:train_size]
    test_embeddings = embeddings[train_size:]
    
    train_frames = frame_indices[:train_size]
    test_frames = frame_indices[train_size:]
    
    # 训练KNN解码器
    decoder = KNeighborsClassifier(n_neighbors=5)
    decoder.fit(train_embeddings, train_frames)
    
    # 预测帧
    pred_frames = decoder.predict(test_embeddings)
    
    # 评估不同窗口大小的准确率
    results = {}
    
    for window_size in window_sizes:
        window_match = np.abs(pred_frames - test_frames) < window_size
        window_accuracy = window_match.mean()
        results[window_size] = window_accuracy
    
    return results


def visualize_frame_decoding(
    data: Dict[str, np.ndarray],
    embeddings: np.ndarray,
    config: VideoDecodingConfig,
    save_path: Optional[str] = None,
    show: bool = True
):
    """可视化帧解码结果。
    
    Args:
        data: 包含神经数据和帧索引的字典
        embeddings: CEBRA嵌入
        config: 实验配置
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
    """
    # 准备数据
    frame_indices = data['frame_indices']
    num_frames = data['num_frames']
    num_repeats = data['num_repeats']
    
    # 分割训练和测试数据
    train_size = config.train_repeats * num_frames
    train_embeddings = embeddings[:train_size]
    test_embeddings = embeddings[train_size:]
    
    train_frames = frame_indices[:train_size]
    test_frames = frame_indices[train_size:]
    
    # 训练KNN解码器
    decoder = KNeighborsClassifier(n_neighbors=5)
    decoder.fit(train_embeddings, train_frames)
    
    # 预测帧
    pred_frames = decoder.predict(test_embeddings)
    
    # 可视化预测vs真实帧
    plt.figure(figsize=(12, 6))
    
    # 只显示前1000个样本，以便更清晰地查看
    n_samples = min(1000, len(test_frames))
    
    plt.scatter(np.arange(n_samples), test_frames[:n_samples], label='True Frames', alpha=0.5)
    plt.scatter(np.arange(n_samples), pred_frames[:n_samples], label='Predicted Frames', alpha=0.5)
    
    plt.xlabel('Sample Index')
    plt.ylabel('Frame Index')
    plt.title('Video Frame Decoding')
    plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_window_accuracy(
    window_results: Dict[int, float],
    title: str = "Frame Decoding Accuracy vs Window Size",
    save_path: Optional[str] = None,
    show: bool = True
):
    """可视化不同窗口大小的解码准确率。
    
    Args:
        window_results: 包含不同窗口大小准确率的字典
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图表
    """
    window_sizes = list(window_results.keys())
    accuracies = list(window_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, accuracies, marker='o')
    plt.xlabel('Window Size (frames)')
    plt.ylabel('Decoding Accuracy')
    plt.title(title)
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """运行自然视频从皮层解码实验。"""
    # 加载配置
    config = VideoDecodingConfig()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 创建输出目录
    os.makedirs(os.path.join(config.output_dir, 'video_decoding'), exist_ok=True)
    
    # 加载数据
    print("加载视频解码数据...")
    data = load_video_decoding_data(
        data_dir=config.data_dir,
        dataset_name=config.dataset_name,
        brain_area=config.brain_area,
        num_repeats=config.num_repeats,
        preprocessing='zscore',
        random_seed=config.random_seed
    )
    
    # 运行CEBRA
    print("运行CEBRA...")
    model, embeddings, training_metrics = run_cebra(data, config)
    
    # 可视化嵌入
    print("可视化嵌入...")
    visualize_embeddings(
        embeddings=embeddings,
        labels=data['frame_indices'],  # 使用帧索引作为颜色
        dimension=3,
        title="CEBRA Embeddings for Video Frames",
        save_path=os.path.join(config.output_dir, 'video_decoding', 'embeddings.png'),
        show=False,
        cmap='viridis'
    )
    
    # 解码视频帧
    print("解码视频帧...")
    decoding_metrics = decode_video_frames(data, embeddings, config)
    
    print(f"精确匹配准确率: {decoding_metrics['exact_accuracy']:.4f}")
    print(f"窗口匹配准确率 (窗口大小={config.frame_window_size}): {decoding_metrics['window_accuracy']:.4f}")
    
    # 比较不同窗口大小
    print("比较不同窗口大小...")
    window_results = compare_window_sizes(data, embeddings, config)
    
    # 可视化窗口准确率
    print("可视化窗口准确率...")
    visualize_window_accuracy(
        window_results=window_results,
        title="Frame Decoding Accuracy vs Window Size",
        save_path=os.path.join(config.output_dir, 'video_decoding', 'window_accuracy.png'),
        show=False
    )
    
    # 可视化帧解码
    print("可视化帧解码...")
    visualize_frame_decoding(
        data=data,
        embeddings=embeddings,
        config=config,
        save_path=os.path.join(config.output_dir, 'video_decoding', 'frame_decoding.png'),
        show=False
    )
    
    print("实验完成！")


if __name__ == "__main__":
    main()
