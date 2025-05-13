"""
配置文件，用于设置CEBRA实验的参数。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class BaseConfig:
    """基本配置类，包含所有实验共享的参数。"""
    
    # 模型参数
    output_dimension: int = 8  # 输出嵌入的维度
    model_architecture: str = "offset10-model"  # 模型架构
    num_hidden_units: int = 256  # 隐藏层单元数
    
    # 训练参数
    batch_size: int = 512  # 批次大小
    learning_rate: float = 0.001  # 学习率
    max_iterations: int = 10000  # 最大迭代次数
    temperature: float = 1.0  # 温度参数
    temperature_mode: str = "constant"  # 温度模式：constant或auto
    
    # 数据加载参数
    time_offsets: int = 10  # 时间偏移
    conditional: str = "time_delta"  # 条件分布
    delta: float = 0.1  # delta参数，用于continuous条件
    distance: str = "cosine"  # 距离度量
    
    # 其他参数
    device: str = "cuda_if_available"  # 设备
    verbose: bool = True  # 是否显示进度条
    random_seed: int = 42  # 随机种子
    
    # 路径参数
    data_dir: str = "data"  # 数据目录
    output_dir: str = "results"  # 输出目录
    
    def __post_init__(self):
        """初始化后的处理。"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)


@dataclass
class SyntheticConfig(BaseConfig):
    """合成数据实验的配置。"""
    
    # 合成数据参数
    num_samples: int = 10000  # 样本数量
    num_neurons: int = 100  # 神经元数量
    num_latent: int = 3  # 潜在变量维度
    noise_level: float = 0.1  # 噪声水平
    
    # 模型参数
    output_dimension: int = 3  # 输出嵌入的维度
    model_architecture: str = "offset1-model"  # 模型架构
    
    # 评估参数
    test_size: float = 0.2  # 测试集比例


@dataclass
class HippocampusConfig(BaseConfig):
    """大鼠海马体数据实验的配置。"""
    
    # 数据参数
    dataset_name: str = "hc-11"  # 数据集名称
    bin_width_ms: int = 25  # 时间窗口宽度（毫秒）
    
    # 模型参数
    output_dimension: int = 3  # 输出嵌入的维度
    
    # 评估参数
    position_decoding_k: int = 5  # 位置解码的k值


@dataclass
class PrimateConfig(BaseConfig):
    """灵长类运动任务实验的配置。"""
    
    # 数据参数
    dataset_name: str = "000127"  # 数据集名称
    bin_width_ms: int = 1  # 时间窗口宽度（毫秒）
    gaussian_smoothing_sigma_ms: int = 40  # 高斯平滑的sigma（毫秒）
    
    # 模型参数
    output_dimension: int = 4  # 输出嵌入的维度
    
    # 评估参数
    position_decoding_k: int = 5  # 位置解码的k值
    active_passive_k: int = 5  # 主动/被动分类的k值


@dataclass
class AllenConfig(BaseConfig):
    """Allen Brain Observatory数据实验的配置。"""
    
    # 数据参数
    calcium_dataset: str = "visual_drift"  # 钙成像数据集
    neuropixels_dataset: str = "visual_coding_neuropixels"  # Neuropixels数据集
    
    # 模型参数
    output_dimension: int = 8  # 输出嵌入的维度
    
    # DINO参数
    dino_model: str = "vit/8"  # DINO模型
    dino_feature_dim: int = 768  # DINO特征维度


@dataclass
class VideoDecodingConfig(BaseConfig):
    """自然视频解码实验的配置。"""
    
    # 数据参数
    dataset_name: str = "visual_coding_neuropixels"  # 数据集名称
    brain_area: str = "V1"  # 脑区
    
    # 模型参数
    output_dimension: int = 8  # 输出嵌入的维度
    
    # 解码参数
    num_repeats: int = 10  # 重复次数
    train_repeats: int = 9  # 训练重复次数
    test_repeat: int = 1  # 测试重复次数
    frame_window_size: int = 30  # 帧窗口大小（1秒 = 30帧）


@dataclass
class MultiSessionConfig(BaseConfig):
    """多会话、多动物实验的配置。"""
    
    # 数据参数
    dataset_names: List[str] = field(default_factory=lambda: ["hc-11", "visual_coding_neuropixels"])
    
    # 模型参数
    output_dimension: int = 8  # 输出嵌入的维度
    
    # 训练参数
    joint_training: bool = True  # 是否联合训练
    
    # 评估参数
    consistency_metric: str = "r2"  # 一致性度量
