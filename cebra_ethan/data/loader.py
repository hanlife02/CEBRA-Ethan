#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""数据加载器，用于批量加载数据进行训练。"""

from typing import Optional, Tuple, Union, List, Dict, Any

import torch
from torch.utils.data import DataLoader

from cebra_ethan.solver.base import Batch


class ContrastiveLoader:
    """用于对比学习的数据加载器。
    
    这个加载器从数据集中生成批次，每个批次包含参考样本、正样本和负样本。
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否在每个epoch开始时打乱数据
        num_workers: 用于数据加载的子进程数
        drop_last: 如果为True，则丢弃最后一个不完整的批次
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
    ):
        """初始化对比加载器。
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否在每个epoch开始时打乱数据
            num_workers: 用于数据加载的子进程数
            drop_last: 如果为True，则丢弃最后一个不完整的批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        # 创建PyTorch DataLoader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        
    def __iter__(self):
        """迭代批次。
        
        Yields:
            Batch对象，包含参考样本、正样本和负样本
        """
        for batch in self.loader:
            if len(batch) == 3:  # 标准对比数据集
                reference, positive, negative = batch
                yield Batch(reference, positive, negative)
            elif len(batch) == 4:  # 带有行为数据的数据集
                reference, positive, negative, behavior = batch
                # 在这里，我们忽略行为数据，只使用神经数据
                yield Batch(reference, positive, negative)
            else:
                raise ValueError(f"不支持的批次格式：{len(batch)}元素")
                
    def __len__(self) -> int:
        """返回批次数量。"""
        return len(self.loader)
        
    @property
    def device(self) -> torch.device:
        """返回数据集的设备。"""
        return self.dataset.device


class MultiObjectiveLoader(ContrastiveLoader):
    """用于多目标学习的数据加载器。
    
    这个加载器生成两种类型的批次：一种用于时间对比学习，一种用于行为对比学习。
    
    Args:
        dataset: 数据集，必须是BehavioralDataset
        batch_size: 批次大小
        shuffle: 是否在每个epoch开始时打乱数据
        num_workers: 用于数据加载的子进程数
        drop_last: 如果为True，则丢弃最后一个不完整的批次
        behavior_similarity_threshold: 行为相似性阈值，用于确定正样本
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        behavior_similarity_threshold: float = 0.1,
    ):
        """初始化多目标加载器。
        
        Args:
            dataset: 数据集，必须是BehavioralDataset
            batch_size: 批次大小
            shuffle: 是否在每个epoch开始时打乱数据
            num_workers: 用于数据加载的子进程数
            drop_last: 如果为True，则丢弃最后一个不完整的批次
            behavior_similarity_threshold: 行为相似性阈值，用于确定正样本
        """
        super().__init__(dataset, batch_size, shuffle, num_workers, drop_last)
        self.behavior_similarity_threshold = behavior_similarity_threshold
        
    def __iter__(self):
        """迭代批次，生成时间对比和行为对比的批次。
        
        Yields:
            (时间对比批次, 行为对比批次)的元组
        """
        for batch in self.loader:
            if len(batch) == 4:  # 带有行为数据的数据集
                reference, positive_time, negative, behavior = batch
                
                # 时间对比批次
                time_batch = Batch(reference, positive_time, negative)
                
                # 为行为对比找到正样本
                # 这里我们使用一个简单的启发式方法：在批次内找到行为最相似的样本
                behavior_distances = torch.cdist(behavior.unsqueeze(1), behavior.unsqueeze(0)).squeeze(1)
                # 将自身的距离设为无穷大，避免选择自身
                behavior_distances.fill_diagonal_(float('inf'))
                # 找到最相似的样本
                _, positive_indices = torch.topk(behavior_distances, k=1, largest=False, dim=1)
                positive_behavior = reference[positive_indices.squeeze()]
                
                # 行为对比批次
                behavior_batch = Batch(reference, positive_behavior, negative)
                
                yield time_batch, behavior_batch
            else:
                raise ValueError("多目标加载器需要带有行为数据的数据集")


# 改进版本：添加带有数据增强的加载器
class AugmentedContrastiveLoader(ContrastiveLoader):
    """带有数据增强的对比学习加载器。
    
    这个加载器在生成批次时应用数据增强，以提高模型的鲁棒性。
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否在每个epoch开始时打乱数据
        num_workers: 用于数据加载的子进程数
        drop_last: 如果为True，则丢弃最后一个不完整的批次
        noise_level: 添加到样本的高斯噪声的标准差
        dropout_prob: 随机将特征设为零的概率
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        noise_level: float = 0.1,
        dropout_prob: float = 0.1,
    ):
        """初始化带有数据增强的对比加载器。
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否在每个epoch开始时打乱数据
            num_workers: 用于数据加载的子进程数
            drop_last: 如果为True，则丢弃最后一个不完整的批次
            noise_level: 添加到样本的高斯噪声的标准差
            dropout_prob: 随机将特征设为零的概率
        """
        super().__init__(dataset, batch_size, shuffle, num_workers, drop_last)
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        
    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """应用数据增强。
        
        Args:
            x: 输入张量
            
        Returns:
            增强后的张量
        """
        # 添加高斯噪声
        if self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
        # 应用特征dropout
        if self.dropout_prob > 0:
            mask = torch.rand_like(x) > self.dropout_prob
            x = x * mask
            
        return x
        
    def __iter__(self):
        """迭代批次，应用数据增强。
        
        Yields:
            Batch对象，包含增强后的参考样本、正样本和负样本
        """
        for batch in self.loader:
            if len(batch) == 3:  # 标准对比数据集
                reference, positive, negative = batch
                
                # 应用数据增强
                reference = self._apply_augmentation(reference)
                positive = self._apply_augmentation(positive)
                negative = self._apply_augmentation(negative)
                
                yield Batch(reference, positive, negative)
            elif len(batch) == 4:  # 带有行为数据的数据集
                reference, positive, negative, behavior = batch
                
                # 应用数据增强
                reference = self._apply_augmentation(reference)
                positive = self._apply_augmentation(positive)
                negative = self._apply_augmentation(negative)
                
                yield Batch(reference, positive, negative)
            else:
                raise ValueError(f"不支持的批次格式：{len(batch)}元素")
