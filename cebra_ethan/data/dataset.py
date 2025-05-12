#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""数据集类，用于加载和处理神经数据。"""

import abc
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NeuralDataset(Dataset):
    """神经数据集的基类。

    这个类提供了加载和处理神经数据的基本功能。

    Args:
        neural_data: 神经数据，形状为(time_steps, neurons)
        device: 数据应该加载到的设备
    """

    def __init__(
        self,
        neural_data: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ):
        """初始化神经数据集。

        Args:
            neural_data: 神经数据，形状为(time_steps, neurons)
            device: 数据应该加载到的设备
        """
        # 转换为torch.Tensor
        if isinstance(neural_data, np.ndarray):
            neural_data = torch.from_numpy(neural_data).float()

        self.neural = neural_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.neural = self.neural.to(self.device)

    def __len__(self) -> int:
        """返回数据集中的样本数。"""
        return len(self.neural)

    def __getitem__(self, idx: Union[int, torch.Tensor, List[int]]) -> torch.Tensor:
        """获取指定索引的样本。

        Args:
            idx: 样本索引或索引列表

        Returns:
            指定索引的样本
        """
        return self.neural[idx]


class ContrastiveDataset(NeuralDataset):
    """用于对比学习的神经数据集。

    这个数据集生成参考样本、正样本和负样本的三元组，用于对比学习。

    Args:
        neural_data: 神经数据，形状为(time_steps, neurons)
        time_offset: 用于生成正样本的时间偏移
        num_negatives: 每个参考样本的负样本数量
        device: 数据应该加载到的设备
    """

    def __init__(
        self,
        neural_data: Union[np.ndarray, torch.Tensor],
        time_offset: int = 1,
        num_negatives: int = 10,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """初始化对比数据集。

        Args:
            neural_data: 神经数据，形状为(time_steps, neurons)
            time_offset: 用于生成正样本的时间偏移
            num_negatives: 每个参考样本的负样本数量
            device: 数据应该加载到的设备
        """
        super().__init__(neural_data, device)
        self.time_offset = time_offset
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        """返回数据集中的样本数。"""
        # 考虑时间偏移，有效样本数减少
        return len(self.neural) - self.time_offset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取对比学习的三元组。

        Args:
            idx: 样本索引

        Returns:
            (参考样本, 正样本, 负样本)的元组
        """
        # 参考样本
        reference = self.neural[idx]

        # 正样本（时间上相邻的样本）
        positive = self.neural[idx + self.time_offset]

        # 负样本（随机选择一个样本）
        neg_idx = torch.randint(0, len(self.neural), (1,)).item()
        negative = self.neural[neg_idx]

        # 确保维度正确 [channels, sequence_length]
        # 对于卷积模型，需要添加序列维度
        if len(reference.shape) == 1:  # 如果是一维张量 [features]
            # 转换为 [features, 1] 形状，适合卷积模型
            reference = reference.unsqueeze(-1)  # 添加序列维度
            positive = positive.unsqueeze(-1)
            negative = negative.unsqueeze(-1)  # 添加序列维度

        return reference, positive, negative


class BehavioralDataset(ContrastiveDataset):
    """包含行为数据的神经数据集。

    这个数据集除了神经数据外，还包含行为数据，可以用于多目标学习。

    Args:
        neural_data: 神经数据，形状为(time_steps, neurons)
        behavior_data: 行为数据，形状为(time_steps, behavior_dims)
        time_offset: 用于生成正样本的时间偏移
        num_negatives: 每个参考样本的负样本数量
        device: 数据应该加载到的设备
    """

    def __init__(
        self,
        neural_data: Union[np.ndarray, torch.Tensor],
        behavior_data: Union[np.ndarray, torch.Tensor],
        time_offset: int = 1,
        num_negatives: int = 10,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """初始化行为数据集。

        Args:
            neural_data: 神经数据，形状为(time_steps, neurons)
            behavior_data: 行为数据，形状为(time_steps, behavior_dims)
            time_offset: 用于生成正样本的时间偏移
            num_negatives: 每个参考样本的负样本数量
            device: 数据应该加载到的设备
        """
        super().__init__(neural_data, time_offset, num_negatives, device)

        # 转换为torch.Tensor
        if isinstance(behavior_data, np.ndarray):
            behavior_data = torch.from_numpy(behavior_data).float()

        self.behavior = behavior_data.to(self.device)

        # 确保神经数据和行为数据的时间步数相同
        assert len(self.neural) == len(self.behavior), "神经数据和行为数据的时间步数必须相同"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取包含行为数据的对比学习四元组。

        Args:
            idx: 样本索引

        Returns:
            (参考样本, 正样本, 负样本, 行为数据)的元组
        """
        reference, positive, negatives = super().__getitem__(idx)
        behavior = self.behavior[idx]

        return reference, positive, negatives, behavior
