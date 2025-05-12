#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""神经网络模型和用于训练CEBRA-Ethan模型的损失函数。"""

import abc
from typing import Tuple, Optional, List, Dict, Any, Union

import torch
import torch.nn.functional as F
from torch import nn

# 导入数据类型
from dataclasses import dataclass

@dataclass
class Offset:
    """表示模型感受野引起的输入和输出序列之间的偏移。

    Attributes:
        left: 左侧偏移量
        right: 右侧偏移量
    """
    left: int
    right: int

    def __len__(self) -> int:
        """返回偏移的总长度。"""
        return self.left + self.right


class Model(nn.Module):
    """CEBRA-Ethan实验的基础模型。

    该模型是一个PyTorch的nn.Module。特征可以通过调用forward()或__call__方法计算。
    这个类不应该直接实例化，而应该作为CEBRA-Ethan模型的基类。

    Args:
        num_input: 输入维度数。传递给forward方法的张量将具有形状(batch, num_input, in_time)。
        num_output: 输出维度数。forward方法返回的张量将具有形状(batch, num_output, out_time)。
        offset: 由于网络的感受野而导致信号左右偏移的规范。偏移指定了输入和输出时间之间的关系，
               in_time - out_time = len(offset)。

    Attributes:
        num_input: 输入信号的输入维度。调用forward时，这是输入参数的预期维度。
                  在CEBRA-Ethan的典型应用中，输入维度对应于神经数据分析的神经元数量，
                  运动学分析的关键点数量，或者在将数据馈送到模型之前进行预处理的情况下，
                  也可以是特征空间的维度。
        num_output: 嵌入空间的输出维度。这是forward返回的值的特征维度。
                   请注意，对于使用归一化的模型，输出维度应至少为3D，
                   而不使用归一化的模型应至少为2D，以学习有意义的嵌入。
                   输出维度通常小于num_input，但这不是强制的。
    """

    def __init__(
        self,
        *,
        num_input: int,
        num_output: int,
        offset: Optional[Offset] = None,
    ):
        super().__init__()
        if num_input < 1:
            raise ValueError(
                f"输入维度至少需要为1，但得到了{num_input}。")
        if num_output < 1:
            raise ValueError(
                f"输出维度至少需要为1，但得到了{num_output}。"
            )
        self.num_input: int = num_input
        self.num_output: int = num_output

    @abc.abstractmethod
    def get_offset(self) -> Offset:
        """由感受野引起的输入和输出序列之间的偏移。

        偏移指定了输入和输出时间序列长度之间的关系。输出序列比输入序列短len(offset)步。
        对于形状为(*, *, len(offset))的输入序列，模型应返回一个丢弃最后一个维度的输出序列。

        Returns:
            网络的偏移。
        """
        raise NotImplementedError()


class _Norm(nn.Module):
    """将输入归一化到单位超球面上。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将输入归一化到单位超球面上。

        Args:
            x: 输入张量，形状为(batch_size, dim, time)

        Returns:
            归一化后的张量，形状为(batch_size, dim, time)
        """
        return F.normalize(x, p=2, dim=1)


class Squeeze(nn.Module):
    """压缩最后一个维度（如果它是1）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """压缩最后一个维度（如果它是1）。

        Args:
            x: 输入张量，形状为(batch_size, dim, time)

        Returns:
            如果time=1，则返回形状为(batch_size, dim)的张量，
            否则返回形状为(batch_size, dim, time)的张量
        """
        if x.shape[-1] == 1:
            return x.squeeze(-1)
        return x


class _Skip(nn.Module):
    """实现残差连接的跳跃连接模块。

    Args:
        *layers: 要应用的层
        crop: 裁剪输入的范围，默认为None
    """

    def __init__(self, *layers, crop=None):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.crop = crop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用跳跃连接。

        Args:
            x: 输入张量

        Returns:
            应用跳跃连接后的张量
        """
        y = self.layers(x)
        if self.crop is not None:
            start, end = self.crop
            if end is None:
                x = x[..., start:]
            else:
                x = x[..., start:end]
        return x + y


class _OffsetModel(Model):
    """基本的偏移模型实现。

    Args:
        *layers: 模型的层
        num_input: 输入维度
        num_output: 输出维度
        normalize: 是否对输出进行归一化
    """

    def __init__(self,
                 *layers,
                 num_input=None,
                 num_output=None,
                 normalize=True):
        super().__init__(num_input=num_input, num_output=num_output)

        if normalize:
            layers += (_Norm(),)
        layers += (Squeeze(),)
        self.net = nn.Sequential(*layers)
        self.normalize = normalize

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """计算给定输入信号的嵌入。

        Args:
            inp: 输入张量，预期形状为(batch_size, channels, time)

        Returns:
            形状为(batch_size, num_output, time - receptive field)的输出张量，
            或者如果time=1，则为(batch_size, num_output)

        根据初始化时使用的参数，输出嵌入可能被归一化到超球面上(normalize = True)。
        """
        # 检查输入维度并调整
        if len(inp.shape) == 2:  # [batch_size, features]
            # 添加时间维度
            inp = inp.unsqueeze(-1)  # 变为 [batch_size, features, 1]
        elif len(inp.shape) == 3:
            # 如果第二个维度不是通道数，则可能需要转置
            if inp.shape[1] != self.num_input and inp.shape[2] == self.num_input:
                # 从 [batch, time, channels] 变为 [batch, channels, time]
                inp = inp.transpose(1, 2)

        return self.net(inp)


class Offset10Model(_OffsetModel):
    """具有10个样本感受野的CEBRA-Ethan模型。

    Args:
        num_neurons: 输入神经元数量
        num_units: 隐藏单元数量
        num_output: 输出维度
        normalize: 是否对输出进行归一化
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"隐藏维度至少需要为1，但得到了{num_units}。"
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            _Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            _Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            _Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> Offset:
        """获取模型的偏移量。"""
        return Offset(5, 5)


class Offset5Model(_OffsetModel):
    """具有5个样本感受野和输出归一化的CEBRA-Ethan模型。

    Args:
        num_neurons: 输入神经元数量
        num_units: 隐藏单元数量
        num_output: 输出维度
        normalize: 是否对输出进行归一化
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            _Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 2),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> Offset:
        """获取模型的偏移量。"""
        return Offset(2, 3)


class Offset1Model(_OffsetModel):
    """具有单个样本感受野和输出归一化的CEBRA-Ethan模型。

    Args:
        num_neurons: 输入神经元数量
        num_units: 隐藏单元数量
        num_output: 输出维度
        normalize: 是否对输出进行归一化
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 2:
            raise ValueError(
                f"隐藏单元数量至少需要为2，但得到了{num_units}。"
            )
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_units,
            ),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, int(num_units // 2)),
            nn.GELU(),
            nn.Linear(int(num_units // 2), num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> Offset:
        """获取模型的偏移量。"""
        return Offset(0, 1)


# 改进版本：添加注意力机制的模型
class AttentionBlock(nn.Module):
    """自注意力模块，用于捕获特征之间的关系。

    Args:
        dim: 输入特征维度
        num_heads: 注意力头的数量
        dropout: Dropout概率
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """应用自注意力机制。

        Args:
            x: 输入张量，形状为(batch_size, seq_len, dim)

        Returns:
            应用自注意力后的张量
        """
        # 转置以适应注意力层的输入格式
        if len(x.shape) == 3 and x.shape[1] != x.shape[2]:  # (batch, dim, time)
            x = x.transpose(1, 2)  # 变为 (batch, time, dim)

        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)

        return x


class AttentionOffsetModel(_OffsetModel):
    """带有注意力机制的CEBRA-Ethan模型。

    这个模型在标准CEBRA模型的基础上添加了自注意力机制，以更好地捕获特征之间的关系。

    Args:
        num_neurons: 输入神经元数量
        num_units: 隐藏单元数量
        num_output: 输出维度
        num_heads: 注意力头的数量
        normalize: 是否对输出进行归一化
    """

    def __init__(self, num_neurons, num_units, num_output, num_heads=4, normalize=True):
        if num_units < 2:
            raise ValueError(
                f"隐藏单元数量至少需要为2，但得到了{num_units}。"
            )

        # 创建基本的MLP层
        layers = [
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(num_neurons, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
        ]

        # 添加注意力层
        self.attention = AttentionBlock(num_units, num_heads=num_heads)

        # 添加输出层
        output_layers = [
            nn.Linear(num_units, num_output),
        ]

        super().__init__(
            *layers,
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

        # 重新定义网络，包含注意力层
        if normalize:
            output_layers.append(_Norm())
        output_layers.append(Squeeze())

        self.output_layers = nn.Sequential(*output_layers)

    def forward(self, inp):
        """计算给定输入信号的嵌入。

        Args:
            inp: 输入张量，预期形状为(batch_size, channels, time)

        Returns:
            形状为(batch_size, num_output)的输出张量
        """
        # 检查输入维度并调整
        if len(inp.shape) == 2:  # [batch_size, features]
            # 添加时间维度
            inp = inp.unsqueeze(-1)  # 变为 [batch_size, features, 1]
        elif len(inp.shape) == 3:
            # 如果第二个维度不是通道数，则可能需要转置
            if inp.shape[1] != self.num_input and inp.shape[2] == self.num_input:
                # 从 [batch, time, channels] 变为 [batch, channels, time]
                inp = inp.transpose(1, 2)

        x = self.net(inp)
        x = self.attention(x)
        x = self.output_layers(x)
        return x

    def get_offset(self) -> Offset:
        """获取模型的偏移量。"""
        return Offset(0, 1)
