#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""用于对比学习的损失函数

CEBRA-Ethan可以使用不同的损失函数来学习嵌入。实现广义InfoNCE度量的损失函数的通用接口由BaseInfoNCE类提供。

提供了具有固定和可学习温度以及不同相似性度量的损失函数。

请注意，损失函数可以有可训练的参数，这些参数由Solver类中实现的训练循环自动处理。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@torch.jit.script
def dot_similarity(ref: torch.Tensor, pos: torch.Tensor,
                   neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算参考样本、正样本和负样本之间的余弦相似度
    
    Args:
        ref: 形状为(n, d)的参考样本
        pos: 形状为(n, d)的正样本
        neg: 形状为(n, d)的负样本
        
    Returns:
        形状为(n,)的参考样本和正样本之间的相似度，以及
        形状为(n, n)的参考样本和负样本之间的相似度
    """
    pos_dist = torch.einsum("ni,ni->n", ref, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref, neg)
    return pos_dist, neg_dist


@torch.jit.script
def euclidean_similarity(
        ref: torch.Tensor, pos: torch.Tensor,
        neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算参考样本、正样本和负样本之间的负L2距离
    
    Args:
        ref: 形状为(n, d)的参考样本
        pos: 形状为(n, d)的正样本
        neg: 形状为(n, d)的负样本
        
    Returns:
        形状为(n,)的参考样本和正样本之间的相似度，以及
        形状为(n, n)的参考样本和负样本之间的相似度
    """
    ref_sq = torch.einsum("ni->n", ref**2)
    pos_sq = torch.einsum("ni->n", pos**2)
    neg_sq = torch.einsum("ni->n", neg**2)
    
    pos_cosine, neg_cosine = dot_similarity(ref, pos, neg)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)
    
    return pos_dist, neg_dist


@torch.jit.script
def infonce(
        pos_dist: torch.Tensor, neg_dist: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """InfoNCE实现
    
    注意:
        - 此函数的行为从CEBRA 0.3.0开始发生了变化。
          InfoNCE实现在数值上更加稳定。
    """
    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()
    
    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c
    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    
    c_mean = c.mean()
    align_corrected = align - c_mean
    uniform_corrected = uniform + c_mean
    
    return align + uniform, align_corrected, uniform_corrected


class ContrastiveLoss(nn.Module):
    """对比损失的基类。"""
    
    def forward(
            self, ref: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算对比损失。
        
        Args:
            ref: 形状为(n, d)的参考样本
            pos: 形状为(n, d)的正样本
            neg: 形状为(n, d)的负样本
        """
        raise NotImplementedError()


class BaseInfoNCE(ContrastiveLoss):
    """所有InfoNCE损失的基类。
    
    给定一个由该类的子类实现的相似性度量φ，广义InfoNCE损失计算为：
    
    sum_{i=1}^n - φ(x_i, y^{+}_i) + log sum_{j=1}^{n} e^{φ(x_i, y^{-}_{ij})}
    
    其中n是批量大小，x是参考样本(ref)，y^{+}是正样本(pos)，y^{-}是负样本(neg)。
    """
    
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor]:
        """相似性度量。
        
        Args:
            ref: 形状为(n, d)的参考样本
            pos: 形状为(n, d)的正样本
            neg: 形状为(n, d)的负样本
            
        Returns:
            形状为(n,)的参考样本和正样本之间的距离，以及
            形状为(n, n)的参考样本和负样本之间的距离
        """
        raise NotImplementedError()
        
    def forward(self, ref, pos,
                neg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算InfoNCE损失。
        
        Args:
            ref: 形状为(n, d)的参考样本
            pos: 形状为(n, d)的正样本
            neg: 形状为(n, d)的负样本
        """
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        return infonce(pos_dist, neg_dist)


class FixedInfoNCE(BaseInfoNCE):
    """具有固定温度的InfoNCE基础损失。
    
    Attributes:
        temperature: softmax温度
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature


class LearnableInfoNCE(BaseInfoNCE):
    """具有可学习温度的InfoNCE基础损失。
    
    Attributes:
        temperature: 可学习温度参数的当前值
        min_temperature: 要使用的最小温度。如果在优化过程中遇到数值问题，请增加最小温度
    """
    
    def __init__(self,
                 temperature: float = 1.0,
                 min_temperature: Optional[float] = None):
        super().__init__()
        if min_temperature is None:
            self.max_inverse_temperature = math.inf
        else:
            self.max_inverse_temperature = 1.0 / min_temperature
        log_inverse_temperature = torch.tensor(
            math.log(1.0 / float(temperature)))
        self.log_inverse_temperature = nn.Parameter(log_inverse_temperature)
        self.min_temperature = min_temperature
        
    @torch.jit.export
    def _prepare_inverse_temperature(self) -> torch.Tensor:
        """计算当前的逆温度。"""
        inverse_temperature = torch.exp(self.log_inverse_temperature)
        inverse_temperature = torch.clamp(inverse_temperature,
                                          max=self.max_inverse_temperature)
        return inverse_temperature
        
    @property
    def temperature(self) -> float:
        with torch.no_grad():
            return 1.0 / self._prepare_inverse_temperature().item()


class FixedCosineInfoNCE(FixedInfoNCE):
    """具有固定温度的余弦相似度函数。
    
    相似性度量为：
    
    φ(x, y) = x^T y / τ
    
    其中τ > 0是固定温度。
    
    注意，此损失函数通常只应与归一化一起使用。
    此类本身不执行任何检查。确保x和y已归一化。
    """
    
    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = dot_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature


class FixedEuclideanInfoNCE(FixedInfoNCE):
    """具有固定温度的L2相似度函数。
    
    相似性度量为：
    
    φ(x, y) = -||x - y|| / τ
    
    其中τ > 0是固定温度。
    """
    
    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = euclidean_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature


class LearnableCosineInfoNCE(LearnableInfoNCE):
    """具有可学习温度的余弦相似度函数。
    
    与FixedCosineInfoNCE类似，但具有可学习的温度参数τ。
    """
    
    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inverse_temperature = self._prepare_inverse_temperature()
        pos, neg = dot_similarity(ref, pos, neg)
        return pos * inverse_temperature, neg * inverse_temperature


class LearnableEuclideanInfoNCE(LearnableInfoNCE):
    """具有可学习温度的L2相似度函数。
    
    与FixedEuclideanInfoNCE类似，但具有可学习的温度参数τ。
    """
    
    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inverse_temperature = self._prepare_inverse_temperature()
        pos, neg = euclidean_similarity(ref, pos, neg)
        return pos * inverse_temperature, neg * inverse_temperature


# 为了兼容性的别名
InfoNCE = FixedCosineInfoNCE
InfoMSE = FixedEuclideanInfoNCE


# 改进版本：添加带有焦点损失的InfoNCE
class FocalInfoNCE(FixedCosineInfoNCE):
    """带有焦点损失的InfoNCE。
    
    这个版本的InfoNCE使用焦点损失来关注更难区分的样本对。
    
    Args:
        temperature: softmax温度
        gamma: 焦点损失的聚焦参数，较高的值会更加关注难以区分的样本
    """
    
    def __init__(self, temperature: float = 1.0, gamma: float = 2.0):
        super().__init__(temperature)
        self.gamma = gamma
        
    def forward(self, ref, pos, neg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算带有焦点损失的InfoNCE损失。
        
        Args:
            ref: 形状为(n, d)的参考样本
            pos: 形状为(n, d)的正样本
            neg: 形状为(n, d)的负样本
        """
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        
        # 计算标准InfoNCE损失
        with torch.no_grad():
            c, _ = neg_dist.max(dim=1, keepdim=True)
        c = c.detach()
        
        pos_dist = pos_dist - c.squeeze(1)
        neg_dist = neg_dist - c
        
        # 计算正样本的概率
        pos_exp = torch.exp(pos_dist)
        neg_exp_sum = torch.sum(torch.exp(neg_dist), dim=1)
        p_pos = pos_exp / (pos_exp + neg_exp_sum)
        
        # 应用焦点损失权重
        focal_weight = (1 - p_pos) ** self.gamma
        
        # 计算加权的对齐和均匀损失
        align = (-focal_weight * pos_dist).mean()
        uniform = (focal_weight * torch.logsumexp(neg_dist, dim=1)).mean()
        
        c_mean = c.mean()
        align_corrected = align - c_mean
        uniform_corrected = uniform + c_mean
        
        return align + uniform, align_corrected, uniform_corrected
