#
# CEBRA-Ethan: 个人改进版的CEBRA (Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables)
# 基于原始CEBRA库 (https://github.com/AdaptiveMotorControlLab/CEBRA)
#
"""CEBRA-Ethan是CEBRA库的个人改进版本，用于估计高维记录的一致性嵌入。

该库包含使用PyTorch实现的自监督学习算法，并支持生物学和神经科学中常见的各种数据集。
相比原始CEBRA库，CEBRA-Ethan提供了更高效的实现和额外的功能。
"""

__version__ = "0.1.0"
__all__ = ["CEBRA"]

# 导入主要组件
from cebra_ethan.models import *
from cebra_ethan.solver import *
from cebra_ethan.data import *
from cebra_ethan.utils import *

# 尝试导入sklearn集成
try:
    from cebra_ethan.integrations.sklearn.cebra import CEBRA
    from cebra_ethan.integrations.sklearn.decoder import KNNDecoder
    from cebra_ethan.integrations.sklearn.decoder import L1LinearRegressor
    is_sklearn_available = True
except ImportError:
    is_sklearn_available = False
