# CEBRA-Ethan

《神经网络的计算基础》24-25学年春季学期课程 期末论文复现作业

## 项目概述

CEBRA-Ethan是对[CEBRA](https://github.com/AdaptiveMotorControlLab/CEBRA)（Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables）库的个人改进版本，用于神经科学数据的自监督学习和嵌入。

本项目在原始CEBRA库的基础上进行了多项改进：

1. **模型改进**：
   - 添加了基于注意力机制的模型（AttentionOffsetModel），以更好地捕获特征之间的关系
   - 优化了模型架构，提高了训练效率和嵌入质量

2. **损失函数改进**：
   - 实现了焦点损失（Focal Loss）版本的InfoNCE，更加关注难以区分的样本对
   - 保留了原始CEBRA的所有损失函数，确保兼容性

3. **训练过程改进**：
   - 添加了早停机制，避免过拟合并减少训练时间
   - 实现了数据增强策略，提高模型的鲁棒性

4. **数据处理改进**：
   - 优化了数据加载和批处理逻辑，提高了训练效率
   - 添加了多目标学习的支持，可以同时学习时间和行为相关的嵌入

## 快速开始

以下是一个简单的使用示例：

```python
import numpy as np
import torch
from cebra_ethan.data.dataset import ContrastiveDataset
from cebra_ethan.data.loader import AugmentedContrastiveLoader
from cebra_ethan.models.model import AttentionOffsetModel
from cebra_ethan.models.criterions import LearnableCosineInfoNCE
from cebra_ethan.solver.base import EarlyStoppingSolver

# 准备数据
neural_data = np.random.randn(1000, 50)  # 示例数据
dataset = ContrastiveDataset(neural_data, time_offset=1, num_negatives=10)
loader = AugmentedContrastiveLoader(dataset, batch_size=32, noise_level=0.05)

# 创建模型和损失函数
model = AttentionOffsetModel(num_neurons=50, num_units=128, num_output=3)
criterion = LearnableCosineInfoNCE(temperature=1.0)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=1e-3
)

# 创建求解器并训练
solver = EarlyStoppingSolver(model, criterion, optimizer, patience=5)
solver.fit(loader, save_frequency=10, logdir="./checkpoints")

# 生成嵌入
embeddings = solver.transform(torch.tensor(neural_data, dtype=torch.float32))
```

更详细的示例可以在`examples`目录中找到。

## 主要组件

CEBRA-Ethan包含以下主要组件：

1. **模型（Models）**：
   - `Offset10Model`：具有10个样本感受野的模型
   - `Offset5Model`：具有5个样本感受野的模型
   - `Offset1Model`：具有单个样本感受野的模型
   - `AttentionOffsetModel`：带有注意力机制的改进模型

2. **损失函数（Criterions）**：
   - `LearnableCosineInfoNCE`：具有可学习温度的余弦相似度InfoNCE
   - `LearnableEuclideanInfoNCE`：具有可学习温度的欧几里得相似度InfoNCE
   - `FocalInfoNCE`：带有焦点损失的InfoNCE

3. **求解器（Solvers）**：
   - `SingleSessionSolver`：标准单会话求解器
   - `EarlyStoppingSolver`：带有早停机制的求解器

4. **数据处理（Data）**：
   - `NeuralDataset`：基本神经数据集
   - `ContrastiveDataset`：用于对比学习的数据集
   - `BehavioralDataset`：包含行为数据的数据集
   - `ContrastiveLoader`：标准对比学习数据加载器
   - `AugmentedContrastiveLoader`：带有数据增强的加载器
   - `MultiObjectiveLoader`：用于多目标学习的加载器

## 与原始CEBRA的比较

CEBRA-Ethan在保持原始CEBRA核心功能的同时，添加了多项改进：

| 特性 | 原始CEBRA | CEBRA-Ethan |
|------|-----------|-------------|
| 注意力机制 | ❌ | ✅ |
| 焦点损失 | ❌ | ✅ |
| 早停机制 | ❌ | ✅ |
| 数据增强 | 有限 | 增强 |
| 多目标学习 | ✅ | 改进 |
| 代码结构 | 复杂 | 简化 |

## 参考文献

- Schneider, S., Lee, J., & Mathis, M. W. (2023). Learnable latent embeddings for joint behavioural and neural analyses. Nature, 616(7958), 487-493.
- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9729-9738).
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).