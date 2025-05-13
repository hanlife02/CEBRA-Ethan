# CEBRA-Ethan

《神经网络的计算基础》24-25 学年春季学期课程 期末论文复现

## 项目概述

CEBRA-Ethan 是对[CEBRA](https://github.com/AdaptiveMotorControlLab/CEBRA)（Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables）库的个人改进版本，用于神经科学数据的自监督学习和嵌入。

本项目在原始 CEBRA 库的基础上进行了多项改进，并复现了论文"Learnable latent embeddings for joint behavioural and neural analysis"中的主要实验。

### 框架改进

1. **模型改进**：

   - 添加了基于注意力机制的模型（AttentionOffsetModel），以更好地捕获特征之间的关系
   - 优化了模型架构，提高了训练效率和嵌入质量

2. **损失函数改进**：

   - 实现了焦点损失（Focal Loss）版本的 InfoNCE，更加关注难以区分的样本对
   - 保留了原始 CEBRA 的所有损失函数，确保兼容性

3. **训练过程改进**：

   - 添加了早停机制，避免过拟合并减少训练时间
   - 实现了数据增强策略，提高模型的鲁棒性

4. **数据处理改进**：
   - 优化了数据加载和批处理逻辑，提高了训练效率
   - 添加了多目标学习的支持，可以同时学习时间和行为相关的嵌入

### 论文实验复现

本项目复现了以下实验：

1. **合成数据验证实验**：使用人工合成的神经元活动数据测试 CEBRA 重构真实潜在空间的能力，并与 t-SNE、UMAP、pi-VAE 和 autoLFADS 等方法进行比较。

2. **大鼠海马体数据分析**：使用 Grosmark 与 Buzsáki 2016 年收集的大鼠线性轨道实验数据，评估不同算法生成的潜在空间在不同大鼠之间的一致性，进行假设驱动分析和发现驱动分析，使用拓扑数据分析验证嵌入的拓扑结构稳健性，比较位置解码性能。

3. **灵长类运动任务潜在动态分析**：使用灵长类体感皮质(S1)八方向中心外运动任务的电生理记录数据，分析主动和被动运动对神经群体活动的影响，测试位置、方向和主动/被动状态的解码性能。

4. **跨模态一致嵌入分析**：使用 Allen Brain Observatory 数据集的钙成像和 Neuropixels 记录，将视频特征作为"行为"标签用于训练 CEBRA 模型，验证不同记录方法是否产生类似的潜在表示，测试联合训练对跨模态一致性的影响，分析不同视觉区域内部和区域间的一致性。

5. **自然视频从皮层解码实验**：使用 CEBRA 模型解码小鼠视觉皮层观看的自然视频，比较单帧和多帧输入的解码性能，分析不同视觉区域和不同皮层层次的视频解码能力。

6. **多会话、多动物 CEBRA 训练**：联合训练跨会话和不同动物的数据，研究联合训练对嵌入一致性的提升，测试预训练模型在新动物数据上的快速适应能力。

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

CEBRA-Ethan 包含以下主要组件：

1. **模型（Models）**：

   - `Offset10Model`：具有 10 个样本感受野的模型
   - `Offset5Model`：具有 5 个样本感受野的模型
   - `Offset1Model`：具有单个样本感受野的模型
   - `AttentionOffsetModel`：带有注意力机制的改进模型

2. **损失函数（Criterions）**：

   - `LearnableCosineInfoNCE`：具有可学习温度的余弦相似度 InfoNCE
   - `LearnableEuclideanInfoNCE`：具有可学习温度的欧几里得相似度 InfoNCE
   - `FocalInfoNCE`：带有焦点损失的 InfoNCE

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

## 与原始 CEBRA 的比较

CEBRA-Ethan 在保持原始 CEBRA 核心功能的同时，添加了多项改进：

| 特性       | 原始 CEBRA | CEBRA-Ethan |
| ---------- | ---------- | ----------- |
| 注意力机制 | ❌         | ✅          |
| 焦点损失   | ❌         | ✅          |
| 早停机制   | ❌         | ✅          |
| 数据增强   | 有限       | 增强        |
| 多目标学习 | ✅         | 改进        |
| 代码结构   | 复杂       | 简化        |

## 项目结构

```
CEBRA-Ethan/
├── cebra_ethan/           # CEBRA-Ethan框架实现
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型实现
│   ├── solver/            # 求解器实现
│   └── utils/             # 工具函数
├── experiments/           # 实验实现
│   ├── synthetic/         # 合成数据验证实验
│   ├── hippocampus/       # 大鼠海马体数据分析
│   ├── primate/           # 灵长类运动任务潜在动态分析
│   ├── allen/             # 跨模态一致嵌入分析
│   ├── video_decoding/    # 自然视频从皮层解码实验
│   └── multi_session/     # 多会话、多动物CEBRA训练
├── examples/              # 使用示例
├── config.py              # 配置文件
├── data_loader.py         # 数据加载模块
├── utils.py               # 工具函数
├── run_all_experiments.py # 运行所有实验的主脚本
└── README.md              # 项目说明
```

## 运行实验

运行单个实验：

```bash
# 运行合成数据验证实验
python -m experiments.synthetic.run_experiment

# 运行大鼠海马体数据分析
python -m experiments.hippocampus.run_experiment

# 运行灵长类运动任务潜在动态分析
python -m experiments.primate.run_experiment

# 运行跨模态一致嵌入分析
python -m experiments.allen.run_experiment

# 运行自然视频从皮层解码实验
python -m experiments.video_decoding.run_experiment

# 运行多会话、多动物CEBRA训练
python -m experiments.multi_session.run_experiment
```

运行所有实验：

```bash
python run_all_experiments.py
```

## 参考文献

- Schneider, S., Lee, J., & Mathis, M. W. (2023). Learnable latent embeddings for joint behavioural and neural analyses. Nature, 616(7958), 487-493.
- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9729-9738).
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
