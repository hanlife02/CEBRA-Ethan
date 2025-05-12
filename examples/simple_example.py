#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""一个简单的CEBRA-Ethan使用示例。"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# 导入CEBRA-Ethan
import sys
import os
sys.path.append(os.path.abspath('..'))

from cebra_ethan.data.dataset import NeuralDataset, ContrastiveDataset
from cebra_ethan.data.loader import ContrastiveLoader, AugmentedContrastiveLoader
from cebra_ethan.models.model import Offset1Model, AttentionOffsetModel
from cebra_ethan.models.criterions import LearnableCosineInfoNCE, FocalInfoNCE
from cebra_ethan.solver.base import SingleSessionSolver, EarlyStoppingSolver


def generate_synthetic_data(num_samples=1000, num_neurons=50, num_latent=3):
    """生成合成神经数据用于演示。

    Args:
        num_samples: 样本数量
        num_neurons: 神经元数量
        num_latent: 潜在变量数量

    Returns:
        合成神经数据，形状为(time_steps, neurons)
    """
    # 生成潜在变量
    latent = np.random.randn(num_samples, num_latent)

    # 生成随机投影矩阵
    projection = np.random.randn(num_latent, num_neurons)

    # 生成神经数据
    neural_data = np.dot(latent, projection)

    # 添加噪声
    neural_data += np.random.randn(num_samples, num_neurons) * 0.1

    # 确保数据形状是 (time_steps, neurons)
    # 这里不需要转置，因为已经是正确的形状

    return neural_data


def main():
    """主函数，展示CEBRA-Ethan的基本用法。"""
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成合成数据
    print("生成合成神经数据...")
    neural_data = generate_synthetic_data(num_samples=1000, num_neurons=50, num_latent=3)
    print(f"神经数据形状: {neural_data.shape}")

    # 创建数据集
    print("创建对比数据集...")
    dataset = ContrastiveDataset(neural_data, time_offset=1, num_negatives=10)

    # 获取一个样本，检查形状
    sample_ref, sample_pos, sample_neg = dataset[0]
    print(f"样本形状 - 参考: {sample_ref.shape}, 正样本: {sample_pos.shape}, 负样本: {sample_neg.shape}")

    # 创建数据加载器
    print("创建数据加载器...")
    batch_size = 32
    loader = AugmentedContrastiveLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        noise_level=0.05,
        dropout_prob=0.1
    )

    # 获取一个批次，检查形状
    for batch in loader:
        print(f"批次形状 - 参考: {batch.reference.shape}, 正样本: {batch.positive.shape}, 负样本: {batch.negative.shape}")
        break

    # 创建验证数据集和加载器
    val_dataset = ContrastiveDataset(neural_data[-200:], time_offset=1, num_negatives=10)
    val_loader = ContrastiveLoader(val_dataset, batch_size=batch_size)

    # 创建模型
    print("创建模型...")
    num_neurons = neural_data.shape[1]
    num_hidden = 128
    num_output = 3

    # 使用Offset1Model，它使用全连接层而不是卷积层
    model = Offset1Model(num_neurons, num_hidden, num_output, normalize=True)

    # 或者使用带有注意力机制的改进模型
    # model = AttentionOffsetModel(num_neurons, num_hidden, num_output, num_heads=4, normalize=True)

    # 创建损失函数
    print("创建损失函数...")
    # 使用标准损失函数
    criterion = LearnableCosineInfoNCE(temperature=1.0)

    # 或者使用带有焦点损失的改进损失函数
    # criterion = FocalInfoNCE(temperature=1.0, gamma=2.0)

    # 创建优化器
    print("创建优化器...")
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )

    # 创建求解器
    print("创建求解器...")
    # 使用标准求解器
    # solver = SingleSessionSolver(model, criterion, optimizer)

    # 或者使用带有早停的改进求解器
    solver = EarlyStoppingSolver(
        model,
        criterion,
        optimizer,
        patience=5,
        min_delta=0.01
    )

    # 训练模型
    print("训练模型...")
    solver.fit(
        loader,
        valid_loader=val_loader,
        save_frequency=10,
        valid_frequency=5,
        logdir="./checkpoints"
    )

    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(solver.log["total"], label="Total Loss")
    plt.plot(solver.log["pos"], label="Positive Loss")
    plt.plot(solver.log["neg"], label="Negative Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("CEBRA-Ethan Training Loss")
    plt.legend()
    plt.savefig("training_loss.png")

    # 使用训练好的模型进行嵌入
    print("生成嵌入...")
    with torch.no_grad():
        embeddings = solver.transform(torch.tensor(neural_data, dtype=torch.float32).to(solver.device))
        embeddings = embeddings.cpu().numpy()

    print(f"嵌入形状: {embeddings.shape}")

    # 可视化嵌入
    if embeddings.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=np.arange(len(embeddings)), cmap='viridis')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('CEBRA-Ethan Embeddings')
        plt.savefig("embeddings_3d.png")
    elif embeddings.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=np.arange(len(embeddings)), cmap='viridis')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('CEBRA-Ethan Embeddings')
        plt.colorbar(label='Time')
        plt.savefig("embeddings_2d.png")

    print("示例完成！")


if __name__ == "__main__":
    main()
