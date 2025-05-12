#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""多目标学习示例，同时学习时间和行为相关的嵌入。"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 导入CEBRA-Ethan
import sys
import os
sys.path.append(os.path.abspath('..'))

from cebra_ethan.data.dataset import BehavioralDataset
from cebra_ethan.data.loader import MultiObjectiveLoader
from cebra_ethan.models.model import AttentionOffsetModel, Offset1Model
from cebra_ethan.models.criterions import LearnableCosineInfoNCE, FocalInfoNCE
from cebra_ethan.solver.base import EarlyStoppingSolver


def generate_synthetic_data(num_samples=1000, num_neurons=50, num_behavior=5, num_latent=3):
    """生成合成神经和行为数据用于演示。

    Args:
        num_samples: 样本数量
        num_neurons: 神经元数量
        num_behavior: 行为特征数量
        num_latent: 潜在变量数量

    Returns:
        合成神经数据和行为数据
    """
    # 生成潜在变量
    latent = np.random.randn(num_samples, num_latent)

    # 生成随机投影矩阵
    neural_projection = np.random.randn(num_latent, num_neurons)
    behavior_projection = np.random.randn(num_latent, num_behavior)

    # 生成神经数据和行为数据
    neural_data = np.dot(latent, neural_projection)
    behavior_data = np.dot(latent, behavior_projection)

    # 添加噪声
    neural_data += np.random.randn(num_samples, num_neurons) * 0.1
    behavior_data += np.random.randn(num_samples, num_behavior) * 0.05

    return neural_data, behavior_data


class MultiObjectiveSolver:
    """多目标求解器，用于同时学习时间和行为相关的嵌入。

    Args:
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        num_behavior_features: 用于行为对比学习的特征数量
        device: 设备
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        num_behavior_features=3,
        device=None,
    ):
        """初始化多目标求解器。"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_behavior_features = num_behavior_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model.to(self.device)
        self.log = {
            "time_loss": [],
            "behavior_loss": [],
            "total_loss": [],
        }
        self.best_loss = float("inf")

    def fit(
        self,
        loader,
        num_epochs=100,
        save_frequency=10,
        logdir=None,
    ):
        """训练模型。

        Args:
            loader: 多目标数据加载器
            num_epochs: 训练轮数
            save_frequency: 保存模型的频率
            logdir: 日志目录
        """
        self.model.train()

        for epoch in range(num_epochs):
            epoch_time_loss = 0.0
            epoch_behavior_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0

            for time_batch, behavior_batch in loader:
                # 将数据移动到设备
                time_batch.reference = time_batch.reference.to(self.device)
                time_batch.positive = time_batch.positive.to(self.device)
                time_batch.negative = time_batch.negative.to(self.device)

                behavior_batch.reference = behavior_batch.reference.to(self.device)
                behavior_batch.positive = behavior_batch.positive.to(self.device)
                behavior_batch.negative = behavior_batch.negative.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()

                # 时间对比学习
                time_ref_embed = self.model(time_batch.reference)
                time_pos_embed = self.model(time_batch.positive)
                time_neg_embed = self.model(time_batch.negative)

                time_loss, _, _ = self.criterion(
                    time_ref_embed, time_pos_embed, time_neg_embed
                )

                # 行为对比学习
                behavior_ref_embed = self.model(behavior_batch.reference)
                behavior_pos_embed = self.model(behavior_batch.positive)
                behavior_neg_embed = self.model(behavior_batch.negative)

                behavior_loss, _, _ = self.criterion(
                    behavior_ref_embed, behavior_pos_embed, behavior_neg_embed
                )

                # 总损失
                total_loss = time_loss + behavior_loss

                # 反向传播和优化
                total_loss.backward()
                self.optimizer.step()

                # 记录损失
                epoch_time_loss += time_loss.item()
                epoch_behavior_loss += behavior_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1

            # 计算平均损失
            epoch_time_loss /= num_batches
            epoch_behavior_loss /= num_batches
            epoch_total_loss /= num_batches

            # 记录日志
            self.log["time_loss"].append(epoch_time_loss)
            self.log["behavior_loss"].append(epoch_behavior_loss)
            self.log["total_loss"].append(epoch_total_loss)

            # 打印进度
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Time Loss: {epoch_time_loss:.4f}, "
                  f"Behavior Loss: {epoch_behavior_loss:.4f}, "
                  f"Total Loss: {epoch_total_loss:.4f}")

            # 保存模型
            if (epoch + 1) % save_frequency == 0 and logdir is not None:
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
                torch.save({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "log": self.log,
                }, os.path.join(logdir, f"checkpoint_{epoch+1}.pth"))

            # 更新最佳模型
            if epoch_total_loss < self.best_loss:
                self.best_loss = epoch_total_loss
                if logdir is not None:
                    if not os.path.exists(logdir):
                        os.makedirs(logdir)
                    torch.save({
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "log": self.log,
                    }, os.path.join(logdir, "checkpoint_best.pth"))

    @torch.no_grad()
    def transform(self, inputs):
        """使用模型生成嵌入。

        Args:
            inputs: 输入数据

        Returns:
            嵌入
        """
        self.model.eval()
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)
        return self.model(inputs).cpu().numpy()


def main():
    """主函数，展示多目标学习的用法。"""
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成合成数据
    print("生成合成神经和行为数据...")
    neural_data, behavior_data = generate_synthetic_data(
        num_samples=1000,
        num_neurons=50,
        num_behavior=5,
        num_latent=3
    )
    print(f"神经数据形状: {neural_data.shape}")
    print(f"行为数据形状: {behavior_data.shape}")

    # 创建数据集
    print("创建行为数据集...")
    dataset = BehavioralDataset(
        neural_data,
        behavior_data,
        time_offset=1,
        num_negatives=1  # 使用单个负样本，与我们对ContrastiveDataset的修改兼容
    )

    # 获取一个样本，检查形状
    sample = dataset[0]
    print(f"样本形状 - 参考: {sample[0].shape}, 正样本: {sample[1].shape}, 负样本: {sample[2].shape}, 行为: {sample[3].shape}")

    # 创建数据加载器
    print("创建多目标数据加载器...")
    batch_size = 32
    loader = MultiObjectiveLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        behavior_similarity_threshold=0.1
    )

    # 获取一个批次，检查形状
    for time_batch, behavior_batch in loader:
        print(f"时间批次形状 - 参考: {time_batch.reference.shape}, 正样本: {time_batch.positive.shape}, 负样本: {time_batch.negative.shape}")
        print(f"行为批次形状 - 参考: {behavior_batch.reference.shape}, 正样本: {behavior_batch.positive.shape}, 负样本: {behavior_batch.negative.shape}")
        break

    # 创建模型
    print("创建模型...")
    num_neurons = neural_data.shape[1]
    num_hidden = 128
    num_output = 6  # 3维用于时间对比，3维用于行为对比

    # 使用Offset1Model，它使用全连接层而不是卷积层
    model = Offset1Model(
        num_neurons=num_neurons,
        num_units=num_hidden,
        num_output=num_output,
        normalize=True
    )

    # 或者使用带有注意力机制的改进模型
    # model = AttentionOffsetModel(
    #     num_neurons=num_neurons,
    #     num_units=num_hidden,
    #     num_output=num_output,
    #     num_heads=4,
    #     normalize=True
    # )

    # 创建损失函数
    print("创建损失函数...")
    # 使用焦点损失
    criterion = FocalInfoNCE(temperature=1.0, gamma=2.0)

    # 创建优化器
    print("创建优化器...")
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )

    # 创建多目标求解器
    print("创建多目标求解器...")
    solver = MultiObjectiveSolver(
        model,
        criterion,
        optimizer,
        num_behavior_features=3
    )

    # 训练模型
    print("训练模型...")
    solver.fit(
        loader,
        num_epochs=10,  # 减少训练轮数以加快示例运行
        save_frequency=5,
        logdir="./checkpoints/multiobjective"
    )

    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(solver.log["time_loss"], label="Time Loss")
    plt.plot(solver.log["behavior_loss"], label="Behavior Loss")
    plt.plot(solver.log["total_loss"], label="Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Multi-objective Training Loss")
    plt.legend()
    plt.savefig("multiobjective_loss.png")

    # 使用训练好的模型进行嵌入
    print("生成嵌入...")
    embeddings = solver.transform(neural_data)

    print(f"嵌入形状: {embeddings.shape}")

    # 可视化嵌入
    # 时间相关嵌入（前3维）
    time_embeddings = embeddings[:, :3]

    # 行为相关嵌入（后3维）
    behavior_embeddings = embeddings[:, 3:]

    # 可视化时间相关嵌入
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(time_embeddings[:, 0], time_embeddings[:, 1], time_embeddings[:, 2],
               c=np.arange(len(time_embeddings)), cmap='viridis')
    ax1.set_title('Time-related Embeddings')
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    ax1.set_zlabel('Dim 3')

    # 可视化行为相关嵌入
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(behavior_embeddings[:, 0], behavior_embeddings[:, 1], behavior_embeddings[:, 2],
               c=np.arange(len(behavior_embeddings)), cmap='plasma')
    ax2.set_title('Behavior-related Embeddings')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_zlabel('Dim 3')

    plt.tight_layout()
    plt.savefig("multiobjective_embeddings.png")

    print("多目标学习示例完成！")


if __name__ == "__main__":
    main()
