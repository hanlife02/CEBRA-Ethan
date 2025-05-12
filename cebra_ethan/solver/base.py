#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""这个包包含不同求解器的抽象基类。

求解器用于打包模型、损失函数和优化器，并实现训练循环。
当子类化抽象求解器时，在最简单的情况下，只需要重写Solver._inference方法。

对于更复杂的用例，可以重写Solver.step和Solver.fit方法来实现对训练循环的更大更改。
"""

import abc
import os
import warnings
from typing import Callable, Dict, List, Literal, Optional, Any, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

# 导入工具类
from cebra_ethan.utils.meter import Meter
from cebra_ethan.utils.progress import ProgressBar


class HasDevice:
    """具有设备属性的类的基类。"""

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        """获取当前设备。"""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "HasDevice":
        """将模型移动到指定设备。

        Args:
            device: 目标设备

        Returns:
            self
        """
        self._device = torch.device(device)
        return self


class Batch:
    """数据批次，包含参考样本、正样本和负样本。

    Args:
        reference: 参考样本
        positive: 正样本
        negative: 负样本
        index: 样本索引
    """

    def __init__(
        self,
        reference: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        index: Optional[torch.Tensor] = None,
    ):
        self.reference = reference
        self.positive = positive
        self.negative = negative
        self.index = index


class Solver(abc.ABC, HasDevice):
    """求解器基类。

    求解器包含用于捆绑模型、损失函数和优化器的辅助方法。

    Attributes:
        model: 用于转换参考、正面和负面样本的编码器。
        criterion: 根据正对之间和负对之间的相似性计算的损失函数。损失函数本身可以有可训练的参数。
        optimizer: 用于更新模型和损失函数参数的PyTorch优化器。
        log: 训练期间记录的日志，通常包含"total"损失以及正对("pos")和负对("neg")的日志。
             对于CEBRA-Ethan中的标准损失函数，还包含"temperature"的值。
        tqdm_on: 是否使用tqdm在训练期间显示进度条。
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        tqdm_on: bool = True,
    ):
        HasDevice.__init__(self)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.log: Dict = {
            "pos": [],
            "neg": [],
            "total": [],
            "temperature": []
        }
        self.tqdm_on = tqdm_on
        self.best_loss = float("inf")

    def state_dict(self) -> dict:
        """返回完全描述当前求解器状态的字典。

        Returns:
            状态字典，包括模型和优化器的状态字典。还包含训练历史和模型训练时使用的CEBRA-Ethan版本。
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
            "log": self.log,
            "version": "0.1.0",  # CEBRA-Ethan版本
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """使用给定的state_dict更新求解器状态。

        Args:
            state_dict: 包含模型、优化器和过去损失历史的参数字典。
            strict: 确保所有状态都可以加载。设置为False以允许部分加载所有给定键的状态。
        """
        def _contains(key):
            if key in state_dict:
                return True
            elif strict:
                raise KeyError(
                    f"state_dict中缺少键{key}。包含：{list(state_dict.keys())}。"
                )
            return False

        def _get(key):
            return state_dict.get(key)

        if _contains("model"):
            self.model.load_state_dict(_get("model"))
        if _contains("criterion"):
            self.criterion.load_state_dict(_get("criterion"))
        if _contains("optimizer"):
            self.optimizer.load_state_dict(_get("optimizer"))
        if _contains("log"):
            self.log = _get("log")

    @property
    def num_parameters(self) -> int:
        """编码器和损失函数中的参数总数。"""
        return sum(p.numel() for p in self.parameters())

    def parameters(self):
        """迭代所有参数。"""
        for parameter in self.model.parameters():
            yield parameter

        for parameter in self.criterion.parameters():
            yield parameter

    def _get_loader(self, loader):
        """获取带有进度条的数据加载器。"""
        return ProgressBar(
            loader,
            "tqdm" if self.tqdm_on else "off",
        )

    def fit(
        self,
        loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        *,
        save_frequency: Optional[int] = None,
        valid_frequency: Optional[int] = None,
        logdir: Optional[str] = None,
        save_hook: Optional[Callable[[int, "Solver"], None]] = None,
    ):
        """训练模型指定的步数。

        Args:
            loader: 数据加载器，它是Batch实例的迭代器。每个批次包含参考、正面和负面输入样本。
            valid_loader: 用于模型验证的数据加载器。
            save_frequency: 如果不是None，则自动保存模型检查点到logdir的频率。
            valid_frequency: 在valid_loader实例上运行验证的频率。
            logdir: 写入模型检查点的日志目录。可以使用solver.load函数再次读取检查点，或者通过加载状态字典手动读取。
        """
        self.to(loader.dataset.device if hasattr(loader.dataset, 'device') else self.device)

        iterator = self._get_loader(loader)
        self.model.train()
        for num_steps, batch in iterator:
            stats = self.step(batch)
            iterator.set_description(stats)

            if save_frequency is None:
                continue
            save_model = num_steps % save_frequency == 0
            run_validation = (valid_loader is not None) and (num_steps % valid_frequency == 0)
            if run_validation:
                validation_loss = self.validation(valid_loader)
                if self.best_loss is None or validation_loss < self.best_loss:
                    self.best_loss = validation_loss
                    self.save(logdir, "checkpoint_best.pth")
            if save_model:
                if save_hook is not None:
                    save_hook(num_steps, self)
                if logdir is not None:
                    self.save(logdir, f"checkpoint_{num_steps:#07d}.pth")

    def step(self, batch: Batch) -> dict:
        """执行单个梯度更新。

        Args:
            batch: 输入样本

        Returns:
            包含训练指标的字典
        """
        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss, align, uniform = self.criterion(prediction.reference,
                                            prediction.positive,
                                            prediction.negative)

        loss.backward()
        self.optimizer.step()

        stats = dict(
            pos=align.item(),
            neg=uniform.item(),
            total=loss.item(),
            temperature=getattr(self.criterion, "temperature", 1.0),
        )
        for key, value in stats.items():
            self.log[key].append(value)
        return stats

    def validation(self, loader: DataLoader) -> float:
        """计算模型在数据上的得分。

        Args:
            loader: 数据加载器，它是Batch实例的迭代器。每个批次包含参考、正面和负面输入样本。

        Returns:
            在数据批次上迭代的平均损失。
        """
        iterator = self._get_loader(loader)
        total_loss = Meter()
        self.model.eval()
        for _, batch in iterator:
            prediction = self._inference(batch)
            loss, _, _ = self.criterion(prediction.reference,
                                      prediction.positive,
                                      prediction.negative)
            total_loss.add(loss.item())
        return total_loss.average

    @torch.no_grad()
    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算嵌入。

        此函数默认仅应用给定模型的forward函数，在将其切换到eval模式后。

        Args:
            inputs: 输入信号

        Returns:
            输出嵌入
        """
        self.model.eval()
        return self.model(inputs)

    @abc.abstractmethod
    def _inference(self, batch: Batch) -> Batch:
        """给定一批输入示例，返回模型输出。

        Args:
            batch: 输入数据，不一定在批次维度上对齐。这意味着batch.index指定了
                  参考/正样本之间的映射（如果不等于None）。

        Returns:
            处理后的数据批次。虽然输入数据可能不在样本维度上对齐，
            但输出数据应该对齐，并且batch.index应该设置为None。
        """
        raise NotImplementedError

    def load(self, logdir, filename="checkpoint.pth"):
        """从其检查点文件加载实验。

        Args:
            filename: 用于加载实验的检查点名称。
        """
        savepath = os.path.join(logdir, filename)
        if not os.path.exists(savepath):
            print("未找到以前的实验。从头开始。")
            return
        checkpoint = torch.load(savepath, map_location=self.device)
        self.load_state_dict(checkpoint, strict=True)

    def save(self, logdir, filename="checkpoint_last.pth"):
        """保存模型和优化器参数。

        Args:
            logdir: 此模型的日志目录。
            filename: 用于保存实验的检查点名称。
        """
        if not os.path.exists(os.path.dirname(logdir)):
            os.makedirs(logdir)
        savepath = os.path.join(logdir, filename)
        torch.save(
            self.state_dict(),
            savepath,
        )


class SingleSessionSolver(Solver):
    """单会话求解器，用于处理单个数据会话。

    这个求解器实现了_inference方法，用于处理单个数据会话的批次。
    """

    def _inference(self, batch: Batch) -> Batch:
        """处理单个会话的批次。

        Args:
            batch: 输入批次

        Returns:
            处理后的批次
        """
        reference = self.model(batch.reference)
        positive = self.model(batch.positive)
        negative = self.model(batch.negative)

        return Batch(
            reference=reference,
            positive=positive,
            negative=negative,
            index=None,
        )


# 改进版本：添加带有早停的求解器
class EarlyStoppingSolver(SingleSessionSolver):
    """带有早停功能的单会话求解器。

    这个求解器在验证损失不再改善时提前停止训练。

    Args:
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        patience: 在停止训练之前等待验证损失改善的轮数
        min_delta: 被视为改进的最小变化量
        tqdm_on: 是否使用tqdm显示进度条
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        patience: int = 10,
        min_delta: float = 0.0,
        tqdm_on: bool = True,
    ):
        super().__init__(model, criterion, optimizer, tqdm_on)
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def fit(
        self,
        loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        *,
        save_frequency: Optional[int] = None,
        valid_frequency: Optional[int] = None,
        logdir: Optional[str] = None,
        save_hook: Optional[Callable[[int, "Solver"], None]] = None,
    ):
        """训练模型，带有早停功能。

        Args:
            loader: 数据加载器
            valid_loader: 验证数据加载器
            save_frequency: 保存模型的频率
            valid_frequency: 验证的频率
            logdir: 日志目录
            save_hook: 保存钩子
        """
        self.to(loader.dataset.device if hasattr(loader.dataset, 'device') else self.device)

        iterator = self._get_loader(loader)
        self.model.train()
        for num_steps, batch in iterator:
            stats = self.step(batch)
            iterator.set_description(stats)

            if valid_loader is not None and num_steps % valid_frequency == 0:
                validation_loss = self.validation(valid_loader)

                # 检查是否需要早停
                if validation_loss < self.best_loss - self.min_delta:
                    self.best_loss = validation_loss
                    self.counter = 0
                    if logdir is not None:
                        self.save(logdir, "checkpoint_best.pth")
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print(f"早停! 验证损失在{self.patience}轮内没有改善。")
                        self.early_stop = True
                        break

            if save_frequency is not None and num_steps % save_frequency == 0:
                if save_hook is not None:
                    save_hook(num_steps, self)
                if logdir is not None:
                    self.save(logdir, f"checkpoint_{num_steps:#07d}.pth")

            if self.early_stop:
                break
