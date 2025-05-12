#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""用于跟踪指标的工具类。"""


class Meter:
    """用于跟踪和计算平均值的简单计量器。
    
    这个类用于在训练或评估过程中跟踪指标的平均值。
    """
    
    def __init__(self):
        """初始化一个空的计量器。"""
        self.reset()
        
    def reset(self):
        """重置计量器。"""
        self.sum = 0.0
        self.count = 0
        
    def add(self, value, count=1):
        """添加一个值到计量器。
        
        Args:
            value: 要添加的值
            count: 该值的计数（默认为1）
        """
        self.sum += value * count
        self.count += count
        
    @property
    def average(self):
        """计算当前的平均值。
        
        Returns:
            当前的平均值，如果计数为0则返回0
        """
        return self.sum / max(self.count, 1)
