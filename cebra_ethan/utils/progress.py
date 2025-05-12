#
# CEBRA-Ethan: 个人改进版的CEBRA
#
"""进度条工具。"""

import sys
from typing import Any, Dict, Iterator, Optional, Tuple, Union

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressBar:
    """进度条包装器，支持多种后端。
    
    这个类为数据加载器提供了一个统一的进度条接口，支持多种后端（如tqdm或简单的文本进度条）。
    
    Args:
        loader: 要包装的数据加载器
        backend: 要使用的后端，可以是"tqdm"或"off"
    """
    
    def __init__(self, loader, backend="tqdm"):
        """初始化进度条。
        
        Args:
            loader: 要包装的数据加载器
            backend: 要使用的后端，可以是"tqdm"或"off"
        """
        self.loader = loader
        self.backend = backend
        self.description = ""
        
        if backend == "tqdm" and not TQDM_AVAILABLE:
            print("警告：请求使用tqdm后端，但tqdm不可用。使用简单的文本进度条代替。", 
                  file=sys.stderr)
            self.backend = "text"
            
    def __iter__(self) -> Iterator[Tuple[int, Any]]:
        """迭代数据加载器，显示进度。
        
        Yields:
            (step, batch) 元组
        """
        if self.backend == "tqdm" and TQDM_AVAILABLE:
            with tqdm(self.loader) as pbar:
                for i, batch in enumerate(pbar):
                    pbar.set_description(self.description)
                    yield i, batch
        elif self.backend == "text":
            total = len(self.loader) if hasattr(self.loader, "__len__") else "?"
            for i, batch in enumerate(self.loader):
                if i % 10 == 0:
                    print(f"\r进度: {i}/{total} {self.description}", end="")
                yield i, batch
            print()  # 打印一个换行符
        else:  # backend == "off"
            for i, batch in enumerate(self.loader):
                yield i, batch
                
    def set_description(self, description: Union[str, Dict[str, float]]):
        """设置进度条的描述。
        
        Args:
            description: 描述字符串或包含指标的字典
        """
        if isinstance(description, dict):
            desc_str = " ".join(f"{k}={v:.4f}" for k, v in description.items())
        else:
            desc_str = str(description)
            
        self.description = desc_str
