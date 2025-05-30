"""
Copyright (c) 2022 Hocheol Lim.
"""

from abc import ABCMeta, abstractmethod
from typing import List, Union


class AbstractLogger:
    @abstractmethod
    def log_metric(self, name: str, value: Union[int, float]):
        raise NotImplementedError

    @abstractmethod
    def log_text(self, name: str, text: str):
        raise NotImplementedError

    @abstractmethod
    def log_values(self, name: str, values: List[float]):
        raise NotImplementedError
