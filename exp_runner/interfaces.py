import abc

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Iterable
from typing import NoReturn


class Dataset(abc.ABC):

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def training(self) -> bool:
        pass


class Saver(abc.ABC):

    @abc.abstractmethod
    def save(self, report: List[Dict[str, Any]]) -> NoReturn:
        pass


class Metric(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true: Iterable[Union[float, int]], y_pred: Iterable[Union[float, int]]) -> float:
        pass
