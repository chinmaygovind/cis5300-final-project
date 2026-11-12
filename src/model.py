from abc import ABC, abstractmethod

from datasets import Dataset
from tqdm import tqdm


class Model(ABC):
    """A system for one task ('matching' or 'ranking'), built from that task's train split."""

    def __init__(self, train: Dataset, task: str):
        self.train = train
        self.task = task

    def predict(self, test: Dataset) -> list[str | None]:
        return [self.predict_one(ex) for ex in tqdm(test, desc=f'{type(self).__name__} {self.task}')]

    @abstractmethod
    def predict_one(self, ex: dict) -> str | None:
        """The answer letter for one example. None means no valid answer."""
