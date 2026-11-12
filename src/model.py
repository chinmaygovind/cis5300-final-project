from abc import ABC, abstractmethod

from datasets import Dataset


class Model(ABC):
    """A system for one task. Built from the train split, then asked to answer a test split."""

    def __init__(self, train: Dataset):
        self.train = train

    @abstractmethod
    def predict(self, test: Dataset) -> list[str | None]:
        """One answer letter per test example, in order. None means no valid answer."""
