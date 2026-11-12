from collections import Counter

from model import Model


class Majority(Model):
    """Always answers the most common training label."""

    def __init__(self, train, task):
        super().__init__(train, task)
        self.label = Counter(train['label']).most_common(1)[0][0]

    def predict_one(self, ex):
        return self.label
