from collections import Counter

from model import Model


class Majority(Model):
    """Always answers the most common training label."""

    def predict(self, test):
        label = Counter(self.train['label']).most_common(1)[0][0]
        return [label] * len(test)
