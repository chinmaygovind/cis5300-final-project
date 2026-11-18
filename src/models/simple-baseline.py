from collections import Counter


class Majority:
    """Always answers the most common training label.
    Test accuracy: matching 18.37% (always E), ranking 50.00% (always B)."""

    def __init__(self, train, task):
        self.label = Counter(train['label']).most_common(1)[0][0]

    def predict_one(self, ex):
        return self.label
