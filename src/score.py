"""Usage: python src/score.py <src/some_model.py> <matching|ranking> [dev|test]"""
import importlib
import sys
from pathlib import Path

from data import TASKS, load_split
from model import Model


def accuracy(predictions, labels):
    return sum(p == l for p, l in zip(predictions, labels)) / len(labels)


if len(sys.argv) not in (3, 4) or sys.argv[2] not in TASKS:
    sys.exit(__doc__)
path, task = sys.argv[1], sys.argv[2]
split = sys.argv[3] if len(sys.argv) == 4 else 'test'

importlib.import_module(Path(path).stem)  # e.g. src/majority.py -> import majority
ModelClass = Model.__subclasses__()[0]      # the class that file defined, e.g. Majority

test = load_split(task, split)
predictions = ModelClass(load_split(task, 'train'), task).predict(test)
print(f'{task} {split}: accuracy {accuracy(predictions, test["label"]):.2%} (n={len(test)})')
if 'tier' in test.column_names:
    for tier in ['easy', 'hard']:
        rows = [(p, l) for p, l, t in zip(predictions, test['label'], test['tier']) if t == tier]
        print(f'  {tier}: {accuracy(*zip(*rows)):.2%} (n={len(rows)})')
