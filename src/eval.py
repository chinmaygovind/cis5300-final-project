"""Usage: python src/eval.py <src/some_model.py> <matching|ranking> [dev|test]"""
import importlib.util
import inspect
import sys

from data import TASKS, load_split
from model import Model


def load_model_class(path):
    spec = importlib.util.spec_from_file_location('m', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return next(c for _, c in inspect.getmembers(module, inspect.isclass) if issubclass(c, Model) and c is not Model)


def accuracy(predictions, labels):
    return sum(p == l for p, l in zip(predictions, labels)) / len(labels)


if len(sys.argv) not in (3, 4) or sys.argv[2] not in TASKS:
    sys.exit(__doc__)
path, task = sys.argv[1], sys.argv[2]
split = sys.argv[3] if len(sys.argv) == 4 else 'test'

test = load_split(task, split)
predictions = load_model_class(path)(load_split(task, 'train')).predict(test)
print(f'{task} {split}: accuracy {accuracy(predictions, test["label"]):.2%} (n={len(test)})')
if 'tier' in test.column_names:
    for tier in ['easy', 'hard']:
        rows = [(p, l) for p, l, t in zip(predictions, test['label'], test['tier']) if t == tier]
        print(f'  {tier}: {accuracy(*zip(*rows)):.2%} (n={len(rows)})')
