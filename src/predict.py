"""Usage: python src/predict.py <src/models/some_model.py> <matching|ranking> <dev|test> <out.jsonl>

Runs a model on one split and writes one JSON line per example:
  {"pair_id": ..., "prediction": "B"}       (ranking)
  {"instance_id": ..., "prediction": "D"}   (matching)

A model file defines one class with __init__(train, task) and predict_one(ex) -> letter or None.
"""
import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import disable_progress_bars, load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / 'Milestone 1' / 'data'
TASKS = ['matching', 'ranking']


def load_split(task, split):
    """One split of a task, with a uniform `id` column and absolute image paths."""
    ds = load_dataset('json', data_files=str(DATA_DIR / task / f'{split}.jsonl'), split='train')
    return ds.map(lambda ex: {'id': ex.get('pair_id') or ex['instance_id'],
                              'image': str(DATA_DIR / ex['image'])})


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[2] not in TASKS:
        sys.exit(__doc__)
    path, task, split, out = sys.argv[1:]
    disable_progress_bars()

    ModelClass = next(v for v in runpy.run_path(path).values()
                      if isinstance(v, type) and hasattr(v, 'predict_one'))
    model = ModelClass(load_split(task, 'train'), task)
    test = load_split(task, split)

    id_key = 'pair_id' if task == 'ranking' else 'instance_id'
    # API models can set `workers` to send requests in parallel; map() keeps the original order
    with ThreadPoolExecutor(getattr(model, 'workers', 1)) as pool, open(out, 'w') as f:
        preds = pool.map(model.predict_one, test)
        for ex, pred in zip(test, tqdm(preds, total=len(test), desc=f'{ModelClass.__name__} {task} {split}')):
            f.write(json.dumps({id_key: ex['id'], 'prediction': pred}) + '\n')
