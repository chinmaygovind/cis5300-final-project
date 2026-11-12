from pathlib import Path

from datasets import Dataset, disable_progress_bars, load_dataset

disable_progress_bars()  # hide datasets' own loading bars; models show a tqdm bar while predicting

DATA_DIR = Path(__file__).resolve().parent.parent / 'Milestone 1' / 'data'
TASKS = ['matching', 'ranking']


def load_split(task: str, split: str) -> Dataset:
    """One split (train/dev/test) of a task, with a uniform `id` column and absolute image paths."""
    ds = load_dataset('json', data_files=str(DATA_DIR / task / f'{split}.jsonl'), split='train')
    return ds.map(lambda ex: {'id': ex.get('pair_id') or ex['instance_id'],
                              'image': str(DATA_DIR / ex['image'])})
