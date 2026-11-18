"""usage: python score.py <true.jsonl> <predictions.jsonl>"""
import json
import sys



def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def row_id(row):
    return row.get('pair_id') or row.get('instance_id')


def accuracy(true_rows, preds):
    return sum(1 for row in true_rows if preds.get(row_id(row)) == row['label']) / len(true_rows)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    true_rows = read_jsonl(sys.argv[1])
    preds = {row_id(row): row['prediction'] for row in read_jsonl(sys.argv[2])}
    acc = accuracy(true_rows, preds)
    print(f'accuracy {acc:.2%} (n={len(true_rows)})')
    for tier in ['easy', 'hard']:
        subset = [row for row in true_rows if row.get('tier') == tier]
        if subset:  # matching has no tiers
            print(f'  {tier}: {accuracy(subset, preds):.2%} (n={len(subset)})')
