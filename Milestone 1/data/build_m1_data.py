"""Build CIS 5300 project Milestone 1 data: Hessel matching + Zhang pairwise ranking.

usage: uv run --no-project --with pandas --with pyarrow python -I build_m1_data.py RAW_DIR OUT_DIR
RAW_DIR holds the HF parquet downloads (see data.md for URLs).
"""
import json, random, sys
from pathlib import Path
import pandas as pd

SPLITS = {"train": "train", "validation": "dev", "test": "test"}
# 0-indexed ranks in Zhang's data: rank 0 = best caption
TIERS = {"easy": (range(0, 10), range(999, 1009)), "hard": (range(29, 39), range(299, 309))}


def make_pairs(r, contest, rng):
    """Pair i-th caption of the top band with i-th of the bottom band, shuffle A/B."""
    by_rank = r.set_index("rank")
    out = []
    for tier, (top, bot) in TIERS.items():
        for i, (t, b) in enumerate(zip(top, bot)):
            hi, lo = by_rank.loc[t], by_rank.loc[b]
            a, bb, label = (hi, lo, "A") if rng.random() < 0.5 else (lo, hi, "B")
            out.append({
                "pair_id": f"{contest}_{tier}_{i}", "contest_number": int(contest), "tier": tier,
                "caption_A": a.caption, "caption_B": bb.caption,
                "rank_A": int(a.name) + 1, "rank_B": int(bb.name) + 1,  # 1-indexed for humans
                "mean_A": round(float(a["mean"]), 4), "mean_B": round(float(bb["mean"]), 4),
                "label": label,  # which caption the crowd rated higher
            })
    return out


def main(raw, out):
    raw, out = Path(raw), Path(out)
    rng = random.Random(5300)
    counts = {}
    seen = {"matching": {}, "ranking": {}}
    for hf, sp in SPLITS.items():
        # --- matching (Hessel split 0, contest-disjoint already) ---
        m = pd.read_parquet(raw / f"hessel_matching_{hf}.parquet")
        img_dir = out / "images" / "matching"; img_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in m.itertuples():
            p = img_dir / f"{r.contest_number}.jpg"
            if not p.exists():
                p.write_bytes(r.image["bytes"])
            rows.append({
                "instance_id": r.instance_id, "contest_number": int(r.contest_number),
                "image": f"images/matching/{r.contest_number}.jpg",
                "location": r.image_location, "description": r.image_description,
                "uncanny": r.image_uncanny_description, "entities": list(r.entities),
                "choices": list(r.caption_choices), "label": r.label,
            })
        write_jsonl(out / "matching" / f"{sp}.jsonl", rows)
        counts[("matching", sp)] = (len(rows), m.contest_number.nunique())
        seen["matching"][sp] = set(m.contest_number)

        # --- pairwise ranking (Zhang's contest-level split) ---
        rat = pd.read_parquet(raw / f"zhang_1_rating_{hf}.parquet")
        desc = pd.read_parquet(raw / f"zhang_2_gpt4o_description_{hf}.parquet")
        cart = pd.read_parquet(raw / f"zhang_cartoons_{hf}.parquet")
        img_dir = out / "images" / "ranking"; img_dir.mkdir(parents=True, exist_ok=True)
        for im in cart.image:
            (img_dir / im["path"]).write_bytes(im["bytes"])
        d = desc.set_index("contest_number")
        pairs = []
        for contest, r in rat.groupby("contest_number"):
            for p in make_pairs(r, contest, rng):
                p["image"] = f"images/ranking/{contest}.jpg"
                dd = d.loc[contest]
                p.update(location=dd.location, description=dd.canny, uncanny=dd.uncanny,
                         entities=list(dd.entities))
                pairs.append(p)
        write_jsonl(out / "ranking" / f"{sp}.jsonl", pairs)
        counts[("ranking", sp)] = (len(pairs), rat.contest_number.nunique())
        seen["ranking"][sp] = set(rat.contest_number)

    # self-check: no contest in two splits, labels balanced-ish, every image present
    for task, s in seen.items():
        assert not (s["train"] & s["dev"] or s["train"] & s["test"] or s["dev"] & s["test"]), task
    for task in ("matching", "ranking"):
        for sp in ("train", "dev", "test"):
            for row in map(json.loads, open(out / task / f"{sp}.jsonl")):
                assert (out / row["image"]).exists(), row["image"]
    for k, v in counts.items():
        print(*k, "items=%d contests=%d" % v)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
