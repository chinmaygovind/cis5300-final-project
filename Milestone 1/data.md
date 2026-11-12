# Data

There are two tasks, both built from The New Yorker Cartoon Caption Contest. Every split is **contest-disjoint**: no cartoon appears in more than one split. `build_m1_data.py` checks this with an assert.

| Task | Split | Items | Contests |
|---|---|---|---|
| Matching | train | 4,896 | 422 |
| Matching | dev | 531 | 141 |
| Matching | test | 528 | 141 |
| Ranking | train | 5,420 | 271 |
| Ranking | dev | 880 | 44 |
| Ranking | test | 940 | 47 |

## Where it came from

- **Matching:** the `matching` config of Hessel et al. (2023), https://huggingface.co/datasets/jmhessel/newyorker_caption_contest (CC-BY-4.0). This is their cross-validation split 0, used as-is so our numbers can be compared directly with the paper's. Its train/dev/test proportions are 60/20/20 by contest.
- **Ranking:** the crowd ratings from Zhang et al. (2024), https://huggingface.co/datasets/yguooo/newyorker_caption_ranking (CC-BY-NC-4.0). This covers 362 contests (#530-#895) and about 2.18M rated captions. We use their contest-level train/validation/test split (75/12/13), so our results line up with both Zhang et al. and Zhou et al. (2025). From each contest's rating table we build 20 pairs:
  - **easy:** caption ranked *k* vs. caption ranked 999+*k*, for *k* = 1..10 (so #1-10 vs. #1000-1009)
  - **hard:** caption ranked 29+*k* vs. caption ranked 299+*k* (so #30-39 vs. #300-309)

  Which caption goes in slot A or B is random (seed 5300). A and B each end up correct about half the time (train: A 2,660, B 2,760).

## Format

The data is JSON Lines, one item per line, in `matching/{train,dev,test}.jsonl` and `ranking/{train,dev,test}.jsonl`. The `image` field is a path relative to this directory. Scene annotations come from humans for matching (Hessel et al.) and from GPT-4o for ranking (Zhang et al.).

Matching example (`label` is the letter of the true caption):
```json
{"instance_id": "415f9efeb6afe23c29fb8d3c28eac706", "contest_number": 575, "image": "images/matching/575.jpg",
 "location": "path", "description": "Two soldiers are on horseback in a field. A bunch of businessmen are trailing along behind them.",
 "uncanny": "The businessmen are in the wrong time period.", "entities": ["...Mongols", "...Businessperson"],
 "choices": ["Evolution can be so tacky.", "Let's pick up the pace. They're billing by the hour.",
             "Well, five acres of popcorn back there says you were.",
             "I'm sure it works. I was involved in the original research.", "Just water for me, thanks."],
 "label": "B"}
```

Ranking example (`label` is the caption the crowd rated higher; `mean` is the crowd score from 1 to 3):
```json
{"pair_id": "543_easy_0", "contest_number": 543, "tier": "easy",
 "caption_A": "I really just thought they would send me a lapel pin when I registered.",
 "caption_B": "Despite his attempts, Jumbo simply could not deny the two people in the room.",
 "rank_A": 1, "rank_B": 1000, "mean_A": 1.8189, "mean_B": 1.2989, "label": "A",
 "image": "images/ranking/543.jpg", "location": "a living room",
 "description": "A large elephant is standing in the middle of a living room. Two people are sitting on chairs, seemingly ignoring the elephant.",
 "uncanny": "It is highly unusual to have an elephant inside a living room.", "entities": ["Elephant", "People"]}
```

## What's in the archive

All six JSONL files are included in full. The cartoon images take up 247 MB, which is too large for Gradescope, so the archive includes only the images the **dev and test** items use (373 cartoons, 58 MB). To get the training images, rebuild everything from the Hugging Face links above with `build_m1_data.py`.
