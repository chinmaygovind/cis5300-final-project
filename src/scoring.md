For both matching and ranking we just use plain percent accuracy. For matching, we compute the percent of instances where the model's prediction matches the ground truth label, i.e. picked the caption that was actually written for that cartoon. For ranking, we compute the percent of instances where the model's prediction matches the ground truth label, i.e. picked the caption the crowd rated funnier for that cartoon.

Formally, for $N$ items with ground truth labels $y_i$ and predictions $\hat{y}_i$:

$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$$

A prediction that is missing, `null`, or anything other than a single valid letter counts as wrong. Chance is 20% for matching (5 options) and 50% for ranking (2 options). For ranking, the script also reports accuracy separately on easy pairs (crowd rank 1-10 vs. 1000-1009) and hard pairs (rank 30-39 vs. 300-309).

This is the same metric Hessel et al. (2023) use for their matching task: [Do Androids Laugh at Electric Sheep? Humor "Understanding" Benchmarks from The New Yorker Caption Contest](https://aclanthology.org/2023.acl-long.41/), ACL 2023. See also [Accuracy and precision (Wikipedia)](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_classification).

To run the evaluation, run the following command from the `Final Project` folder:

```bash
python src/score.py <labels.jsonl> <predictions.jsonl>
```

The labels file is a split from `Milestone 1/data/`. The predictions file has one JSON object per line, with the item's ID and the predicted letter, matched to the ground truth file by ID:

```
{"pair_id": "543_easy_0", "prediction": "B"}            (ranking)
{"instance_id": "415f9efe...", "prediction": "D"}        (matching)
```

`src/predict.py` writes files in this format for any model in `src/models/`.

For example, to run the evaluation for the matching task using the simple baseline, run the following commands:

```bash
python src/predict.py src/models/simple-baseline.py matching test simple_matching_test.jsonl
python src/score.py "Milestone 1/data/matching/test.jsonl" simple_matching_test.jsonl
```

```
accuracy 18.37% (n=528)
```

To run the evaluation for the ranking task using the strong baseline (GPT-4.1-nano), run the following commands:

```bash
python src/predict.py src/models/gpt-nano.py ranking test "Milestone 2/gpt-4.1-nano/ranking_test.jsonl"
python src/score.py "Milestone 1/data/ranking/test.jsonl" "Milestone 2/gpt-4.1-nano/ranking_test.jsonl"
```

```
accuracy 54.79% (n=940)
  easy: 55.74% (n=470)
  hard: 53.83% (n=470)
```
