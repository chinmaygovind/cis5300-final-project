# Dev-only caption-order check

Examples: 880; valid in both orders: 880.

Original accuracy: 50.80%.

Swapped accuracy (gold labels flipped): 53.41%.

Same caption selected: 159 pairs.

Same letter selected (different caption): 721 pairs.

B selections: original 812/880; swapped 789/880.

A->B and B->A preserve the chosen caption; A->A and B->B change the chosen caption. Inspect order-swap-summary.json for transition counts, percentages, and tier results. This is a dev diagnostic, not a replacement test score or a calibrated model.
