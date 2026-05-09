# Positive-Demonstration Follow-Up Experiments

All reported intervals are 95% nonparametric bootstrap confidence intervals over 10 seeds unless otherwise noted. Paired comparisons use same-seed paired bootstrap differences with 5000 resamples. Negative paired differences mean simplified PSAIM is lower/better.

## Scenario Slowdown Table

| experiment                   | scenario           | severity | method                 | average_slowdown        | degradation_pct            | task_failure_rate    | p95_completion_time     |
| ---------------------------- | ------------------ | -------- | ---------------------- | ----------------------- | -------------------------- | -------------------- | ----------------------- |
| heuristic_breaking_shift     | clean              | clean    | SJF best-fit heuristic | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| heuristic_breaking_shift     | clean              | clean    | Simplified PSAIM       | 3.360 [3.197, 3.519]    | 0.000 [0.000, 0.000]       | 0.199 [0.188, 0.210] | 70.013 [67.637, 72.325] |
| heuristic_breaking_shift     | duration_misreport | high     | SJF best-fit heuristic | 6.062 [6.062, 6.062]    | 43.697 [43.697, 43.697]    | 0.354 [0.354, 0.354] | 44.500 [44.500, 44.500] |
| heuristic_breaking_shift     | duration_misreport | high     | Simplified PSAIM       | 3.260 [3.115, 3.425]    | -2.564 [-7.944, 3.361]     | 0.193 [0.183, 0.205] | 68.425 [66.100, 70.925] |
| heuristic_breaking_shift     | duration_misreport | low      | SJF best-fit heuristic | 5.714 [5.714, 5.714]    | 35.458 [35.458, 35.458]    | 0.312 [0.312, 0.312] | 39.500 [39.500, 39.500] |
| heuristic_breaking_shift     | duration_misreport | low      | Simplified PSAIM       | 3.362 [3.181, 3.546]    | 0.651 [-6.481, 8.196]      | 0.200 [0.188, 0.213] | 69.653 [67.137, 72.156] |
| heuristic_breaking_shift     | duration_misreport | medium   | SJF best-fit heuristic | 6.062 [6.062, 6.062]    | 43.697 [43.697, 43.697]    | 0.354 [0.354, 0.354] | 44.500 [44.500, 44.500] |
| heuristic_breaking_shift     | duration_misreport | medium   | Simplified PSAIM       | 3.273 [3.143, 3.424]    | -2.032 [-8.352, 4.288]     | 0.193 [0.183, 0.205] | 68.319 [66.081, 70.713] |
| reward_corruption_robustness | clean              | clean    | DQN                    | 3.392 [3.235, 3.545]    | 0.000 [0.000, 0.000]       | 0.198 [0.189, 0.208] | 70.697 [68.188, 73.050] |
| reward_corruption_robustness | clean              | clean    | DQN+RND                | 3.419 [3.235, 3.609]    | 0.000 [0.000, 0.000]       | 0.202 [0.191, 0.214] | 70.234 [68.134, 72.206] |
| reward_corruption_robustness | clean              | clean    | PG proxy               | 5.993 [4.304, 8.466]    | 0.000 [0.000, 0.000]       | 0.343 [0.241, 0.481] | 68.175 [55.624, 79.600] |
| reward_corruption_robustness | clean              | clean    | SJF best-fit heuristic | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| reward_corruption_robustness | clean              | clean    | Simplified PSAIM       | 3.344 [3.152, 3.528]    | 0.000 [0.000, 0.000]       | 0.198 [0.186, 0.210] | 69.591 [66.953, 72.213] |
| reward_corruption_robustness | reward_corruption  | high     | DQN                    | 12.801 [9.728, 15.655]  | 274.814 [190.368, 351.466] | 0.682 [0.503, 0.860] | 80.062 [69.361, 89.485] |
| reward_corruption_robustness | reward_corruption  | high     | DQN+RND                | 17.182 [16.678, 17.553] | 407.103 [372.378, 440.624] | 0.957 [0.913, 0.993] | 92.081 [89.572, 94.141] |
| reward_corruption_robustness | reward_corruption  | high     | PG proxy               | 13.263 [9.962, 16.501]  | 169.915 [81.716, 265.000]  | 0.739 [0.546, 0.919] | 82.725 [70.773, 91.500] |
| reward_corruption_robustness | reward_corruption  | high     | SJF best-fit heuristic | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| reward_corruption_robustness | reward_corruption  | high     | Simplified PSAIM       | 14.850 [11.956, 17.237] | 352.924 [256.663, 434.635] | 0.818 [0.651, 0.966] | 85.444 [76.243, 92.984] |

## Reward-Corruption Paired Comparison

| comparison             | metric              | paired_diff_mean_ci       | status       |
| ---------------------- | ------------------- | ------------------------- | ------------ |
| Simplified PSAIM - DQN | average_slowdown    | 2.049 [-2.424, 6.194]     | not decisive |
| Simplified PSAIM - DQN | degradation_pct     | 78.111 [-64.821, 206.164] | not decisive |
| Simplified PSAIM - DQN | task_failure_rate   | 0.136 [-0.122, 0.373]     | not decisive |
| Simplified PSAIM - DQN | p95_completion_time | 5.381 [-9.697, 19.425]    | not decisive |

## Duration-Misreport Paired Comparison

| severity | comparison             | metric           | paired_diff_mean_ci        | status      |
| -------- | ---------------------- | ---------------- | -------------------------- | ----------- |
| low      | Simplified PSAIM - SJF | average_slowdown | -2.352 [-2.538, -2.170]    | PSAIM lower |
| low      | Simplified PSAIM - SJF | degradation_pct  | -34.807 [-41.848, -27.322] | PSAIM lower |
| medium   | Simplified PSAIM - SJF | average_slowdown | -2.789 [-2.917, -2.645]    | PSAIM lower |
| medium   | Simplified PSAIM - SJF | degradation_pct  | -45.729 [-51.911, -39.465] | PSAIM lower |
| high     | Simplified PSAIM - SJF | average_slowdown | -2.802 [-2.944, -2.636]    | PSAIM lower |
| high     | Simplified PSAIM - SJF | degradation_pct  | -46.261 [-51.709, -40.302] | PSAIM lower |

## Hypothesis Status

- Reward-corruption robustness: refuted. Hypothesis: simplified PSAIM should degrade less than DQN under high reward corruption. Interpretation: simplified PSAIM mean degradation was 352.9% versus DQN 274.8%.
- Heuristic-breaking duration misreporting: supported. Hypothesis: simplified PSAIM should beat or statistically tie SJF when reported durations invert the SJF ordering. Interpretation: low: PSAIM lower; medium: PSAIM lower; high: PSAIM lower.
- Long-horizon regime drift: skipped in this positive-demonstration bundle because the 5x training and 5x evaluation protocol is substantially more expensive than the first two priority experiments. No result is claimed for the freezing/gate long-horizon hypothesis.

## Protocol Notes

- PSAIM hyperparameters were not changed from the main experiments.
- The duration-misreport stressor changes only the reported duration feature in observations; true job durations, workload generation, execution dynamics, and completion-time metrics are unchanged.
- The SJF best-fit heuristic and simplified PSAIM are evaluated on the same single-node, two-resource proxy environment.
