# C4 Reward-Corruption and Long-Horizon Drift Extension

All intervals are 95% bootstrap intervals over 10 seeds. Paired comparisons use same-seed paired bootstrap with 5000 resamples. PSAIM hyperparameters are unchanged from the main experiments.

## Summary Table

| experiment                          | scenario                   | severity     | method                        | average_slowdown        | degradation_pct            | task_failure_rate    | p95_completion_time        |
| ----------------------------------- | -------------------------- | ------------ | ----------------------------- | ----------------------- | -------------------------- | -------------------- | -------------------------- |
| long_horizon_monotonic_drift        | monotonic_drift            | long_horizon | Simplified PSAIM              | 8.609 [7.938, 9.125]    | 0.000 [0.000, 0.000]       | 0.161 [0.150, 0.172] | 143.214 [127.250, 156.314] |
| long_horizon_monotonic_drift        | monotonic_drift            | long_horizon | Simplified PSAIM, no freezing | 8.963 [8.647, 9.239]    | 0.000 [0.000, 0.000]       | 0.165 [0.157, 0.173] | 148.862 [141.847, 156.342] |
| long_horizon_monotonic_drift        | monotonic_drift            | long_horizon | Simplified PSAIM, no gate     | 8.540 [7.831, 9.074]    | 0.000 [0.000, 0.000]       | 0.160 [0.150, 0.170] | 140.367 [122.235, 154.125] |
| reward_corruption_online_adaptation | clean                      | clean        | DQN                           | 3.708 [3.255, 4.446]    | 0.000 [0.000, 0.000]       | 0.209 [0.188, 0.240] | 71.394 [68.559, 74.306]    |
| reward_corruption_online_adaptation | clean                      | clean        | DQN+RND                       | 3.538 [3.411, 3.634]    | 0.000 [0.000, 0.000]       | 0.210 [0.201, 0.218] | 72.584 [70.928, 73.547]    |
| reward_corruption_online_adaptation | clean                      | clean        | PG proxy                      | 7.356 [4.949, 10.135]   | 0.000 [0.000, 0.000]       | 0.394 [0.273, 0.540] | 72.225 [58.250, 84.775]    |
| reward_corruption_online_adaptation | clean                      | clean        | SJF best-fit heuristic        | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250]    |
| reward_corruption_online_adaptation | clean                      | clean        | Simplified PSAIM              | 3.423 [3.246, 3.597]    | 0.000 [0.000, 0.000]       | 0.204 [0.192, 0.216] | 70.338 [67.750, 72.888]    |
| reward_corruption_online_adaptation | reward_corruption_online   | high         | DQN                           | 3.649 [3.565, 3.724]    | 3.911 [-10.180, 13.959]    | 0.211 [0.206, 0.216] | 70.631 [69.338, 71.747]    |
| reward_corruption_online_adaptation | reward_corruption_online   | high         | DQN+RND                       | 3.681 [3.579, 3.789]    | 4.298 [0.374, 8.639]       | 0.211 [0.205, 0.216] | 71.188 [69.928, 72.319]    |
| reward_corruption_online_adaptation | reward_corruption_online   | high         | PG proxy                      | 7.815 [5.115, 10.884]   | 5.432 [0.434, 11.544]      | 0.432 [0.290, 0.587] | 71.834 [57.953, 84.335]    |
| reward_corruption_online_adaptation | reward_corruption_online   | high         | SJF best-fit heuristic        | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250]    |
| reward_corruption_online_adaptation | reward_corruption_online   | high         | Simplified PSAIM              | 3.557 [3.496, 3.622]    | 4.722 [-1.516, 10.989]     | 0.208 [0.204, 0.211] | 70.097 [69.263, 70.969]    |
| reward_corruption_training_sweep    | clean                      | clean        | DQN                           | 3.309 [3.171, 3.463]    | 0.000 [0.000, 0.000]       | 0.194 [0.185, 0.206] | 69.959 [68.022, 71.916]    |
| reward_corruption_training_sweep    | clean                      | clean        | DQN+RND                       | 3.311 [3.167, 3.463]    | 0.000 [0.000, 0.000]       | 0.193 [0.185, 0.203] | 70.122 [68.003, 72.288]    |
| reward_corruption_training_sweep    | clean                      | clean        | PG proxy                      | 5.839 [4.497, 7.363]    | 0.000 [0.000, 0.000]       | 0.341 [0.260, 0.449] | 69.950 [56.649, 80.050]    |
| reward_corruption_training_sweep    | clean                      | clean        | SJF best-fit heuristic        | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250]    |
| reward_corruption_training_sweep    | clean                      | clean        | Simplified PSAIM              | 3.258 [3.108, 3.424]    | 0.000 [0.000, 0.000]       | 0.192 [0.183, 0.203] | 68.312 [66.269, 70.603]    |
| reward_corruption_training_sweep    | reward_corruption_training | high         | DQN                           | 12.788 [9.327, 16.188]  | 287.430 [181.909, 389.928] | 0.720 [0.519, 0.918] | 77.741 [65.350, 89.491]    |
| reward_corruption_training_sweep    | reward_corruption_training | high         | DQN+RND                       | 14.965 [11.945, 17.396] | 350.631 [262.619, 424.143] | 0.841 [0.672, 0.978] | 84.737 [73.337, 93.334]    |
| reward_corruption_training_sweep    | reward_corruption_training | high         | PG proxy                      | 6.471 [4.863, 8.837]    | 24.002 [-13.681, 75.108]   | 0.376 [0.282, 0.496] | 76.475 [67.449, 84.600]    |
| reward_corruption_training_sweep    | reward_corruption_training | high         | SJF best-fit heuristic        | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250]    |
| reward_corruption_training_sweep    | reward_corruption_training | high         | Simplified PSAIM              | 12.671 [9.388, 15.947]  | 290.702 [188.380, 388.128] | 0.684 [0.496, 0.867] | 77.406 [64.737, 88.691]    |
| reward_corruption_training_sweep    | reward_corruption_training | low          | DQN                           | 5.387 [3.838, 7.497]    | 62.180 [16.001, 120.480]   | 0.305 [0.234, 0.409] | 76.850 [73.175, 81.338]    |
| reward_corruption_training_sweep    | reward_corruption_training | low          | DQN+RND                       | 4.709 [3.892, 5.613]    | 42.753 [17.648, 70.966]    | 0.283 [0.229, 0.358] | 75.822 [71.831, 79.744]    |
| reward_corruption_training_sweep    | reward_corruption_training | low          | PG proxy                      | 7.224 [4.394, 10.930]   | 51.975 [-22.018, 142.726]  | 0.399 [0.244, 0.598] | 68.375 [54.675, 80.850]    |
| reward_corruption_training_sweep    | reward_corruption_training | low          | SJF best-fit heuristic        | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250]    |
| reward_corruption_training_sweep    | reward_corruption_training | low          | Simplified PSAIM              | 3.322 [3.134, 3.518]    | 2.455 [-4.317, 10.052]     | 0.192 [0.183, 0.203] | 69.269 [66.631, 72.175]    |
| reward_corruption_training_sweep    | reward_corruption_training | medium       | DQN                           | 8.942 [7.198, 10.635]   | 175.809 [114.571, 234.129] | 0.523 [0.422, 0.620] | 77.972 [71.897, 83.816]    |
| reward_corruption_training_sweep    | reward_corruption_training | medium       | DQN+RND                       | 8.124 [6.498, 10.071]   | 148.109 [93.730, 213.606]  | 0.459 [0.358, 0.577] | 74.216 [67.550, 80.841]    |
| reward_corruption_training_sweep    | reward_corruption_training | medium       | PG proxy                      | 7.329 [4.691, 10.823]   | 26.288 [-7.075, 63.221]    | 0.396 [0.252, 0.585] | 67.150 [53.499, 79.051]    |
| reward_corruption_training_sweep    | reward_corruption_training | medium       | SJF best-fit heuristic        | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250]    |
| reward_corruption_training_sweep    | reward_corruption_training | medium       | Simplified PSAIM              | 11.173 [9.672, 12.614]  | 247.702 [192.926, 301.992] | 0.678 [0.601, 0.757] | 82.272 [77.834, 87.144]    |

## Reward-Corruption Training-Time Pairing

| experiment                       | scenario                   | severity | comparison             | metric              | paired_diff_mean_ci         | status                 |
| -------------------------------- | -------------------------- | -------- | ---------------------- | ------------------- | --------------------------- | ---------------------- |
| reward_corruption_training_sweep | reward_corruption_training | high     | Simplified PSAIM - DQN | average_slowdown    | -0.117 [-4.754, 4.386]      | not decisive           |
| reward_corruption_training_sweep | reward_corruption_training | high     | Simplified PSAIM - DQN | degradation_pct     | 3.271 [-146.010, 148.536]   | not decisive           |
| reward_corruption_training_sweep | reward_corruption_training | high     | Simplified PSAIM - DQN | task_failure_rate   | -0.035 [-0.300, 0.219]      | not decisive           |
| reward_corruption_training_sweep | reward_corruption_training | high     | Simplified PSAIM - DQN | p95_completion_time | -0.334 [-18.095, 16.125]    | not decisive           |
| reward_corruption_training_sweep | reward_corruption_training | low      | Simplified PSAIM - DQN | average_slowdown    | -2.066 [-4.137, -0.653]     | Simplified PSAIM lower |
| reward_corruption_training_sweep | reward_corruption_training | low      | Simplified PSAIM - DQN | degradation_pct     | -59.725 [-117.389, -18.968] | Simplified PSAIM lower |
| reward_corruption_training_sweep | reward_corruption_training | low      | Simplified PSAIM - DQN | task_failure_rate   | -0.113 [-0.215, -0.046]     | Simplified PSAIM lower |
| reward_corruption_training_sweep | reward_corruption_training | low      | Simplified PSAIM - DQN | p95_completion_time | -7.581 [-9.959, -5.206]     | Simplified PSAIM lower |
| reward_corruption_training_sweep | reward_corruption_training | medium   | Simplified PSAIM - DQN | average_slowdown    | 2.231 [-0.559, 4.790]       | not decisive           |
| reward_corruption_training_sweep | reward_corruption_training | medium   | Simplified PSAIM - DQN | degradation_pct     | 71.893 [-17.337, 158.161]   | not decisive           |
| reward_corruption_training_sweep | reward_corruption_training | medium   | Simplified PSAIM - DQN | task_failure_rate   | 0.156 [0.006, 0.303]        | DQN lower              |
| reward_corruption_training_sweep | reward_corruption_training | medium   | Simplified PSAIM - DQN | p95_completion_time | 4.300 [-0.316, 8.456]       | not decisive           |

## Reward-Corruption Online-Adaptation Pairing

| experiment                          | scenario                 | severity | comparison             | metric              | paired_diff_mean_ci     | status                 |
| ----------------------------------- | ------------------------ | -------- | ---------------------- | ------------------- | ----------------------- | ---------------------- |
| reward_corruption_online_adaptation | reward_corruption_online | high     | Simplified PSAIM - DQN | average_slowdown    | -0.092 [-0.167, -0.012] | Simplified PSAIM lower |
| reward_corruption_online_adaptation | reward_corruption_online | high     | Simplified PSAIM - DQN | degradation_pct     | 0.812 [-9.835, 13.499]  | not decisive           |
| reward_corruption_online_adaptation | reward_corruption_online | high     | Simplified PSAIM - DQN | task_failure_rate   | -0.004 [-0.009, 0.002]  | not decisive           |
| reward_corruption_online_adaptation | reward_corruption_online | high     | Simplified PSAIM - DQN | p95_completion_time | -0.534 [-1.853, 1.066]  | not decisive           |

## Long-Horizon Drift Pairing

| experiment                   | scenario        | severity     | comparison                                       | metric              | paired_diff_mean_ci      | status       |
| ---------------------------- | --------------- | ------------ | ------------------------------------------------ | ------------------- | ------------------------ | ------------ |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no gate - Simplified PSAIM     | average_slowdown    | -0.069 [-0.815, 0.628]   | not decisive |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no gate - Simplified PSAIM     | degradation_pct     | 0.000 [0.000, 0.000]     | not decisive |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no gate - Simplified PSAIM     | task_failure_rate   | -0.001 [-0.011, 0.008]   | not decisive |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no gate - Simplified PSAIM     | p95_completion_time | -2.847 [-21.311, 15.966] | not decisive |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no freezing - Simplified PSAIM | average_slowdown    | 0.354 [-0.019, 0.901]    | not decisive |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no freezing - Simplified PSAIM | degradation_pct     | 0.000 [0.000, 0.000]     | not decisive |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no freezing - Simplified PSAIM | task_failure_rate   | 0.004 [-0.001, 0.011]    | not decisive |
| long_horizon_monotonic_drift | monotonic_drift | long_horizon | Simplified PSAIM, no freezing - Simplified PSAIM | p95_completion_time | 5.648 [-6.946, 21.652]   | not decisive |

## Hypothesis Status

- reward_corruption_training_sweep: high: refuted, PSAIM degradation=290.7%, DQN degradation=287.4%, paired PSAIM-DQN=3.3% [-146.0, 148.5]; low: supported, PSAIM degradation=2.5%, DQN degradation=62.2%, paired PSAIM-DQN=-59.7% [-117.4, -19.0]; medium: refuted, PSAIM degradation=247.7%, DQN degradation=175.8%, paired PSAIM-DQN=71.9% [-17.3, 158.2].
- reward_corruption_online_adaptation: high: refuted, PSAIM degradation=4.7%, DQN degradation=3.9%, paired PSAIM-DQN=0.8% [-9.8, 13.5].
- Long-horizon drift: Simplified PSAIM, no gate minus full=-0.069 [-0.815, 0.628] (refuted); Simplified PSAIM, no freezing minus full=0.354 [-0.019, 0.901] (partially supported).

## Protocol Notes

- Training-time reward corruption trains learned agents with delayed biased rewards, then evaluates task metrics without using corrupted task metrics as reported outcomes.
- Online-adaptation reward corruption clean-trains learned agents and then allows online updates during evaluation using corrupted reward feedback.
- Monotonic drift uses the same proxy allocator and workload generator with an added drift schedule: the probability of high-entropy arrivals increases linearly over the episode.