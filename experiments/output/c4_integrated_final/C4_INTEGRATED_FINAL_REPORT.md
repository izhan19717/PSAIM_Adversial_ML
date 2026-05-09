# Integrated Positive-Demonstration Empirical Report

This report integrates the initial positive-demonstration run and the later reward-corruption/long-horizon-drift extension. All values are 10-seed means with 95% nonparametric bootstrap confidence intervals. Paired comparisons use same-seed paired bootstrap differences with 5000 resamples. Negative paired differences mean the first method listed is lower/better.

## Executive Verdict

| Claim target | Status | Principal-scientist reading |
| - | - | - |
| PSAIM beats or statistically ties SJF in at least one operational regime | Strongly supported | Duration-misreporting breaks SJF's shortest-job ordering assumption. Simplified PSAIM has significantly lower average slowdown at low, medium, and high misreport severities. |
| High-severity reward-corruption robustness from intrinsic reward | Refuted / not supported | The initial run and the later high-severity training-time sweep do not show PSAIM degradation in the desired 0-50% range. At high severity, PSAIM and DQN both collapse or are statistically indistinguishable. |
| Low-severity reward-corruption robustness | Supported | In the training-time reward sweep, PSAIM has much lower slowdown, degradation, task-failure rate, and p95 completion than DQN at low reward-corruption severity. |
| Online high reward-corruption adaptation | Partially supported | PSAIM has a statistically lower stressed slowdown than DQN, but degradation, failure-rate, and p95 differences are not decisive. |
| Long-horizon drift gate/freezing mechanisms | Weak / partial | No-gate is not worse than full PSAIM. No-freezing is directionally worse, but paired CIs cross zero. This is not strong evidence that gate/freezing are load-bearing. |

## Paper-Safe Claims

- Strong: `Under a duration-misreport distribution shift that violates SJF's job-ordering assumption, simplified PSAIM significantly outperforms SJF best-fit on average slowdown across all tested severities.`
- Strong with caveat: `The same duration-misreport regime also reduces PSAIM task-failure rate relative to SJF, although SJF retains lower p95 completion time in this proxy.`
- Moderate: `Simplified PSAIM is robust to low-severity reward corruption relative to DQN.`
- Weak/partial: `Under online high reward corruption, simplified PSAIM slightly improves slowdown relative to DQN, but the evidence is not broad across other metrics.`
- Do not claim: `Simplified PSAIM is robust to severe training-time reward corruption.`
- Do not claim: `The gate is load-bearing under long-horizon drift.`

## Initial Follow-Up: High-Severity Reward-Corruption Robustness

Hypothesis tested: PSAIM's intrinsic-reward signal is less coupled to corrupted extrinsic reward than bare DQN, so PSAIM should degrade less than DQN under severe reward corruption. The target positive demonstration was PSAIM degradation in the 0-50% range while DQN remains strongly degraded.

| method           | average_slowdown        | degradation_pct            | task_failure_rate    | p95_completion_time     |
| ---------------- | ----------------------- | -------------------------- | -------------------- | ----------------------- |
| SJF heuristic    | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| PG proxy         | 13.263 [9.882, 16.501]  | 169.915 [77.410, 263.662]  | 0.739 [0.544, 0.920] | 82.725 [70.449, 91.675] |
| DQN              | 12.801 [9.715, 15.636]  | 274.814 [188.760, 352.138] | 0.682 [0.501, 0.852] | 80.062 [69.175, 89.531] |
| DQN+RND          | 17.182 [16.693, 17.552] | 407.103 [373.721, 441.934] | 0.957 [0.914, 0.993] | 92.081 [89.875, 94.144] |
| Simplified PSAIM | 14.850 [11.986, 17.240] | 352.924 [258.504, 434.908] | 0.818 [0.653, 0.965] | 85.444 [76.453, 92.997] |


Paired comparison against DQN:

| comparison             | metric              | paired_diff_mean_ci       | status       |
| ---------------------- | ------------------- | ------------------------- | ------------ |
| Simplified PSAIM - DQN | average_slowdown    | 2.049 [-2.296, 6.265]     | not decisive |
| Simplified PSAIM - DQN | degradation_pct     | 78.111 [-63.163, 208.937] | not decisive |
| Simplified PSAIM - DQN | task_failure_rate   | 0.136 [-0.112, 0.373]     | not decisive |
| Simplified PSAIM - DQN | p95_completion_time | 5.381 [-9.216, 19.503]    | not decisive |


Reading: refuted. The initial run shows PSAIM slowdown `14.850 [11.956, 17.237]` and degradation `352.924% [256.663, 434.635]`, while DQN slowdown is `12.801 [9.728, 15.655]` and degradation `274.814% [190.368, 351.466]`. The paired PSAIM-DQN differences are not decisive, and the PSAIM degradation is far outside the desired 0-50% band.

## Initial Follow-Up: Heuristic-Breaking Duration Misreporting

Hypothesis tested: when observed job durations are biased so SJF's shortest-job assumption is violated, PSAIM should remain robust while SJF degrades.

| severity | method           | average_slowdown     | degradation_pct         | task_failure_rate    | p95_completion_time     |
| -------- | ---------------- | -------------------- | ----------------------- | -------------------- | ----------------------- |
| low      | SJF heuristic    | 5.714 [5.714, 5.714] | 35.458 [35.458, 35.458] | 0.312 [0.312, 0.312] | 39.500 [39.500, 39.500] |
| low      | Simplified PSAIM | 3.362 [3.174, 3.544] | 0.651 [-6.505, 8.321]   | 0.200 [0.188, 0.213] | 69.653 [67.097, 72.163] |
| medium   | SJF heuristic    | 6.062 [6.062, 6.062] | 43.697 [43.697, 43.697] | 0.354 [0.354, 0.354] | 44.500 [44.500, 44.500] |
| medium   | Simplified PSAIM | 3.273 [3.143, 3.418] | -2.032 [-8.083, 4.045]  | 0.193 [0.183, 0.204] | 68.319 [66.031, 70.669] |
| high     | SJF heuristic    | 6.062 [6.062, 6.062] | 43.697 [43.697, 43.697] | 0.354 [0.354, 0.354] | 44.500 [44.500, 44.500] |
| high     | Simplified PSAIM | 3.260 [3.116, 3.423] | -2.564 [-8.138, 3.449]  | 0.193 [0.183, 0.205] | 68.425 [66.106, 70.844] |


Paired comparison against SJF:

| severity | comparison                       | metric              | paired_diff_mean_ci        | status                 |
| -------- | -------------------------------- | ------------------- | -------------------------- | ---------------------- |
| low      | Simplified PSAIM - SJF heuristic | average_slowdown    | -2.352 [-2.539, -2.167]    | Simplified PSAIM lower |
| low      | Simplified PSAIM - SJF heuristic | degradation_pct     | -34.807 [-41.927, -27.122] | Simplified PSAIM lower |
| low      | Simplified PSAIM - SJF heuristic | p95_completion_time | 30.153 [27.628, 32.694]    | SJF heuristic lower    |
| low      | Simplified PSAIM - SJF heuristic | task_failure_rate   | -0.112 [-0.125, -0.099]    | Simplified PSAIM lower |
| medium   | Simplified PSAIM - SJF heuristic | average_slowdown    | -2.789 [-2.916, -2.641]    | Simplified PSAIM lower |
| medium   | Simplified PSAIM - SJF heuristic | degradation_pct     | -45.729 [-51.850, -39.599] | Simplified PSAIM lower |
| medium   | Simplified PSAIM - SJF heuristic | p95_completion_time | 23.819 [21.544, 26.156]    | SJF heuristic lower    |
| medium   | Simplified PSAIM - SJF heuristic | task_failure_rate   | -0.161 [-0.171, -0.149]    | Simplified PSAIM lower |
| high     | Simplified PSAIM - SJF heuristic | average_slowdown    | -2.802 [-2.949, -2.639]    | Simplified PSAIM lower |
| high     | Simplified PSAIM - SJF heuristic | degradation_pct     | -46.261 [-51.788, -40.362] | Simplified PSAIM lower |
| high     | Simplified PSAIM - SJF heuristic | p95_completion_time | 23.925 [21.587, 26.312]    | SJF heuristic lower    |
| high     | Simplified PSAIM - SJF heuristic | task_failure_rate   | -0.161 [-0.172, -0.150]    | Simplified PSAIM lower |


Reading: strongly supported for average slowdown and degradation. PSAIM beats SJF at all three severities with paired CIs entirely below zero. The strongest paper claim should be framed around average slowdown and failure rate. Caveat: SJF still has lower p95 completion time, so we should not claim PSAIM dominates every metric.

## Extended Follow-Up: Reward-Corruption Training-Time Sweep

This run repeats reward corruption at low, medium, and high severities with PSAIM hyperparameters unchanged.

| severity | method           | average_slowdown        | degradation_pct            | task_failure_rate    | p95_completion_time     |
| -------- | ---------------- | ----------------------- | -------------------------- | -------------------- | ----------------------- |
| low      | SJF heuristic    | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| low      | PG proxy         | 7.224 [4.371, 10.867]   | 51.975 [-21.704, 142.449]  | 0.399 [0.242, 0.597] | 68.375 [54.823, 80.725] |
| low      | DQN              | 5.387 [3.858, 7.556]    | 62.180 [17.025, 121.662]   | 0.305 [0.235, 0.412] | 76.850 [73.175, 81.350] |
| low      | DQN+RND          | 4.709 [3.875, 5.644]    | 42.753 [17.457, 72.809]    | 0.283 [0.228, 0.360] | 75.822 [71.569, 79.826] |
| low      | Simplified PSAIM | 3.322 [3.131, 3.519]    | 2.455 [-4.450, 9.824]      | 0.192 [0.183, 0.203] | 69.269 [66.587, 72.291] |
| medium   | SJF heuristic    | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| medium   | PG proxy         | 7.329 [4.651, 10.854]   | 26.288 [-6.569, 64.136]    | 0.396 [0.251, 0.589] | 67.150 [54.100, 79.351] |
| medium   | DQN              | 8.942 [7.137, 10.651]   | 175.809 [113.162, 234.596] | 0.523 [0.419, 0.622] | 77.972 [72.025, 83.775] |
| medium   | DQN+RND          | 8.124 [6.488, 10.041]   | 148.109 [93.641, 215.127]  | 0.459 [0.359, 0.575] | 74.216 [67.324, 80.766] |
| medium   | Simplified PSAIM | 11.173 [9.658, 12.620]  | 247.702 [193.465, 301.999] | 0.678 [0.600, 0.757] | 82.272 [77.763, 87.128] |
| high     | SJF heuristic    | 4.219 [4.219, 4.219]    | 0.000 [0.000, 0.000]       | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| high     | PG proxy         | 6.471 [4.869, 8.950]    | 24.002 [-14.055, 76.606]   | 0.376 [0.281, 0.498] | 76.475 [67.275, 84.300] |
| high     | DQN              | 12.788 [9.301, 16.196]  | 287.430 [179.615, 391.720] | 0.720 [0.518, 0.918] | 77.741 [65.022, 89.541] |
| high     | DQN+RND          | 14.965 [12.064, 17.338] | 350.631 [265.777, 424.036] | 0.841 [0.678, 0.972] | 84.737 [73.688, 93.356] |
| high     | Simplified PSAIM | 12.671 [9.399, 15.877]  | 290.702 [188.806, 390.030] | 0.684 [0.496, 0.863] | 77.406 [65.218, 88.769] |


Paired comparison against DQN:

| severity | comparison             | metric              | paired_diff_mean_ci         | status                 |
| -------- | ---------------------- | ------------------- | --------------------------- | ---------------------- |
| low      | Simplified PSAIM - DQN | average_slowdown    | -2.066 [-4.129, -0.634]     | Simplified PSAIM lower |
| low      | Simplified PSAIM - DQN | degradation_pct     | -59.725 [-117.129, -17.838] | Simplified PSAIM lower |
| low      | Simplified PSAIM - DQN | p95_completion_time | -7.581 [-9.813, -5.156]     | Simplified PSAIM lower |
| low      | Simplified PSAIM - DQN | task_failure_rate   | -0.113 [-0.212, -0.045]     | Simplified PSAIM lower |
| medium   | Simplified PSAIM - DQN | average_slowdown    | 2.231 [-0.580, 4.854]       | not decisive           |
| medium   | Simplified PSAIM - DQN | degradation_pct     | 71.893 [-17.806, 159.996]   | not decisive           |
| medium   | Simplified PSAIM - DQN | p95_completion_time | 4.300 [-0.294, 8.441]       | not decisive           |
| medium   | Simplified PSAIM - DQN | task_failure_rate   | 0.156 [0.003, 0.308]        | DQN lower              |
| high     | Simplified PSAIM - DQN | average_slowdown    | -0.117 [-4.628, 4.505]      | not decisive           |
| high     | Simplified PSAIM - DQN | degradation_pct     | 3.271 [-142.468, 150.252]   | not decisive           |
| high     | Simplified PSAIM - DQN | p95_completion_time | -0.334 [-17.570, 16.260]    | not decisive           |
| high     | Simplified PSAIM - DQN | task_failure_rate   | -0.035 [-0.296, 0.226]      | not decisive           |


Reading: low severity is supported; medium and high are not. At low severity, PSAIM-DQN slowdown difference is `-2.066 [-4.137, -0.653]` and degradation difference is `-59.725 [-117.389, -18.968]`. At high severity, PSAIM-DQN slowdown difference is `-0.117 [-4.754, 4.386]`, which is not decisive, and degradation remains around `290.702%` for PSAIM. This means the intrinsic signal is not enough to protect the agent from severe reward-channel poisoning.

## Extended Follow-Up: Online High Reward-Corruption Adaptation

This variant clean-trains agents and then allows online adaptation during high reward-corruption evaluation.

| severity | method           | average_slowdown      | degradation_pct         | task_failure_rate    | p95_completion_time     |
| -------- | ---------------- | --------------------- | ----------------------- | -------------------- | ----------------------- |
| high     | SJF heuristic    | 4.219 [4.219, 4.219]  | 0.000 [0.000, 0.000]    | 0.240 [0.240, 0.240] | 40.250 [40.250, 40.250] |
| high     | PG proxy         | 7.815 [5.029, 10.971] | 5.432 [0.313, 11.690]   | 0.432 [0.289, 0.590] | 71.834 [58.178, 83.985] |
| high     | DQN              | 3.649 [3.561, 3.722]  | 3.911 [-10.250, 13.779] | 0.211 [0.206, 0.216] | 70.631 [69.256, 71.778] |
| high     | DQN+RND          | 3.681 [3.579, 3.791]  | 4.298 [0.317, 8.649]    | 0.211 [0.205, 0.216] | 71.188 [69.941, 72.369] |
| high     | Simplified PSAIM | 3.557 [3.496, 3.621]  | 4.722 [-1.377, 11.003]  | 0.208 [0.204, 0.211] | 70.097 [69.250, 70.941] |


Paired comparison against DQN:

| severity | comparison             | metric              | paired_diff_mean_ci     | status                 |
| -------- | ---------------------- | ------------------- | ----------------------- | ---------------------- |
| high     | Simplified PSAIM - DQN | average_slowdown    | -0.092 [-0.168, -0.008] | Simplified PSAIM lower |
| high     | Simplified PSAIM - DQN | degradation_pct     | 0.812 [-9.648, 14.250]  | not decisive           |
| high     | Simplified PSAIM - DQN | p95_completion_time | -0.534 [-1.888, 1.147]  | not decisive           |
| high     | Simplified PSAIM - DQN | task_failure_rate   | -0.004 [-0.009, 0.002]  | not decisive           |


Reading: partially supported. PSAIM has lower slowdown than DQN with paired difference `-0.092 [-0.167, -0.012]`, but degradation, failure-rate, and p95 completion-time differences are not decisive. This is a narrow result, not a broad reward-corruption robustness result.

## Extended Follow-Up: Long-Horizon Monotonic Drift

This run uses 5x longer training exposure and a 5x longer monotonic-drift evaluation episode. It compares full simplified PSAIM against no-gate and no-freezing ablations.

| method           | average_slowdown     | degradation_pct      | task_failure_rate    | p95_completion_time        |
| ---------------- | -------------------- | -------------------- | -------------------- | -------------------------- |
| Simplified PSAIM | 8.609 [7.956, 9.126] | 0.000 [0.000, 0.000] | 0.161 [0.150, 0.172] | 143.214 [127.233, 156.815] |
| No gate          | 8.540 [7.804, 9.065] | 0.000 [0.000, 0.000] | 0.160 [0.150, 0.170] | 140.367 [121.957, 153.927] |
| No freezing      | 8.963 [8.647, 9.237] | 0.000 [0.000, 0.000] | 0.165 [0.157, 0.173] | 148.862 [141.860, 156.349] |


Paired ablation comparisons:

| comparison                     | metric              | paired_diff_mean_ci      | status       |
| ------------------------------ | ------------------- | ------------------------ | ------------ |
| No gate - Simplified PSAIM     | average_slowdown    | -0.069 [-0.807, 0.652]   | not decisive |
| No gate - Simplified PSAIM     | degradation_pct     | 0.000 [0.000, 0.000]     | not decisive |
| No gate - Simplified PSAIM     | task_failure_rate   | -0.001 [-0.010, 0.008]   | not decisive |
| No gate - Simplified PSAIM     | p95_completion_time | -2.847 [-21.231, 16.389] | not decisive |
| No freezing - Simplified PSAIM | average_slowdown    | 0.354 [-0.017, 0.921]    | not decisive |
| No freezing - Simplified PSAIM | degradation_pct     | 0.000 [0.000, 0.000]     | not decisive |
| No freezing - Simplified PSAIM | task_failure_rate   | 0.004 [-0.001, 0.011]    | not decisive |
| No freezing - Simplified PSAIM | p95_completion_time | 5.648 [-6.681, 21.982]   | not decisive |


Reading: weak/partial. No-gate is not worse than full PSAIM. No-freezing is directionally worse on slowdown (`0.354 [-0.019, 0.901]`) and p95 (`5.648 [-6.946, 21.652]`), but the CIs cross zero. We can describe this as suggestive, not conclusive.

## Why The Reward-Corruption Hypothesis Fails At High Severity

The current simplified PSAIM is not reward-corruption-aware. It adds a small intrinsic reward to the same corrupted extrinsic reward used by DQN. In the main configuration, `intrinsic_scale=0.001`, so under severe reward corruption the Bellman target is still dominated by corrupted extrinsic reward. The method estimates transition uncertainty and entropy-regime structure; it does not explicitly estimate reward-channel trustworthiness.

Therefore, the high-severity failure is not merely a bad dataset draw. It is a theory-to-implementation mismatch: severe reward corruption attacks the reward labels directly, while simplified PSAIM's implemented defenses operate mainly on state-transition uncertainty. A stronger reward-corruption claim would require an explicit reward-trust mechanism, such as reward prediction ensembles, reward-disagreement gates, SLO consistency checks, robust target clipping, or downweighting extrinsic reward when reward corruption is detected.

## Figure Package

- `fig_c4_1_initial_reward_corruption`: high-severity reward-corruption refutation.
- `fig_c4_2_duration_misreport_positive_demo`: positive demonstration against SJF.
- `fig_c4_3_reward_training_sweep`: low/medium/high reward-corruption sweep.
- `fig_c4_4_online_reward_and_long_drift`: online corruption and drift ablations.
