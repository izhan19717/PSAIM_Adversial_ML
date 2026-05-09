# PSAIM Mechanism-Control Follow-Up Experiments

All experiments use the existing single-node, two-resource proxy environment and existing simplified PSAIM hyperparameters. The simulator and duration-misreport workload generator were not changed. Values are 10-seed means with 95% nonparametric bootstrap CIs unless this was a smoke run with fewer seeds. Paired comparisons use same-seed paired bootstrap with 5000 resamples; negative paired differences mean the left method is lower/better.

## Experiment A: SJF + Static-Rejection Control

Hypothesis: PSAIM's duration-misreport advantage is mechanism-driven, not reproduced by an SJF heuristic with a uniform random rejection prior.

### Performance Table

| severity | method           | average_slowdown     | task_failure_rate    | p95_completion_time     |
| -------- | ---------------- | -------------------- | -------------------- | ----------------------- |
| low      | SJF-R(0.10)      | 4.863 [4.805, 4.924] | 0.299 [0.295, 0.304] | 62.362 [60.675, 64.081] |
| low      | SJF-R(0.17)      | 4.870 [4.770, 4.971] | 0.307 [0.303, 0.312] | 77.266 [75.944, 78.759] |
| low      | SJF-R(0.18)      | 4.851 [4.781, 4.914] | 0.309 [0.305, 0.313] | 75.344 [73.250, 77.572] |
| low      | SJF-R(0.25)      | 5.135 [5.037, 5.230] | 0.333 [0.328, 0.338] | 81.728 [81.263, 82.178] |
| low      | Simplified PSAIM | 3.384 [3.212, 3.551] | 0.202 [0.190, 0.214] | 70.222 [67.875, 72.497] |
| medium   | SJF-R(0.10)      | 5.192 [5.132, 5.254] | 0.313 [0.310, 0.316] | 65.359 [63.244, 67.391] |
| medium   | SJF-R(0.17)      | 5.203 [5.139, 5.263] | 0.323 [0.320, 0.326] | 76.700 [74.978, 78.225] |
| medium   | SJF-R(0.18)      | 5.183 [5.068, 5.305] | 0.324 [0.318, 0.331] | 78.000 [77.031, 78.963] |
| medium   | SJF-R(0.25)      | 5.487 [5.422, 5.555] | 0.349 [0.342, 0.356] | 83.047 [82.103, 83.966] |
| medium   | Simplified PSAIM | 3.179 [3.079, 3.316] | 0.186 [0.180, 0.196] | 67.003 [65.391, 69.088] |
| high     | SJF-R(0.10)      | 5.171 [5.106, 5.237] | 0.310 [0.307, 0.312] | 65.588 [63.809, 67.559] |
| high     | SJF-R(0.17)      | 5.126 [5.041, 5.215] | 0.315 [0.311, 0.318] | 78.244 [76.769, 79.819] |
| high     | SJF-R(0.18)      | 5.210 [5.155, 5.268] | 0.322 [0.321, 0.324] | 78.069 [76.244, 80.144] |
| high     | SJF-R(0.25)      | 5.476 [5.379, 5.572] | 0.341 [0.337, 0.345] | 83.669 [82.806, 84.478] |
| high     | Simplified PSAIM | 3.236 [3.093, 3.410] | 0.191 [0.181, 0.203] | 67.916 [66.150, 70.338] |

### Paired Comparison: Simplified PSAIM vs SJF-R(0.17)

| severity | comparison                     | metric            | paired_diff_mean_ci     | status      |
| -------- | ------------------------------ | ----------------- | ----------------------- | ----------- |
| low      | Simplified PSAIM - SJF-R(0.17) | average_slowdown  | -1.486 [-1.726, -1.243] | PSAIM lower |
| low      | Simplified PSAIM - SJF-R(0.17) | task_failure_rate | -0.105 [-0.120, -0.090] | PSAIM lower |
| medium   | Simplified PSAIM - SJF-R(0.17) | average_slowdown  | -2.025 [-2.127, -1.907] | PSAIM lower |
| medium   | Simplified PSAIM - SJF-R(0.17) | task_failure_rate | -0.137 [-0.145, -0.126] | PSAIM lower |
| high     | Simplified PSAIM - SJF-R(0.17) | average_slowdown  | -1.890 [-2.076, -1.699] | PSAIM lower |
| high     | Simplified PSAIM - SJF-R(0.17) | task_failure_rate | -0.124 [-0.135, -0.111] | PSAIM lower |

### Experiment A Interpretation

Status: supported. Simplified PSAIM is significantly lower than SJF-R(0.17) on both slowdown and failure rate at all three duration-misreport severities.

## Experiment B: Raw Total-Variance Ablation

Hypothesis: PSAIM's epistemic-aleatoric decomposition is responsible for its behavior beyond a simpler total-variance penalty.

### Downstream Duration-Misreport Performance

| severity | method                 | average_slowdown     | task_failure_rate    | p95_completion_time     |
| -------- | ---------------------- | -------------------- | -------------------- | ----------------------- |
| low      | PSAIM-RawVar           | 3.139 [3.066, 3.236] | 0.183 [0.180, 0.186] | 66.428 [65.606, 67.688] |
| low      | SJF best-fit heuristic | 5.714 [5.714, 5.714] | 0.312 [0.312, 0.312] | 39.500 [39.500, 39.500] |
| low      | Simplified PSAIM       | 3.384 [3.212, 3.551] | 0.202 [0.190, 0.214] | 70.222 [67.875, 72.497] |
| medium   | PSAIM-RawVar           | 3.150 [3.056, 3.282] | 0.185 [0.180, 0.193] | 66.991 [65.472, 69.019] |
| medium   | SJF best-fit heuristic | 6.062 [6.062, 6.062] | 0.354 [0.354, 0.354] | 44.500 [44.500, 44.500] |
| medium   | Simplified PSAIM       | 3.179 [3.079, 3.316] | 0.186 [0.180, 0.196] | 67.003 [65.391, 69.088] |
| high     | PSAIM-RawVar           | 3.217 [3.068, 3.386] | 0.189 [0.180, 0.200] | 67.803 [65.650, 70.078] |
| high     | SJF best-fit heuristic | 6.062 [6.062, 6.062] | 0.354 [0.354, 0.354] | 44.500 [44.500, 44.500] |
| high     | Simplified PSAIM       | 3.236 [3.093, 3.410] | 0.191 [0.181, 0.203] | 67.916 [66.150, 70.338] |

### Action-Distribution Audit

| severity | method                 | allocate_rate        | defer_rate           | reject_rate          |
| -------- | ---------------------- | -------------------- | -------------------- | -------------------- |
| low      | PSAIM-RawVar           | 0.838 [0.834, 0.841] | 0.003 [0.002, 0.004] | 0.159 [0.156, 0.163] |
| low      | SJF best-fit heuristic | 0.927 [0.927, 0.927] | 0.073 [0.073, 0.073] | 0.000 [0.000, 0.000] |
| low      | Simplified PSAIM       | 0.818 [0.807, 0.831] | 0.004 [0.003, 0.005] | 0.178 [0.166, 0.189] |
| medium   | PSAIM-RawVar           | 0.834 [0.825, 0.840] | 0.003 [0.002, 0.005] | 0.163 [0.157, 0.172] |
| medium   | SJF best-fit heuristic | 0.917 [0.917, 0.917] | 0.083 [0.083, 0.083] | 0.000 [0.000, 0.000] |
| medium   | Simplified PSAIM       | 0.834 [0.823, 0.840] | 0.004 [0.002, 0.005] | 0.163 [0.156, 0.174] |
| high     | PSAIM-RawVar           | 0.831 [0.819, 0.840] | 0.004 [0.002, 0.005] | 0.166 [0.157, 0.177] |
| high     | SJF best-fit heuristic | 0.917 [0.917, 0.917] | 0.083 [0.083, 0.083] | 0.000 [0.000, 0.000] |
| high     | Simplified PSAIM       | 0.830 [0.816, 0.840] | 0.003 [0.002, 0.005] | 0.167 [0.157, 0.180] |

### Paired Comparison: PSAIM-RawVar vs Simplified PSAIM

| severity | comparison                      | metric           | paired_diff_mean_ci     | status                 |
| -------- | ------------------------------- | ---------------- | ----------------------- | ---------------------- |
| low      | PSAIM-RawVar - Simplified PSAIM | average_slowdown | -0.246 [-0.444, -0.029] | RawVar lower           |
| medium   | PSAIM-RawVar - Simplified PSAIM | average_slowdown | -0.029 [-0.212, 0.161]  | draw / CI crosses zero |
| high     | PSAIM-RawVar - Simplified PSAIM | average_slowdown | -0.019 [-0.278, 0.236]  | draw / CI crosses zero |
| low      | PSAIM-RawVar - Simplified PSAIM | reject_rate      | -0.019 [-0.032, -0.005] | RawVar lower           |
| medium   | PSAIM-RawVar - Simplified PSAIM | reject_rate      | -0.000 [-0.016, 0.015]  | draw / CI crosses zero |
| high     | PSAIM-RawVar - Simplified PSAIM | reject_rate      | -0.001 [-0.019, 0.016]  | draw / CI crosses zero |

### Held-Out Signal-Separation Summary

| method           | regime       | metric  | value                    |
| ---------------- | ------------ | ------- | ------------------------ |
| PSAIM-RawVar     | high_entropy | V_ale   | NA                       |
| PSAIM-RawVar     | low_entropy  | V_ale   | NA                       |
| PSAIM-RawVar     | high_entropy | V_epi   | NA                       |
| PSAIM-RawVar     | low_entropy  | V_epi   | NA                       |
| PSAIM-RawVar     | high_entropy | gate_h3 | NA                       |
| PSAIM-RawVar     | low_entropy  | gate_h3 | NA                       |
| PSAIM-RawVar     | high_entropy | r_int   | -9.152 [-10.378, -7.733] |
| PSAIM-RawVar     | low_entropy  | r_int   | -8.211 [-9.448, -6.712]  |
| PSAIM-RawVar     | high_entropy | raw_var | 1.552 [0.990, 2.177]     |
| PSAIM-RawVar     | low_entropy  | raw_var | 1.024 [0.672, 1.384]     |
| Simplified PSAIM | high_entropy | V_ale   | 0.072 [0.071, 0.073]     |
| Simplified PSAIM | low_entropy  | V_ale   | 0.059 [0.058, 0.060]     |
| Simplified PSAIM | high_entropy | V_epi   | 0.075 [0.069, 0.082]     |
| Simplified PSAIM | low_entropy  | V_epi   | 0.035 [0.030, 0.041]     |
| Simplified PSAIM | high_entropy | gate_h3 | 0.799 [0.777, 0.819]     |
| Simplified PSAIM | low_entropy  | gate_h3 | 0.906 [0.888, 0.919]     |
| Simplified PSAIM | high_entropy | r_int   | -0.056 [-0.088, -0.028]  |
| Simplified PSAIM | low_entropy  | r_int   | -0.009 [-0.026, 0.003]   |
| Simplified PSAIM | high_entropy | raw_var | NA                       |
| Simplified PSAIM | low_entropy  | raw_var | NA                       |

### Intrinsic-Reward Switch Check

| method           | low_entropy_r_int_mean | high_entropy_r_int_mean | paired_high_minus_low_ci | surprise_agnostic_switch |
| ---------------- | ---------------------- | ----------------------- | ------------------------ | ------------------------ |
| Simplified PSAIM | -0.009                 | -0.056                  | -0.046 [-0.065, -0.030]  | no                       |
| PSAIM-RawVar     | -8.211                 | -9.152                  | -0.941 [-1.121, -0.771]  | no                       |

### Experiment B Interpretation

Status: mixed and claim-limiting. Simplified PSAIM itself does not show the strict positive-low / negative-high intrinsic-reward sign switch in this mechanism-control signal audit, so this run cannot support the paper's strict sign-switch wording. PSAIM-RawVar matches or improves the downstream slowdown/rejection behavior, which means the explicit decomposition is not shown to be operationally load-bearing for the duration-misreport advantage under this proxy.

## Protocol Notes

- SJF-R(p) rejects the head-of-queue task with probability p whenever the queue is non-empty; otherwise it delegates to the existing SJF best-fit heuristic. The rejection coin flip is uniform random and does not inspect state features.
- PSAIM-RawVar uses the same K=5 scalar Q-head ensemble and main PSAIM lambda/sigma0 values, but replaces the intrinsic reward with `-lambda * log(1 + Var(Y_k)/sigma0^2)` and uses no gate or epistemic/aleatoric decomposition.
- The RawVar implementation computes `Y_k = gamma * max_a Q_k(S', a)` over valid actions for the observed next state. No extra next-state sampler or simulator branch was introduced.
- Draws are reported as draws; no directional claim should be made when the paired CI crosses zero.
