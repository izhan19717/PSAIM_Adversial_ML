# Duration-Misreport Action Distribution Audit

This focused audit reruns the duration-misreport setup with action-type logging. Values are 10-seed means with 95% bootstrap CIs over seed-level action rates. Paired differences use same-seed paired bootstrap. The PSAIM hyperparameters and proxy environment are unchanged.

## Action-Type Rates

| severity | method                 | allocate_rate        | defer_rate           | reject_rate          |
| -------- | ---------------------- | -------------------- | -------------------- | -------------------- |
| clean    | SJF best-fit heuristic | 0.917 [0.917, 0.917] | 0.083 [0.083, 0.083] | 0.000 [0.000, 0.000] |
| clean    | Simplified PSAIM       | 0.814 [0.802, 0.827] | 0.004 [0.002, 0.006] | 0.182 [0.170, 0.195] |
| low      | SJF best-fit heuristic | 0.927 [0.927, 0.927] | 0.073 [0.073, 0.073] | 0.000 [0.000, 0.000] |
| low      | Simplified PSAIM       | 0.829 [0.817, 0.839] | 0.003 [0.002, 0.003] | 0.168 [0.159, 0.180] |
| medium   | SJF best-fit heuristic | 0.917 [0.917, 0.917] | 0.083 [0.083, 0.083] | 0.000 [0.000, 0.000] |
| medium   | Simplified PSAIM       | 0.826 [0.814, 0.836] | 0.003 [0.002, 0.004] | 0.171 [0.161, 0.183] |
| high     | SJF best-fit heuristic | 0.917 [0.917, 0.917] | 0.083 [0.083, 0.083] | 0.000 [0.000, 0.000] |
| high     | Simplified PSAIM       | 0.835 [0.824, 0.841] | 0.003 [0.002, 0.004] | 0.162 [0.156, 0.173] |

## Paired Differences

| severity | comparison                       | metric        | paired_diff_mean_ci     |
| -------- | -------------------------------- | ------------- | ----------------------- |
| clean    | Simplified PSAIM - SJF heuristic | allocate_rate | -0.103 [-0.114, -0.090] |
| clean    | Simplified PSAIM - SJF heuristic | defer_rate    | -0.080 [-0.081, -0.077] |
| clean    | Simplified PSAIM - SJF heuristic | reject_rate   | 0.182 [0.170, 0.194]    |
| low      | Simplified PSAIM - SJF heuristic | allocate_rate | -0.098 [-0.110, -0.089] |
| low      | Simplified PSAIM - SJF heuristic | defer_rate    | -0.070 [-0.071, -0.070] |
| low      | Simplified PSAIM - SJF heuristic | reject_rate   | 0.168 [0.159, 0.180]    |
| medium   | Simplified PSAIM - SJF heuristic | allocate_rate | -0.091 [-0.102, -0.081] |
| medium   | Simplified PSAIM - SJF heuristic | defer_rate    | -0.080 [-0.081, -0.080] |
| medium   | Simplified PSAIM - SJF heuristic | reject_rate   | 0.171 [0.161, 0.183]    |
| high     | Simplified PSAIM - SJF heuristic | allocate_rate | -0.082 [-0.092, -0.076] |
| high     | Simplified PSAIM - SJF heuristic | defer_rate    | -0.080 [-0.082, -0.079] |
| high     | Simplified PSAIM - SJF heuristic | reject_rate   | 0.162 [0.156, 0.173]    |
