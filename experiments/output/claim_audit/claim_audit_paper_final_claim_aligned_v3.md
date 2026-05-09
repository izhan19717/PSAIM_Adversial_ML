# Claim Evidence Audit

## Experiment 1 Stress Evidence

| method | scenario | high-severity degradation mean [95% CI] | claim status |
| - | - | - | - |
| PG proxy | observation_noise | 41.333 [23.059, 61.522] | SUPPORTED |
| PG proxy | reward_corruption | 135.152 [54.675, 217.160] | SUPPORTED |
| PG proxy | distribution_shift | 85.704 [43.123, 129.937] | SUPPORTED |
| PG proxy | co_tenant_interference | 51.714 [34.393, 68.724] | SUPPORTED |
| DQN | observation_noise | 86.548 [78.113, 95.745] | SUPPORTED |
| DQN | reward_corruption | 349.633 [282.631, 410.513] | SUPPORTED |
| DQN | distribution_shift | 203.368 [190.277, 216.418] | SUPPORTED |
| DQN | co_tenant_interference | 79.404 [71.088, 87.949] | SUPPORTED |

Reviewer rule used here: a perturbation is strongly supported only when the 95% bootstrap CI for high-severity degradation is entirely above zero. Positive mean with a crossing CI is marked partial.

## Experiment 2 Downstream Evidence

| method | stressed slowdown mean [95% CI] |
| - | - |
| heuristic | 2.412 [2.177, 2.701] |
| PG proxy | 3.731 [3.007, 4.816] |
| DQN | 3.954 [3.201, 4.889] |
| DQN+RND | 3.921 [3.032, 5.175] |
| simplified PSAIM | 3.303 [3.109, 3.516] |
| simplified PSAIM, no aleatoric penalty | 4.461 [3.003, 7.040] |
| simplified PSAIM, no gate | 3.316 [3.118, 3.561] |
| simplified PSAIM, no freezing | 3.337 [3.116, 3.593] |

| comparison | PSAIM slowdown minus comparator [95% CI] | superiority status |
| - | - | - |
| simplified PSAIM vs heuristic | 0.891 [0.732, 1.030] | UNSUPPORTED |
| simplified PSAIM vs PG proxy | -0.428 [-1.529, 0.228] | WEAK / PARTIAL |
| simplified PSAIM vs DQN | -0.651 [-1.627, -0.001] | SUPPORTED |
| simplified PSAIM vs DQN+RND | -0.618 [-1.867, 0.186] | WEAK / PARTIAL |

Negative paired differences mean simplified PSAIM is better. The stronger statistical-superiority claim requires the entire paired bootstrap CI to be below zero.

## Held-Out PSAIM Signal Evidence

| metric | low entropy mean [95% CI] | high entropy mean [95% CI] | high-low paired/pooled direction |
| - | - | - | - |
| V_epi | 0.025 [0.024, 0.027] | 0.064 [0.062, 0.066] | high > low |
| V_ale | 0.057 [0.057, 0.058] | 0.072 [0.072, 0.073] | high > low |
| V_ale_excess | 0.003 [0.003, 0.003] | 0.008 [0.008, 0.009] | high > low |
| gate_h3 | 0.930 [0.928, 0.932] | 0.829 [0.824, 0.834] | high <= low |
| r_int | 0.004 [0.003, 0.006] | -0.033 [-0.038, -0.029] | high <= low |

## Training-Episode Epistemic Evidence

| regime | initial V_epi | final V_epi | final-initial paired diff [95% CI] | trend status |
| - | - | - | - | - |
| low_entropy | 0.350 | 0.008 | -0.342 [-0.342, -0.342] | SUPPORTED |
| high_entropy | 0.350 | 0.070 | -0.280 [-0.302, -0.251] | SUPPORTED |

Training evidence uses fixed probe states evaluated at checkpoints 0 to 100. This is the correct evidence type for the draft phrase `V_epi decreases within each regime as experience accumulates`.

## Exploration-Behavior Evidence

| behavior metric | low mean | high mean | high-low paired diff [95% CI] | status |
| - | - | - | - | - |
| positive_intrinsic_rate | 0.775 | 0.472 | -0.303 [-0.370, -0.247] | SUPPORTED |
| allocate_rate | 0.637 | 0.633 | -0.004 [-0.019, 0.015] | WEAK / PARTIAL |
| defer_rate | 0.293 | 0.227 | -0.066 [-0.082, -0.055] | SUPPORTED |
| reject_rate | 0.070 | 0.140 | 0.070 [0.043, 0.094] | SUPPORTED |
| mean_r_int | -0.001 | -0.044 | -0.043 [-0.074, -0.021] | SUPPORTED |
| mean_V_epi | 0.036 | 0.071 | 0.035 [0.030, 0.041] | SUPPORTED |
| mean_V_ale | 0.062 | 0.072 | 0.010 [0.007, 0.013] | SUPPORTED |

This section treats action distribution as the direct behavioral evidence. Intrinsic reward sign alone is not counted as behavior.
