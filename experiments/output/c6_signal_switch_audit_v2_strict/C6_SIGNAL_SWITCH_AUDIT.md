# C6 Surprise-Agnostic Switch Contradiction Audit

This audit reruns simplified PSAIM under the exact C4 signal-separation seed/protocol and compares it with the C4 and C5 signal artifacts already on disk. All CIs are 95% nonparametric bootstrap intervals over seed-level regime means. The simulator, agent, and hyperparameters were not modified.

## Resolution

fragility finding; apply decision rule (b) for the paper. C4 is exactly reproducible with current code under the recovered C4 seed/protocol, but C5 used the same PSAIM hyperparameters, episode length, and regime-block protocol with independently seeded checkpoints/workloads and did not reproduce the positive low-entropy sign. The discrepancy is therefore not lambda/sigma/protocol drift; it is checkpoint/workload-seed sensitivity.

## Protocol And Configuration Comparison

| item                 | C4 original                                                 | C6 rerun                                                 | C5 simplified PSAIM                                                     |
| -------------------- | ----------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| lambda_aleatoric     | 3.0 via make_experiment_2_config()                          | 3.000                                                    | 3.0 via make_experiment_2_config()                                      |
| sigma0_sq            | 0.05 via make_experiment_2_config()                         | 0.050                                                    | 0.05 via make_experiment_2_config()                                     |
| alpha_gate           | 0.16 via make_experiment_2_config()                         | 0.160                                                    | 0.16 via make_experiment_2_config()                                     |
| probe_horizon        | 3 via make_experiment_2_config()                            | 3                                                        | 3 via make_experiment_2_config()                                        |
| train/eval episodes  | 100 / 8                                                     | 100 / 8                                                  | 100 / 8                                                                 |
| horizon/block        | 96 / 24                                                     | 96 / 24                                                  | 96 / 24                                                                 |
| seed/checkpoint path | agent/template/train seed = seed+300; eval seed = 4000+seed | same as C4 original                                      | agent seed base 83000+seed; train seed 84000+seed; eval seed 85000+seed |
| code path            | src.experiment_runner.experiment_2 full multi-agent loop    | src.experiment_runner.experiment_2 full multi-agent loop | run_c5_mechanism_controls.py focused psaim_lite/rawvar signal audit     |
| saved checkpoint     | not available                                               | fresh deterministic rerun                                | fresh deterministic run                                                 |

## Per-Regime Summary

| source                                | metric        | low_entropy            | high_entropy            | paired_high_minus_low   |
| ------------------------------------- | ------------- | ---------------------- | ----------------------- | ----------------------- |
| C4 original artifact                  | V_ale         | 0.057 [0.055, 0.059]   | 0.072 [0.071, 0.074]    | 0.015 [0.014, 0.017]    |
| C4 original artifact                  | V_epi         | 0.025 [0.022, 0.030]   | 0.064 [0.058, 0.071]    | 0.039 [0.035, 0.043]    |
| C4 original artifact                  | allocate_rate | 0.633 [0.607, 0.662]   | 0.717 [0.698, 0.735]    | 0.084 [0.063, 0.105]    |
| C4 original artifact                  | defer_rate    | 0.344 [0.320, 0.366]   | 0.210 [0.193, 0.225]    | -0.135 [-0.164, -0.101] |
| C4 original artifact                  | gate_h3       | 0.930 [0.921, 0.938]   | 0.829 [0.808, 0.850]    | -0.101 [-0.115, -0.086] |
| C4 original artifact                  | r_int         | 0.004 [0.001, 0.007]   | -0.033 [-0.048, -0.020] | -0.037 [-0.050, -0.026] |
| C4 original artifact                  | reject_rate   | 0.023 [0.015, 0.030]   | 0.073 [0.049, 0.093]    | 0.050 [0.033, 0.066]    |
| C6 strict original experiment_2 rerun | V_ale         | 0.057 [0.055, 0.059]   | 0.072 [0.071, 0.074]    | 0.015 [0.014, 0.017]    |
| C6 strict original experiment_2 rerun | V_epi         | 0.025 [0.022, 0.030]   | 0.064 [0.058, 0.071]    | 0.039 [0.035, 0.043]    |
| C6 strict original experiment_2 rerun | allocate_rate | 0.633 [0.607, 0.662]   | 0.717 [0.698, 0.735]    | 0.084 [0.063, 0.105]    |
| C6 strict original experiment_2 rerun | defer_rate    | 0.344 [0.320, 0.366]   | 0.210 [0.193, 0.225]    | -0.135 [-0.164, -0.101] |
| C6 strict original experiment_2 rerun | gate_h3       | 0.930 [0.921, 0.938]   | 0.829 [0.808, 0.850]    | -0.101 [-0.115, -0.086] |
| C6 strict original experiment_2 rerun | r_int         | 0.004 [0.001, 0.007]   | -0.033 [-0.048, -0.020] | -0.037 [-0.050, -0.026] |
| C6 strict original experiment_2 rerun | reject_rate   | 0.023 [0.015, 0.030]   | 0.073 [0.049, 0.093]    | 0.050 [0.033, 0.066]    |
| C5 signal audit                       | V_ale         | 0.059 [0.058, 0.060]   | 0.072 [0.071, 0.073]    | 0.013 [0.012, 0.014]    |
| C5 signal audit                       | V_epi         | 0.035 [0.030, 0.041]   | 0.075 [0.069, 0.082]    | 0.041 [0.037, 0.044]    |
| C5 signal audit                       | allocate_rate | 0.642 [0.623, 0.656]   | 0.708 [0.653, 0.747]    | 0.066 [0.025, 0.099]    |
| C5 signal audit                       | defer_rate    | 0.335 [0.312, 0.363]   | 0.231 [0.178, 0.301]    | -0.104 [-0.141, -0.054] |
| C5 signal audit                       | gate_h3       | 0.906 [0.888, 0.919]   | 0.799 [0.777, 0.819]    | -0.107 [-0.123, -0.092] |
| C5 signal audit                       | r_int         | -0.009 [-0.026, 0.003] | -0.056 [-0.090, -0.028] | -0.046 [-0.066, -0.030] |
| C5 signal audit                       | reject_rate   | 0.023 [0.011, 0.035]   | 0.061 [0.034, 0.086]    | 0.038 [0.022, 0.054]    |

## Seed-Level Values

| source                                | seed | regime       | r_int     | V_epi    | V_ale    | gate_h3  | allocate_rate | defer_rate | reject_rate |
| ------------------------------------- | ---- | ------------ | --------- | -------- | -------- | -------- | ------------- | ---------- | ----------- |
| C4 original artifact                  | 0    | high_entropy | -0.059543 | 0.083895 | 0.074837 | 0.776817 | 0.755102      | 0.237245   | 0.007653    |
| C4 original artifact                  | 0    | low_entropy  | 0.002165  | 0.040615 | 0.062258 | 0.897664 | 0.728723      | 0.271277   | 0.000000    |
| C4 original artifact                  | 1    | high_entropy | -0.084546 | 0.070218 | 0.073000 | 0.806551 | 0.727041      | 0.186224   | 0.086735    |
| C4 original artifact                  | 1    | low_entropy  | -0.007848 | 0.023882 | 0.053981 | 0.933144 | 0.617021      | 0.367021   | 0.015957    |
| C4 original artifact                  | 2    | high_entropy | -0.024577 | 0.060132 | 0.073708 | 0.845621 | 0.724490      | 0.158163   | 0.117347    |
| C4 original artifact                  | 2    | low_entropy  | 0.005601  | 0.026154 | 0.059766 | 0.928141 | 0.643617      | 0.319149   | 0.037234    |
| C4 original artifact                  | 3    | high_entropy | -0.044660 | 0.079016 | 0.072409 | 0.777195 | 0.765306      | 0.234694   | 0.000000    |
| C4 original artifact                  | 3    | low_entropy  | 0.007282  | 0.033412 | 0.060238 | 0.916119 | 0.688830      | 0.311170   | 0.000000    |
| C4 original artifact                  | 4    | high_entropy | -0.021196 | 0.066057 | 0.073811 | 0.821559 | 0.716837      | 0.204082   | 0.079082    |
| C4 original artifact                  | 4    | low_entropy  | 0.008256  | 0.024122 | 0.056529 | 0.932503 | 0.601064      | 0.367021   | 0.031915    |
| C4 original artifact                  | 5    | high_entropy | -0.023231 | 0.054848 | 0.072117 | 0.855364 | 0.711735      | 0.209184   | 0.079082    |
| C4 original artifact                  | 5    | low_entropy  | 0.006051  | 0.025586 | 0.059403 | 0.932379 | 0.648936      | 0.316489   | 0.034574    |
| C4 original artifact                  | 6    | high_entropy | -0.037078 | 0.059083 | 0.071711 | 0.841400 | 0.665816      | 0.244898   | 0.089286    |
| C4 original artifact                  | 6    | low_entropy  | 0.001964  | 0.025006 | 0.057523 | 0.933403 | 0.617021      | 0.351064   | 0.031915    |
| C4 original artifact                  | 7    | high_entropy | -0.016526 | 0.064596 | 0.069697 | 0.827659 | 0.729592      | 0.181122   | 0.089286    |
| C4 original artifact                  | 7    | low_entropy  | 0.007005  | 0.019706 | 0.052923 | 0.940898 | 0.574468      | 0.393617   | 0.031915    |
| C4 original artifact                  | 8    | high_entropy | -0.018122 | 0.055238 | 0.069172 | 0.856022 | 0.676020      | 0.221939   | 0.102041    |
| C4 original artifact                  | 8    | low_entropy  | 0.003197  | 0.016726 | 0.053618 | 0.944625 | 0.587766      | 0.388298   | 0.023936    |
| C4 original artifact                  | 9    | high_entropy | -0.001414 | 0.049361 | 0.074355 | 0.882638 | 0.698980      | 0.219388   | 0.081633    |
| C4 original artifact                  | 9    | low_entropy  | 0.010232  | 0.018501 | 0.055937 | 0.942219 | 0.619681      | 0.359043   | 0.021277    |
| C5 signal audit                       | 0    | high_entropy | -0.048743 | 0.063427 | 0.071502 | 0.835340 | 0.755102      | 0.229592   | 0.015306    |
| C5 signal audit                       | 0    | low_entropy  | -0.007162 | 0.027117 | 0.059937 | 0.923996 | 0.664894      | 0.335106   | 0.000000    |
| C5 signal audit                       | 1    | high_entropy | -0.038645 | 0.071687 | 0.069928 | 0.800345 | 0.693878      | 0.216837   | 0.089286    |
| C5 signal audit                       | 1    | low_entropy  | 0.001826  | 0.033499 | 0.056508 | 0.908831 | 0.648936      | 0.313830   | 0.037234    |
| C5 signal audit                       | 2    | high_entropy | -0.138849 | 0.089371 | 0.072492 | 0.738523 | 0.785714      | 0.122449   | 0.091837    |
| C5 signal audit                       | 2    | low_entropy  | -0.029850 | 0.040778 | 0.060424 | 0.890816 | 0.664894      | 0.281915   | 0.053191    |
| C5 signal audit                       | 3    | high_entropy | -0.024849 | 0.080456 | 0.070293 | 0.779037 | 0.747449      | 0.165816   | 0.086735    |
| C5 signal audit                       | 3    | low_entropy  | 0.000816  | 0.031933 | 0.058966 | 0.919775 | 0.648936      | 0.319149   | 0.031915    |
| C5 signal audit                       | 4    | high_entropy | 0.009295  | 0.072465 | 0.073879 | 0.817256 | 0.755102      | 0.137755   | 0.107143    |
| C5 signal audit                       | 4    | low_entropy  | 0.014298  | 0.028509 | 0.058265 | 0.923714 | 0.640957      | 0.332447   | 0.026596    |
| C5 signal audit                       | 5    | high_entropy | -0.025388 | 0.075476 | 0.071977 | 0.800845 | 0.744898      | 0.255102   | 0.000000    |
| C5 signal audit                       | 5    | low_entropy  | 0.002030  | 0.028807 | 0.055682 | 0.921876 | 0.614362      | 0.385638   | 0.000000    |
| C5 signal audit                       | 6    | high_entropy | -0.024608 | 0.060751 | 0.073827 | 0.849183 | 0.676020      | 0.216837   | 0.107143    |
| C5 signal audit                       | 6    | low_entropy  | 0.006204  | 0.029120 | 0.060805 | 0.923529 | 0.651596      | 0.311170   | 0.037234    |
| C5 signal audit                       | 7    | high_entropy | -0.052909 | 0.077365 | 0.072223 | 0.790753 | 0.714286      | 0.214286   | 0.071429    |
| C5 signal audit                       | 7    | low_entropy  | 0.002744  | 0.037446 | 0.060545 | 0.903165 | 0.662234      | 0.300532   | 0.037234    |
| C5 signal audit                       | 8    | high_entropy | -0.157046 | 0.093751 | 0.072798 | 0.749516 | 0.494898      | 0.502551   | 0.002551    |
| C5 signal audit                       | 8    | low_entropy  | -0.072800 | 0.057817 | 0.059987 | 0.836189 | 0.571809      | 0.428191   | 0.000000    |
| C5 signal audit                       | 9    | high_entropy | -0.055903 | 0.068420 | 0.074087 | 0.828157 | 0.714286      | 0.247449   | 0.038265    |
| C5 signal audit                       | 9    | low_entropy  | -0.010971 | 0.032805 | 0.059616 | 0.908218 | 0.654255      | 0.340426   | 0.005319    |
| C6 strict original experiment_2 rerun | 0    | high_entropy | -0.059543 | 0.083895 | 0.074837 | 0.776817 | 0.755102      | 0.237245   | 0.007653    |
| C6 strict original experiment_2 rerun | 0    | low_entropy  | 0.002165  | 0.040615 | 0.062258 | 0.897664 | 0.728723      | 0.271277   | 0.000000    |
| C6 strict original experiment_2 rerun | 1    | high_entropy | -0.084546 | 0.070218 | 0.073000 | 0.806551 | 0.727041      | 0.186224   | 0.086735    |
| C6 strict original experiment_2 rerun | 1    | low_entropy  | -0.007848 | 0.023882 | 0.053981 | 0.933144 | 0.617021      | 0.367021   | 0.015957    |
| C6 strict original experiment_2 rerun | 2    | high_entropy | -0.024577 | 0.060132 | 0.073708 | 0.845621 | 0.724490      | 0.158163   | 0.117347    |
| C6 strict original experiment_2 rerun | 2    | low_entropy  | 0.005601  | 0.026154 | 0.059766 | 0.928141 | 0.643617      | 0.319149   | 0.037234    |
| C6 strict original experiment_2 rerun | 3    | high_entropy | -0.044660 | 0.079016 | 0.072409 | 0.777195 | 0.765306      | 0.234694   | 0.000000    |
| C6 strict original experiment_2 rerun | 3    | low_entropy  | 0.007282  | 0.033412 | 0.060238 | 0.916119 | 0.688830      | 0.311170   | 0.000000    |
| C6 strict original experiment_2 rerun | 4    | high_entropy | -0.021196 | 0.066057 | 0.073811 | 0.821559 | 0.716837      | 0.204082   | 0.079082    |
| C6 strict original experiment_2 rerun | 4    | low_entropy  | 0.008256  | 0.024122 | 0.056529 | 0.932503 | 0.601064      | 0.367021   | 0.031915    |
| C6 strict original experiment_2 rerun | 5    | high_entropy | -0.023231 | 0.054848 | 0.072117 | 0.855364 | 0.711735      | 0.209184   | 0.079082    |
| C6 strict original experiment_2 rerun | 5    | low_entropy  | 0.006051  | 0.025586 | 0.059403 | 0.932379 | 0.648936      | 0.316489   | 0.034574    |
| C6 strict original experiment_2 rerun | 6    | high_entropy | -0.037078 | 0.059083 | 0.071711 | 0.841400 | 0.665816      | 0.244898   | 0.089286    |
| C6 strict original experiment_2 rerun | 6    | low_entropy  | 0.001964  | 0.025006 | 0.057523 | 0.933403 | 0.617021      | 0.351064   | 0.031915    |
| C6 strict original experiment_2 rerun | 7    | high_entropy | -0.016526 | 0.064596 | 0.069697 | 0.827659 | 0.729592      | 0.181122   | 0.089286    |
| C6 strict original experiment_2 rerun | 7    | low_entropy  | 0.007005  | 0.019706 | 0.052923 | 0.940898 | 0.574468      | 0.393617   | 0.031915    |
| C6 strict original experiment_2 rerun | 8    | high_entropy | -0.018122 | 0.055238 | 0.069172 | 0.856022 | 0.676020      | 0.221939   | 0.102041    |
| C6 strict original experiment_2 rerun | 8    | low_entropy  | 0.003197  | 0.016726 | 0.053618 | 0.944625 | 0.587766      | 0.388298   | 0.023936    |
| C6 strict original experiment_2 rerun | 9    | high_entropy | -0.001414 | 0.049361 | 0.074355 | 0.882638 | 0.698980      | 0.219388   | 0.081633    |
| C6 strict original experiment_2 rerun | 9    | low_entropy  | 0.010232  | 0.018501 | 0.055937 | 0.942219 | 0.619681      | 0.359043   | 0.021277    |

## Interpretation

- The C4 reference artifact and the C6 rerun use the recovered C4 seed/checkpoint path and produce the same aggregate sign pattern: low-entropy `r_int` is positive on average and high-entropy `r_int` is negative on average.
- The C5 simplified-PSAIM audit uses the same lambda, sigma0, alpha, probe horizon, episode length, and block length, but a different trained checkpoint path and evaluation workload seeds. Its aggregate low-entropy `r_int` is slightly negative.
- PSAIM-RawVar's much larger negative intrinsic values in C5 are expected from its different total-variance penalty formula; they do not imply that simplified PSAIM used a different lambda or sigma0.
- The seed-level table shows that the low-entropy sign is not uniformly positive across seeds even in the C4 artifact. Therefore, the safest paper wording is directional separation rather than a robust clean sign switch.

## Paper Decision

Use the softened wording unless the paper explicitly ties the sign-switch statement to the C4 seed/protocol. Recommended replacement: `Simplified PSAIM shows directional separation in V_epi, V_ale, and gate values across hidden regimes; the sign of r_int is seed-sensitive in this proxy, so we do not claim a robust surprise-agnostic sign switch.`