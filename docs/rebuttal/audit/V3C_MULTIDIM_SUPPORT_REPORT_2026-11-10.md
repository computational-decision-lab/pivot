# V3C Multidimensional Calibration Support Audit

This is a descriptive audit of the sealed v3c calibration and ledger. It does not filter roots, refit the model, or change any selector result.

- Calibration rows: 960 (240 roots); ledger rows: 5400.
- Multivariate distance uses standardized features with ridge `1e-3`; training distance thresholds are p95 `16.2033` and p99 `19.9735`.
- Test candidate vectors above those thresholds: 80.3% (p95), 80.0% (p99).
- `round_fraction` is constant at zero in calibration but spans `[0,1]` in tests.
- Recomputed five-fold root-held-out calibration correction RMSE/MAE: 1.0325/0.7468; working 95% interval coverage: 93.0%. This is a calibration root-held-out diagnostic and is not a coverage guarantee for OOD deployment states.

## Feature support

| Feature | Calibration min | Calibration max | Test min | Test max |
|---|---:|---:|---:|---:|
| `pressure` | 0.0540625 | 0.695938 | 0.0378735 | 1.99393 |
| `response_strength` | 0.35 | 0.85 | 0.35 | 0.85 |
| `incumbent_intensity` | -0.297009 | 0.349999 | -0.0228242 | 0.369263 |
| `incumbent_bias` | -0.119832 | 0.118716 | -0.194925 | 0.139699 |
| `delta_intensity` | -0.110795 | 0.106873 | -0.114391 | 0.108117 |
| `delta_bias` | -0.0601143 | 0.0647326 | -0.0584072 | 0.0684186 |
| `proxy_delta` | -2.15224 | 2.03372 | -1.86249 | 1.81711 |
| `pressure_x_delta_intensity` | -0.071782 | 0.0647731 | -0.196226 | 0.171657 |
| `response_x_delta_intensity` | -0.0930368 | 0.081206 | -0.0953573 | 0.0873049 |
| `round_fraction` | 0 | 0 | 0 | 1 |

The calibration feature `round_fraction` is constant at zero, so the model has no calibration variation from which to identify its later-round effect. Pressure, incumbent parameters, proxy deltas, and interaction features also show test values outside their one-dimensional calibration ranges on some trajectories.

## OOD and residual summary

| Regime | Method | Round | Candidate feature-cell OOD | State any OOD (incl. round) | State any OOD (excl. round) | Mean squared distance | Residual RMSE | 95% interval coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| low | PIVOT-KG-Loop-v3c | 1 | 0.0% | 0.8% | 0.8% | 9.204 | 0.352 | 100.0% |
| low | PIVOT-KG-Loop-v3c | 2 | 10.1% | 100.0% | 2.5% | 69.477 | 0.165 | 100.0% |
| low | PIVOT-KG-Loop-v3c | 3 | 10.0% | 100.0% | 0.0% | 258.924 | 0.211 | 100.0% |
| low | PIVOT-KG-Loop-v3c | 4 | 13.0% | 100.0% | 30.8% | 572.693 | 0.227 | 100.0% |
| low | PIVOT-KG-Loop-v3c | 5 | 14.3% | 100.0% | 44.2% | 1010.889 | 0.234 | 100.0% |
| low | IVR-Online-v3c | 1 | 0.0% | 0.8% | 0.8% | 9.204 | 0.352 | 100.0% |
| low | IVR-Online-v3c | 2 | 10.1% | 100.0% | 2.5% | 69.477 | 0.165 | 100.0% |
| low | IVR-Online-v3c | 3 | 10.0% | 100.0% | 0.0% | 257.814 | 0.200 | 100.0% |
| low | IVR-Online-v3c | 4 | 10.3% | 100.0% | 5.0% | 570.819 | 0.226 | 100.0% |
| low | IVR-Online-v3c | 5 | 10.5% | 100.0% | 6.7% | 1008.483 | 0.218 | 100.0% |
| low | Top-Proxy-Online-v3c | 1 | 0.0% | 0.8% | 0.8% | 9.204 | 0.352 | 100.0% |
| low | Top-Proxy-Online-v3c | 2 | 10.1% | 100.0% | 2.5% | 69.477 | 0.165 | 100.0% |
| low | Top-Proxy-Online-v3c | 3 | 10.0% | 100.0% | 0.0% | 259.165 | 0.213 | 100.0% |
| low | Top-Proxy-Online-v3c | 4 | 17.2% | 100.0% | 71.7% | 573.954 | 0.234 | 100.0% |
| low | Top-Proxy-Online-v3c | 5 | 15.7% | 100.0% | 58.3% | 1011.650 | 0.231 | 100.0% |
| medium | PIVOT-KG-Loop-v3c | 1 | 0.0% | 1.7% | 1.7% | 6.393 | 0.343 | 100.0% |
| medium | PIVOT-KG-Loop-v3c | 2 | 10.0% | 100.0% | 1.7% | 65.638 | 0.413 | 100.0% |
| medium | PIVOT-KG-Loop-v3c | 3 | 10.0% | 100.0% | 0.0% | 253.981 | 0.471 | 100.0% |
| medium | PIVOT-KG-Loop-v3c | 4 | 10.0% | 100.0% | 0.0% | 566.977 | 0.509 | 100.0% |
| medium | PIVOT-KG-Loop-v3c | 5 | 10.0% | 100.0% | 1.7% | 1004.464 | 0.497 | 100.0% |
| medium | IVR-Online-v3c | 1 | 0.0% | 1.7% | 1.7% | 6.393 | 0.343 | 100.0% |
| medium | IVR-Online-v3c | 2 | 12.5% | 100.0% | 25.0% | 69.964 | 0.676 | 100.0% |
| medium | IVR-Online-v3c | 3 | 11.2% | 100.0% | 8.3% | 255.378 | 0.570 | 99.8% |
| medium | IVR-Online-v3c | 4 | 12.6% | 100.0% | 17.5% | 568.925 | 0.702 | 99.4% |
| medium | IVR-Online-v3c | 5 | 12.0% | 100.0% | 15.8% | 1005.576 | 0.624 | 100.0% |
| medium | Top-Proxy-Online-v3c | 1 | 0.0% | 1.7% | 1.7% | 6.393 | 0.343 | 100.0% |
| medium | Top-Proxy-Online-v3c | 2 | 10.5% | 100.0% | 5.8% | 67.457 | 0.602 | 100.0% |
| medium | Top-Proxy-Online-v3c | 3 | 12.2% | 100.0% | 20.0% | 256.123 | 0.625 | 100.0% |
| medium | Top-Proxy-Online-v3c | 4 | 12.4% | 100.0% | 21.7% | 568.899 | 0.620 | 100.0% |
| medium | Top-Proxy-Online-v3c | 5 | 12.0% | 100.0% | 17.5% | 1006.219 | 0.574 | 100.0% |
| high | PIVOT-KG-Loop-v3c | 1 | 0.1% | 5.8% | 5.8% | 9.633 | 0.570 | 100.0% |
| high | PIVOT-KG-Loop-v3c | 2 | 11.4% | 100.0% | 15.0% | 70.847 | 0.879 | 99.0% |
| high | PIVOT-KG-Loop-v3c | 3 | 12.4% | 100.0% | 22.5% | 259.654 | 0.934 | 99.4% |
| high | PIVOT-KG-Loop-v3c | 4 | 12.8% | 100.0% | 25.8% | 571.908 | 0.901 | 100.0% |
| high | PIVOT-KG-Loop-v3c | 5 | 12.4% | 100.0% | 26.7% | 1008.903 | 0.830 | 100.0% |
| high | IVR-Online-v3c | 1 | 0.1% | 5.8% | 5.8% | 9.633 | 0.570 | 100.0% |
| high | IVR-Online-v3c | 2 | 11.4% | 100.0% | 18.3% | 70.813 | 0.876 | 99.0% |
| high | IVR-Online-v3c | 3 | 13.2% | 100.0% | 25.8% | 260.982 | 1.070 | 97.5% |
| high | IVR-Online-v3c | 4 | 18.4% | 100.0% | 56.7% | 587.230 | 1.910 | 87.1% |
| high | IVR-Online-v3c | 5 | 17.8% | 100.0% | 53.3% | 1024.143 | 1.826 | 92.3% |
| high | Top-Proxy-Online-v3c | 1 | 0.1% | 5.8% | 5.8% | 9.633 | 0.570 | 100.0% |
| high | Top-Proxy-Online-v3c | 2 | 13.4% | 100.0% | 31.7% | 72.531 | 0.965 | 98.1% |
| high | Top-Proxy-Online-v3c | 3 | 16.5% | 100.0% | 52.5% | 261.392 | 0.972 | 99.2% |
| high | Top-Proxy-Online-v3c | 4 | 15.5% | 100.0% | 46.7% | 573.104 | 0.880 | 99.8% |
| high | Top-Proxy-Online-v3c | 5 | 13.9% | 100.0% | 33.3% | 1009.764 | 0.818 | 99.8% |

The state-OOD column including `round_fraction` is 100% after round 1 because that feature is constant at zero in calibration but nonzero later. The companion column excludes `round_fraction` and is the more informative state-support diagnostic. OOD rows are a property of the method-induced trajectory and therefore cannot be interpreted as an independent covariate: a selector can change its own future state distribution. The residual and interval columns use the stored independent deployment evaluation and are reported for calibration diagnosis only.

## OOD-stratified ISR

ISR is the state-level deployment-selection regret. The groups below use the full-feature support definition (any candidate feature outside its calibration min/max, including `round_fraction`); entries give root counts and mean ISR. Round-1 OOD groups are small (1-7 roots). In rounds 2-5, every state is OOD because calibration `round_fraction` is fixed at zero. Since selectors induce their own evolving state distributions, these descriptive group means do not identify a causal mechanism for ISR differences.

| Regime | Method | Round | In-support states; mean ISR | OOD states; mean ISR |
|---|---|---:|---:|---:|
| low | PIVOT-KG-Loop-v3c | 1 | 119; 0.0000 | 1; 0.0000 |
| low | PIVOT-KG-Loop-v3c | 2 | - | 120; 0.0016 |
| low | PIVOT-KG-Loop-v3c | 3 | - | 120; 0.0027 |
| low | PIVOT-KG-Loop-v3c | 4 | - | 120; 0.0026 |
| low | PIVOT-KG-Loop-v3c | 5 | - | 120; 0.0020 |
| low | IVR-Online-v3c | 1 | 119; 0.0000 | 1; 0.0000 |
| low | IVR-Online-v3c | 2 | - | 120; 0.0120 |
| low | IVR-Online-v3c | 3 | - | 120; 0.0161 |
| low | IVR-Online-v3c | 4 | - | 120; 0.0150 |
| low | IVR-Online-v3c | 5 | - | 120; 0.0127 |
| low | Top-Proxy-Online-v3c | 1 | 119; 0.0000 | 1; 0.0000 |
| low | Top-Proxy-Online-v3c | 2 | - | 120; 0.0000 |
| low | Top-Proxy-Online-v3c | 3 | - | 120; 0.0022 |
| low | Top-Proxy-Online-v3c | 4 | - | 120; 0.0317 |
| low | Top-Proxy-Online-v3c | 5 | - | 120; 0.0371 |
| medium | PIVOT-KG-Loop-v3c | 1 | 118; 0.0006 | 2; 0.0000 |
| medium | PIVOT-KG-Loop-v3c | 2 | - | 120; 0.0003 |
| medium | PIVOT-KG-Loop-v3c | 3 | - | 120; 0.0001 |
| medium | PIVOT-KG-Loop-v3c | 4 | - | 120; 0.0003 |
| medium | PIVOT-KG-Loop-v3c | 5 | - | 120; 0.0001 |
| medium | IVR-Online-v3c | 1 | 118; 0.1996 | 2; 0.2525 |
| medium | IVR-Online-v3c | 2 | - | 120; 0.0810 |
| medium | IVR-Online-v3c | 3 | - | 120; 0.0751 |
| medium | IVR-Online-v3c | 4 | - | 120; 0.0343 |
| medium | IVR-Online-v3c | 5 | - | 120; 0.0495 |
| medium | Top-Proxy-Online-v3c | 1 | 118; 0.0745 | 2; 0.1692 |
| medium | Top-Proxy-Online-v3c | 2 | - | 120; 0.0505 |
| medium | Top-Proxy-Online-v3c | 3 | - | 120; 0.0422 |
| medium | Top-Proxy-Online-v3c | 4 | - | 120; 0.0344 |
| medium | Top-Proxy-Online-v3c | 5 | - | 120; 0.0293 |
| high | PIVOT-KG-Loop-v3c | 1 | 113; 0.0334 | 7; 0.0104 |
| high | PIVOT-KG-Loop-v3c | 2 | - | 120; 0.0201 |
| high | PIVOT-KG-Loop-v3c | 3 | - | 120; 0.0176 |
| high | PIVOT-KG-Loop-v3c | 4 | - | 120; 0.0100 |
| high | PIVOT-KG-Loop-v3c | 5 | - | 120; 0.0091 |
| high | IVR-Online-v3c | 1 | 113; 0.0259 | 7; 0.1132 |
| high | IVR-Online-v3c | 2 | - | 120; 0.0724 |
| high | IVR-Online-v3c | 3 | - | 120; 0.5941 |
| high | IVR-Online-v3c | 4 | - | 120; 0.1472 |
| high | IVR-Online-v3c | 5 | - | 120; 0.1405 |
| high | Top-Proxy-Online-v3c | 1 | 113; 0.0962 | 7; 0.0939 |
| high | Top-Proxy-Online-v3c | 2 | - | 120; 0.0917 |
| high | Top-Proxy-Online-v3c | 3 | - | 120; 0.0558 |
| high | Top-Proxy-Online-v3c | 4 | - | 120; 0.0444 |
| high | Top-Proxy-Online-v3c | 5 | - | 120; 0.0455 |

## Limits

- The six Holm-adjusted v3c comparisons remain the prespecified primary tests; this audit adds no success claim.
- OOD rates do not justify removing unfavorable roots or changing response regimes.
- Persistent cross-round calibration learning was not part of v3c (`cross_round_update=none`).
