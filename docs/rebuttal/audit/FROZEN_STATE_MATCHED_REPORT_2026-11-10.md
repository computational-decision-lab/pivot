# Frozen-State Matched Selector Diagnostic

This post-review auxiliary experiment holds the decision state, incumbent, candidate panel, proxy values, calibration prior, query budget, evaluation seeds, and terminal rule fixed across selectors. It is separate from the v3c closed-loop confirmation.

## Mean ISR

| Response | PIVOT-KG | IVR | Top-Proxy | IVR - PIVOT (95% CI) | Top-Proxy - PIVOT (95% CI) |
|---|---:|---:|---:|---:|---:|
| low | 0.00000 | 0.00000 | 0.00000 | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] |
| medium | 0.00054 | 0.19957 | 0.06443 | 0.19903 [0.17628, 0.22139] | 0.06389 [0.04962, 0.07882] |
| high | 0.03297 | 0.03051 | 0.09510 | -0.00246 [-0.01516, 0.01050] | 0.06213 [0.04968, 0.07507] |

## Paired contrasts

Positive ISR contrast favors PIVOT; positive deployment-return contrast also favors PIVOT.

| Contrast | ISR difference (95% CI) | Holm p | Deployment-return difference (95% CI) |
|---|---:|---:|---:|
| low:IVR-Online-v3c_minus_PIVOT_ISR | 0.00000 [0.00000, 0.00000] | 1 | 0.00000 [0.00000, 0.00000] |
| low:Top-Proxy-Online-v3c_minus_PIVOT_ISR | 0.00000 [0.00000, 0.00000] | 1 | 0.00000 [0.00000, 0.00000] |
| medium:IVR-Online-v3c_minus_PIVOT_ISR | 0.19903 [0.17628, 0.22139] | 5.9999e-05 | 0.19803 [0.17542, 0.21953] |
| medium:Top-Proxy-Online-v3c_minus_PIVOT_ISR | 0.06389 [0.04962, 0.07882] | 5.9999e-05 | 0.06383 [0.04951, 0.07837] |
| high:IVR-Online-v3c_minus_PIVOT_ISR | -0.00246 [-0.01516, 0.01050] | 1 | -0.00249 [-0.01472, 0.00984] |
| high:Top-Proxy-Online-v3c_minus_PIVOT_ISR | 0.06213 [0.04968, 0.07507] | 5.9999e-05 | 0.06207 [0.04989, 0.07477] |

This is an auxiliary diagnostic specified after the earlier v1-v4a and v3c results. Its tests do not extend the v3c confirmation claim or correct for that preceding variant search. All roots are retained.

Independent replay checks: `PASS`; 2160 query decisions replayed.
