# ML-ContamAndLeakBench-200 — Reproducible Verification Report (`LeakSpec v2.7.0`)

Deterministic 200-scenario evaluation suite (`100` clean causal / decontaminated scenarios + `100` leaky / contaminated scenarios across 5 benchmark families) executed with zero external Python dependencies.

## Aggregate Metrics

| Metric | Value | Wilson 95% CI |
| :--- | :--- | :--- |
| **Scenarios Evaluated** | `200` (`100` clean + `100` leaky) | — |
| **True Positives / False Negatives** | `TP = 100`, `FN = 0` | — |
| **True Negatives / False Positives** | `TN = 100`, `FP = 0` | — |
| **True Positive Rate (Recall)** | `1.0000` | `[0.9630, 1.0000]` |
| **False Positive Rate (FPR)** | `0.0000` | `[0.0000, 0.0370]` |
| **Specificity (TNR)** | `1.0000` | `[0.9630, 1.0000]` |
| **Precision (PPV)** | `1.0000` | `[0.9630, 1.0000]` |
| **F1 Score** | `1.0000` | — |
| **Overall Accuracy** | `1.0000` | `[0.9812, 1.0000]` |

## Per-Family Breakdown (`20` Clean + `20` Leaky per Family)

| Family | Clean (`TN/20`) | Leaky (`TP/20`) | TPR (95% CI) | FPR (95% CI) | Mean Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `FAM1_ROLLING_NORMALIZATION` | `20/20` | `20/20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `453.18` |
| `FAM2_POST_HORIZON_CPCV_EMBARGO` | `20/20` | `20/20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `2.48` |
| `FAM3_MULTIFEATURE_D20_LOOKAHEAD` | `20/20` | `20/20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `2901.85` |
| `FAM4_CONTAMINATION_N6000_PLUS` | `20/20` | `20/20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `771.54` |
| `FAM5_CENTERED_MA_FILTER` | `20/20` | `20/20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `124.94` |

## Runtime Latency & Cryptographic Audit Provenance

- **Total Wall-Clock Runtime (200 scenarios)**: `241.0885 s`
- **Per-Scenario Latency**: mean `850.796 ms`, median (`p50`) `125.870 ms`, `p95` `3966.901 ms`
- **`N > 6,000` Pigeonhole SimHash64 + MinHash-LSH Corpus Index Build Times (ms)**: `[20168.85, 25159.77, 11657.06, 12662.98]`
- **RFC 6962 Scenarios Merkle Root (SHA-256)**: `d15252201824e9e165ca219af9297f12f15dad24690bc7a72e6e5fcd14cf1b8f`
- **RFC 8785 JCS Summary Digest (SHA-256)**: `3baebc0cf8153e6c82e55c4b6d3ee6ede8f8127000a44ee6e559e6406dcacc97`

## Reproduction Command

```bash
python3 bench/ml_leakbench_200/run_leakbench.py
```
