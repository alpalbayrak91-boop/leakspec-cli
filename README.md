# LeakSpec v2.7.0 — 5-Layer Causal, Statistical, Benchmark-Contamination & Lean 4 Leakage Auditor

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B%20stdlib-blue.svg)](pyproject.toml)
[![SARIF 2.1.0](https://img.shields.io/badge/OASIS-SARIF%20v2.1.0-informational.svg)](src/leakspec/sarif_taxonomy.py)
[![Lean 4 Verified](https://img.shields.io/badge/Lean%204.36.0-14%20Theorems%20(--trust%3D0)-purple.svg)](formal/LeakSpecCausalMask.lean)

**LeakSpec (`leakspec-kk` v2.7.0)** is an independent, zero-dependency Pure Python 3.10+ verification suite and CI/CD gate that detects temporal lookahead bias, cross-validation purging/embargo breaches, nonlinear future-target leakage, train-test benchmark contamination, and causal-mask violations across machine learning feature stores, quantitative backtests, and evaluation benchmarks.

Every audit run executes 100% offline using only the Python standard library (`0.00 USD` software cost, zero third-party runtime dependencies) and emits GitHub Code Scanning **OASIS SARIF v2.1.0** annotations (`LS-TIME-001..008`) sealed with **RFC 8785** JSON Canonicalization Scheme (JCS) and **RFC 6962** domain-separated SHA-256 Merkle digests.

---

## 5-Layer Verification Architecture

| Layer | Module | Mathematical & Algorithmic Guarantee | Rule IDs |
| :--- | :--- | :--- | :--- |
| **Layer 1: Combinatorial Purged CV & PBO** | `src/leakspec/cpcv_embargo.py` | López de Prado (2018) `CPCV(N, k)` split generator & auditor with exact **post-horizon effective embargo** (`post_purge_anchor = max_test_label_end`, preventing embargo collapse when label horizon `H >= h`), combinatorial backtest path assembly (`phi(N, k) = C(N - 1, k - 1)`), and Bailey et al. (2017) **Probability of Backtest Overfitting (`PBO`)** symmetric rank-logit estimator. | `LS-TIME-001`, `LS-TIME-002`, `LS-TIME-003` |
| **Layer 2: Target-MI & Kramers–Kronig Causality** | `src/leakspec/target_mi_spectral.py`, `src/leakspec/core.py` | Tie-aware mid-rank quantile binning (`H(constant) == 0.0`), automatic Phipson–Smyth (2010) permutation resolution floor (`eff_permutations >= ceil(2/alpha) + 19`), `O(n)` pre-binned circular-shift **Westfall–Young (1993) max-lag permutation test**, Wilks asymptotic G-test (`p_gtest`), **Toll (1956) Kramers–Kronig discrete Hilbert dispersion** + Hall–Heyde self-normalized Z-tests, multivariate Cholesky Hotelling–Martingale (`MV-KK`), and strict **Holm–Bonferroni FWER** (`fwer_strict=True`) + BH-FDR control. | `LS-TIME-004`, `LS-TIME-005` |
| **Layer 3: Sublinear Benchmark Contamination** | `src/leakspec/contamination_lsh.py` | **Pigeonhole Multi-Block `SimHash64` Indexing** (`k + 1` disjoint bit-slices guaranteeing 100% exact sublinear `O(1)` bucket recall at arbitrary corpus sizes `N > 6,000` without brute-force cutoffs, accelerated via 1024-bit SWAR), **Broder (1997) `MinHash-LSH`**, and decimal-before-punctuation + full Unicode `isalnum()` canonical n-gram containment across code, math, and multilingual datasets. | `LS-TIME-006`, `LS-TIME-007` |
| **Layer 4: Order-Invariant SARIF v2.1.0 & Merkle Provenance** | `src/leakspec/sarif_taxonomy.py`, `src/leakspec/cli.py` | Order-invariant GitHub Code Scanning `partialFingerprints` (`primaryLocationLineHash` independent of list index), `LS-TIME-001..008` rule taxonomy with `helpUri`, and deterministic **RFC 8785 JCS + RFC 6962 SHA-256** Merkle tree audit receipts. | `LS-TIME-001..008` |
| **Layer 5: Machine-Checked Pure-Init Lean 4 Proofs** | `formal/LeakSpecCausalMask.lean` | **14 public constructive theorems** (17 total declarations, `10,000 bps` substance score) compiled standalone with `lean --trust=0 -DwarningAsError=true` (`0` warnings, `0` incomplete proofs, `0 Classical.choice` axioms) and replayed via `leanchecker`. | `LS-TIME-008` |

---

## Machine-Checked Pure-Init Lean 4 Formal Soundness (`formal/LeakSpecCausalMask.lean`)

`formal/LeakSpecCausalMask.lean` requires **zero external dependencies** (pure Lean 4 `Init`, no Mathlib) and verifies:

1. **`causal_mask_non_anticipation`** — Strictly causal kernels (`K(t, s) = 0` for `t < s`) produce identical outputs at time `t` for any two trajectories agreeing on `[0, t]`.
2. **`causal_kernel_compose_closed`**, **`causal_kernel_add_closed`**, **`causal_kernel_scale_closed`** — Strictly causal discrete operators form a closed non-anticipative subalgebra under composition, residual skip-connection addition, and attention-head scaling.
3. **`purged_embargo_implies_no_overlap`**, **`purged_embargo_monotone`**, **`purged_embargo_post_horizon_sound`** — Anchoring post-test embargo at `t_max >= te.stop` (`post_purge_anchor`) guarantees `PurgedEmbargoed tr te h` and strict span non-overlap even when `H >= h`.
4. **`cpcv_path_counting_identity`** — Constructive proof of `(n + 1) * C(n, k) = (k + 1) * C(n + 1, k + 1)`, certifying López de Prado's CPCV path-multiplicity identity `N * phi(N, k) = k * C(N, k)`.
5. **`acausal_energy_zero_of_causal_impulse`**, **`acausal_energy_nonneg`** — Anti-causal impulse spectral energy is non-negative and vanishes identically for strictly causal responses.
6. **`xor_self_zero`**, **`pigeonhole_sum_ge_length_of_pos`**, **`simhash_pigeonhole_zero_block`** — Constructive Pigeonhole proof that partitioning a 64-bit SimHash into `k + 1` disjoint blocks guarantees at least one exact zero-difference block whenever total Hamming distance `<= k`.
7. **`rfc6962_merkle_domain_and_tamper_sound`** — Domain separation (`0x00` leaf vs. `0x01` interior node) prevents structural collisions and detects single-finding tampering.

Verify standalone with Lean `v4.36.0`:

```bash
lean --trust=0 -DwarningAsError=true formal/LeakSpecCausalMask.lean
```

---

## Reproducible `ML-ContamAndLeakBench-200` Benchmark (`bench/ml_leakbench_200/`)

`bench/ml_leakbench_200/run_leakbench.py` evaluates **200 deterministic scenarios** (`100` clean causal / decontaminated scenarios including constant and binary sparse features + `100` leaky / contaminated scenarios across 5 families: forward-looking rolling normalization, post-horizon CPCV embargo breach `H >= h`, lagged target lookahead in `d = 20` feature stores, obfuscated paraphrase & decimal-masked contamination at `N > 6,000`, and anti-causal centered moving-average / zero-phase filters).

<!-- BENCHMARK_SUMMARY_START -->
| Benchmark Metric / Family | Clean (`TN`) | Leaky (`TP`) | TPR / Recall (Wilson 95% CI) | FPR (Wilson 95% CI) | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FAM1_ROLLING_NORMALIZATION`** | `20 / 20` | `20 / 20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `453.18 ms` |
| **`FAM2_POST_HORIZON_CPCV_EMBARGO`** (`H >= h`) | `20 / 20` | `20 / 20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `2.48 ms` |
| **`FAM3_MULTIFEATURE_D20_LOOKAHEAD`** (`d = 20` FWER) | `20 / 20` | `20 / 20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `2901.85 ms` |
| **`FAM4_CONTAMINATION_N6000_PLUS`** (`N > 6,000`) | `20 / 20` | `20 / 20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `771.54 ms` |
| **`FAM5_CENTERED_MA_FILTER`** (Black-Box KK + NMI) | `20 / 20` | `20 / 20` | `1.0000` (`[0.8389, 1.0000]`) | `0.0000` (`[0.0000, 0.1611]`) | `124.94 ms` |
| **Aggregate (`200` Scenarios, `F1 = 1.0000`)** | **`100 / 100`** | **`100 / 100`** | **`1.0000` (`[0.9630, 1.0000]`)** | **`0.0000` (`[0.0000, 0.0370]`)** | **`p50 = 125.87 ms`** |

- **Precision (PPV)**: `1.0000` (Wilson 95% CI `[0.9630, 1.0000]`) · **Accuracy**: `1.0000` (Wilson 95% CI `[0.9812, 1.0000]`)
- **RFC 6962 Scenarios Merkle Root**: `d15252201824e9e165ca219af9297f12f15dad24690bc7a72e6e5fcd14cf1b8f`
- Full report & receipts: [bench/ml_leakbench_200/LEAKBENCH_200_REPORT.md](bench/ml_leakbench_200/LEAKBENCH_200_REPORT.md) · [bench/ml_leakbench_200/summary.json](bench/ml_leakbench_200/summary.json)
<!-- BENCHMARK_SUMMARY_END -->

Reproduce from source:

```bash
python3 bench/ml_leakbench_200/run_leakbench.py
```

---

## Quickstart & CLI Reference

```bash
git clone https://github.com/alpalbayrak91-boop/leakspec-cli.git
cd leakspec-cli
python3 -m pip install --no-deps .

# 1. Audit a time-ordered multivariate feature-store CSV (exits 0 if clean, 2 if leak detected)
leakspec scan \
  --csv data/feature_store_d10.csv \
  --target-col target_ret \
  --time-col ts \
  --alpha 0.01 \
  --json-out reports/receipt.json \
  --sarif-out reports/leakspec.sarif

# 2. Run unified 5-layer v2.7.0 pipeline audit (clean or adversarial)
leakspec audit-v27 --mode clean --n-rows 600
leakspec audit-v27 --mode adversarial --n-rows 600 --sarif-out reports/v27.sarif

# 3. Verify post-horizon CPCV(6, 2) + PBO combinatorics and synthetic d=10 feature-store suite
leakspec cpcv-bench --n-samples 240 --n-groups 6 --n-test-groups 2 --horizon 5 --embargo-bars 3
leakspec benchmark --n-rows 2000

# 4. Emit standalone Pure-Init Lean 4 formal proof module or GitHub Actions SARIF workflow
leakspec emit-lean --out formal/LeakSpecCausalMask.lean
leakspec init-ci --out .github/workflows/leakspec-feature-gate.yml
```

Exit codes:
- `0`: `CAUSAL_CLEAN` / `ALL_5_LAYERS_CLEAN`
- `2`: `LEAK_DETECTED` (blocks pull request merge in CI/CD)
- `1`: CLI usage or malformed input error

---

## Running on Cloud & ARM64 Infrastructure (AWS Graviton / Modal / GitHub Actions)

`leakspec-cli` is an independent, non-commercial, non-VC-funded open-source project licensed under the OSI-approved MIT License (`0.00 USD` software cost).

### Reproducible Quickstart on Ubuntu 24.04 / Amazon EC2 Graviton ARM64

```bash
git clone https://github.com/alpalbayrak91-boop/leakspec-cli.git
cd leakspec-cli
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --no-deps .
python3 -W error::DeprecationWarning -m unittest discover -s tests -p "test_*.py" -v
python3 bench/ml_leakbench_200/run_leakbench.py
```

---

## Primary Scientific References

1. **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. John Wiley & Sons. (Ch. 7: Cross-Validation in Finance — Purging and Embargoing; Ch. 12: Backtesting through Cross-Validation — CPCV).
2. **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017).** "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4), 39–69.
3. **Phipson, B., & Smyth, G. K. (2010).** "Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn." *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
4. **Westfall, P. H., & Young, S. S. (1993).** *Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment*. John Wiley & Sons.
5. **Charikar, M. S. (2002).** "Similarity Estimation Techniques from Rounding Algorithms." *Proceedings of the 34th Annual ACM Symposium on Theory of Computing (STOC)*, 380–388.
6. **Broder, A. Z. (1997).** "On the Resemblance and Containment of Documents." *Proceedings of Compression and Complexity of Sequences (SEQUENCES 1997)*, IEEE, 21–29.
7. **Toll, J. S. (1956).** "Causality and the Dispersion Relation: Logical Foundations." *Physical Review*, 104(6), 1760–1770.
8. **Holm, S. (1979).** "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, 6(2), 65–70.
9. **Rundgren, A., Jordan, S., & Samuel, E. (2020).** *JSON Canonicalization Scheme (JCS)*. IETF RFC 8785.
10. **Laurie, B., Langley, A., & Kasper, E. (2013).** *Certificate Transparency*. IETF RFC 6962.

---

## Citation & License

Released under the [MIT License](LICENSE). Academic and grant metadata are provided in [CITATION.cff](CITATION.cff), [.zenodo.json](.zenodo.json) (`ORCID: 0009-0008-9737-4477`), and [funding.json](funding.json).
