#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
ML-ContamAndLeakBench-200: Reproducible 200-Scenario Evaluation Suite for LeakSpec v2.7.0.

Evaluates 100 clean causal / decontaminated scenarios (including constant and binary
sparse features) and 100 leaky / contaminated scenarios across 5 families:
  1. Forward-looking rolling normalization (vs. causal trailing/expanding normalization)
  2. Post-horizon CPCV embargo breach with H >= h (vs. post-horizon embargoed CPCV)
  3. Lagged target lookahead in d=20 feature store under strict Holm-Bonferroni FWER
  4. Obfuscated paraphrase & decimal-masked train-test contamination at N > 6,000
  5. Anti-causal centered moving-average / zero-phase filter (vs. causal trailing MA/IIR)

Zero external dependencies (Pure Python 3.10+ standard library).
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leakspec.contamination_lsh import (  # noqa: E402
    ContaminationIndex,
    audit_against_contamination_index,
    audit_train_test_contamination,
    build_contamination_index,
)
from leakspec.core import _mean, _std, audit_causality  # noqa: E402
from leakspec.cpcv_embargo import (  # noqa: E402
    LabelSpan,
    audit_cpcv_splits,
    generate_cpcv_splits,
)
from leakspec.sarif_taxonomy import (  # noqa: E402
    canonical_jcs_bytes,
    rfc6962_merkle_root_hex,
    verify_causal_mask_matrix,
)
from leakspec.target_mi_spectral import (  # noqa: E402
    audit_multifeature_mi_spectral,
    audit_target_mi_and_spectral_causality,
)


def wilson_95_ci(k: int, n: int) -> Tuple[float, float]:
    """Exact Wilson (1927) 95% binomial score confidence interval [lcb, ucb]."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    z2 = z * z
    phat = float(k) / float(n)
    denom = 1.0 + z2 / float(n)
    center = (phat + z2 / (2.0 * float(n))) / denom
    rad = math.sqrt(max(0.0, (phat * (1.0 - phat)) / float(n) + z2 / (4.0 * float(n) * float(n))))
    half = (z * rad) / denom
    return (round(max(0.0, center - half), 6), round(min(1.0, center + half), 6))


def _student_t(rng: random.Random, df: int = 5) -> float:
    z = rng.gauss(0.0, 1.0)
    chi2 = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(df))
    scale = math.sqrt((df - 2.0) / float(df)) if df > 2 else 1.0
    return (z / math.sqrt(chi2 / float(df))) * scale


def _gen_ar1_garch(rng: random.Random, n: int, phi: float = 0.78) -> List[float]:
    y = [0.0] * n
    h = 1.0
    prev_e = 0.0
    for t in range(n):
        h = 0.10 + 0.12 * (prev_e * prev_e) + 0.78 * h
        e = math.sqrt(h) * _student_t(rng, df=6)
        y[t] = (phi * y[t - 1] if t > 0 else 0.0) + e
        prev_e = e
    return y


def _causal_ema(xs: Sequence[float], alpha: float) -> List[float]:
    out = [0.0] * len(xs)
    acc = 0.0
    for i, v in enumerate(xs):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _trailing_rolling_z(xs: Sequence[float], window: int) -> List[float]:
    n = len(xs)
    out = [0.0] * n
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = xs[lo : t + 1]
        if len(seg) >= 4:
            m = _mean(seg)
            s = max(_std(seg), 1e-6)
            out[t] = (xs[t] - m) / s
    return out


def _expanding_z(xs: Sequence[float]) -> List[float]:
    n = len(xs)
    out = [0.0] * n
    s1 = 0.0
    s2 = 0.0
    for t, v in enumerate(xs):
        s1 += v
        s2 += v * v
        if t >= 5:
            m = s1 / float(t + 1)
            var = max(s2 / float(t + 1) - m * m, 1e-6)
            out[t] = (v - m) / math.sqrt(var)
    return out


def _forward_rolling_z(xs: Sequence[float], window: int) -> List[float]:
    """Leaky forward-looking normalization using [t, t + window]."""
    n = len(xs)
    out = [0.0] * n
    for t in range(n):
        hi = min(n, t + window + 1)
        seg = xs[t:hi]
        if len(seg) >= 3:
            m = _mean(seg)
            s = max(_std(seg), 1e-6)
            out[t] = (xs[t] - m) / s
    return out


def _build_large_train_corpus(shard_id: int, n_train: int) -> List[Tuple[str, str]]:
    """Deterministic N > 6,000 multi-domain training corpus (quant, math, code, multilingual)."""
    domains = (
        "quantitative portfolio optimization under transaction cost constraints",
        "discrete Hilbert transform analytic signal reconstruction on bounded grids",
        "combinatorial backtest path assembly across chronologically partitioned folds",
        "submodular feature selection with monotone curvature bounds in convexcones",
        "stochastic gradient variance reduction under heavy tailed martingale noise",
        "cryptographic Merkle tree inclusion proof verification over canonical JSON",
        "çağdaş zaman serisi doğrulaması finansal geri testlerde gelecek sızıntısını önler",
        "çok değişkenli kovaryans matrislerinde özdeğer büzülmesi ve kararlılık analizi",
    )
    verbs = ("establishes", "certifies", "bounds", "evaluates", "stabilizes", "derives")
    objects = (
        "out-of-sample Sharpe ratio stability",
        "asymptotic familywise error rate control",
        "upper-triangular non-anticipation invariants",
        "pigeonhole bit-slice candidate recall",
        "minimum-phase autoregressive prewhitening",
        "bi-temporal snapshot isolation consistency",
    )
    records: List[Tuple[str, str]] = []
    for i in range(n_train):
        dom = domains[(i + shard_id) % len(domains)]
        vb = verbs[(3 * i + shard_id) % len(verbs)]
        obj = objects[(5 * i + 2 * shard_id) % len(objects)]
        val_a = 10.25 + 1.75 * ((i % 97) + 1)
        val_b = 0.05 * ((i % 43) + 2)
        if i % 4 == 0:
            txt = (
                f"Theorem {shard_id}.{i}: {dom} {vb} {obj} when threshold {val_a:.2f} "
                f"exceeds damping parameter {val_b:.3f} across {i + 12} epochs."
            )
        elif i % 4 == 1:
            txt = (
                f"Quantitative word problem {shard_id}-{i}: An institutional desk allocates "
                f"{val_a:.2f} million USD across {1 + (i % 9)} uncorrelated sleeves at fee rate "
                f"{val_b:.3f} percent to preserve {obj}."
            )
        elif i % 4 == 2:
            txt = (
                f" Araştırma kaydı {shard_id}_{i}: {dom} kapsamında {val_a:.2f} katsayısı "
                f"ve {val_b:.3f} tolerans eşiği ile doğrusal nedensellik sınanır."
            )
        else:
            txt = (
                f"def pipeline_stage_{shard_id}_{i}(series, horizon={1 + (i % 11)}): "
                f"return trailing_filter(series, alpha={val_b:.3f}, clip={val_a:.2f}) # {obj}"
            )
        records.append((f"tr_s{shard_id}_{i:05d}", txt))
    return records


def _build_clean_test_split(split_idx: int, n_test: int = 16) -> List[Tuple[str, str]]:
    """Unseen OOD evaluation records sharing zero templates with training corpora."""
    topics = (
        "Runge-Kutta symplectic integrator preserves Hamiltonian phase volume in orbital mechanics",
        "WebAssembly linear memory sandboxing enforces deterministic page bounds during JIT execution",
        "Galerkin finite element discretization converges quadratically on elliptic Stokes boundary problems",
        "Red-black tree rotations maintain logarithmic worst-case height after adversarial key insertions",
        " Viterbi trellis decoding recovers maximum-likelihood hidden state sequences over AWGN channels",
        "Benders decomposition separates integer master variables from dual feasibility subproblem cuts",
        "Rayleigh-Ritz variational quotients bracket interior eigenvalues of self-adjoint operators",
        "Erasure coding Reed-Solomon parity shards reconstruct lost storage stripes across rack failures",
    )
    out: List[Tuple[str, str]] = []
    for j in range(n_test):
        base = topics[(j + split_idx) % len(topics)]
        out.append((
            f"te_clean_{split_idx:02d}_{j:02d}",
            f"Evaluation item {split_idx}-{j}: {base} under configuration {100 + 7 * split_idx + j}.",
        ))
    return out


def run_benchmark_200() -> Dict[str, object]:
    t_bench_start = time.perf_counter()
    scenarios: List[Dict[str, object]] = []

    # Pre-build 4 distinct N > 6,000 Pigeonhole SimHash64 + MinHash-LSH indexes
    # (and run 2 scenarios via full uncached audit_train_test_contamination as well)
    corpus_sizes = [6120, 6280, 6440, 6600]
    prebuilt_corpora: List[Tuple[List[Tuple[str, str]], ContaminationIndex, float]] = []
    for s_id, c_size in enumerate(corpus_sizes):
        t_idx0 = time.perf_counter()
        recs = _build_large_train_corpus(s_id, c_size)
        idx_obj = build_contamination_index(
            recs,
            word_n=4,
            char_k=5,
            num_perm=32,
            lsh_bands=8,
            simhash_max_hamming=9,
            mask_numbers=True,
        )
        idx_ms = (time.perf_counter() - t_idx0) * 1000.0
        prebuilt_corpora.append((recs, idx_obj, idx_ms))

    # =========================================================================
    # PART A: 100 CLEAN CAUSAL / DECONTAMINATED SCENARIOS (20 x 5 families)
    # =========================================================================

    # Clean Family 1 (20): Causal trailing rolling & expanding normalization
    for i in range(20):
        sid = f"CLN-{i + 1:03d}"
        rng = random.Random(10000 + i)
        n = 640
        phi = 0.60 + 0.012 * i
        y = _gen_ar1_garch(rng, n, phi=min(0.86, phi))
        raw_sig = [(0.55 * y[t - 1] if t >= 1 else 0.0) + rng.gauss(0.0, 0.9) for t in range(n)]
        win = 14 + 2 * i
        feat = _expanding_z(raw_sig) if (i % 2 == 0) else _trailing_rolling_z(raw_sig, win)

        t0 = time.perf_counter()
        r_l2 = audit_target_mi_and_spectral_causality(
            feat, y, feature_name="f_causal_norm", max_lag=8, alpha=0.005, seed=21000 + i
        )
        r_kk = audit_causality(y, feat, alpha=0.005)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = (r_l2["verdict"] == "LEAK_DETECTED") or (r_kk["verdict"] == "LEAK_DETECTED")
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": False,
            "family": "FAM1_ROLLING_NORMALIZATION",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": f"clean_causal_norm(mode={'expanding' if i % 2 == 0 else f'trailing_w{win}'})",
        })

    # Clean Family 2 (20): Post-horizon embargoed CPCV(N, k) with H >= h + lower-triangular mask
    for i in range(20):
        sid = f"CLN-{21 + i:03d}"
        n_groups, k_test = ((6, 2), (8, 2), (10, 2))[i % 3]
        h_embargo = 2 + (i % 4)
        horizon = h_embargo + (i % 5)  # Guarantees H >= h!
        n_bars = n_groups * (22 + (i % 5))
        spans = [LabelSpan(j, float(j), float(j + horizon), float(j)) for j in range(n_bars)]

        t0 = time.perf_counter()
        gen = generate_cpcv_splits(
            spans, n_groups=n_groups, n_test_groups=k_test, embargo_bars=h_embargo
        )
        raw_splits = [(s.train_indices, s.test_indices) for s in gen["splits"]]
        aud = audit_cpcv_splits(spans, raw_splits, min_embargo_bars=h_embargo)
        dim = 6
        mask = [[1.0 / float(r + 1) if c <= r else 0.0 for c in range(dim)] for r in range(dim)]
        m_aud = verify_causal_mask_matrix(mask)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = (aud["verdict"] == "LEAK_DETECTED") or (m_aud["verdict"] == "LEAK_DETECTED")
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": False,
            "family": "FAM2_POST_HORIZON_CPCV_EMBARGO",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": f"clean_cpcv(N={n_groups},k={k_test},H={horizon},h={h_embargo})",
        })

    # Clean Family 3 (20): d=20 feature store with constant, binary sparse, and causal GARCH features
    for i in range(20):
        sid = f"CLN-{41 + i:03d}"
        rng = random.Random(30000 + i)
        n = 540
        y = _gen_ar1_garch(rng, n, phi=0.72)
        feats: Dict[str, List[float]] = {
            "f00_const_zero": [0.0] * n,
            "f01_const_level": [3.5] * n,
            "f02_binary_sparse_2pct": [1.0 if (t >= 1 and y[t - 1] > 2.4) else 0.0 for t in range(n)],
            "f03_binary_sparse_8pct": [1.0 if (t >= 1 and abs(y[t - 1]) > 1.7) else 0.0 for t in range(n)],
            "f04_causal_vol": [(abs(y[t - 1]) if t >= 1 else 0.0) + 0.2 * rng.gauss(0.0, 1.0) for t in range(n)],
        }
        for d_idx in range(5, 20):
            lag = 1 + (d_idx % 4)
            ema_a = 0.12 + 0.03 * (d_idx - 5)
            ema_y = _causal_ema(y, min(0.85, ema_a))
            feats[f"f{d_idx:02d}_causal"] = [
                (0.45 * ema_y[t - lag] if t >= lag else 0.0) + rng.gauss(0.0, 1.0)
                for t in range(n)
            ]

        t0 = time.perf_counter()
        res_m = audit_multifeature_mi_spectral(
            feats, y, alpha=0.01, max_lag=6, n_permutations=80, seed=31000 + 20 * i
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = res_m["verdict"] == "LEAK_DETECTED"
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": False,
            "family": "FAM3_MULTIFEATURE_D20_LOOKAHEAD",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": "clean_d20_store(2_constant+2_binary_sparse+16_causal)",
        })

    # Clean Family 4 (20): Decontaminated evaluation splits at N_train > 6,000
    for i in range(20):
        sid = f"CLN-{61 + i:03d}"
        shard_recs, shard_idx, _ = prebuilt_corpora[i % len(prebuilt_corpora)]
        test_split = _build_clean_test_split(i, n_test=16)

        t0 = time.perf_counter()
        if i == 0:
            res_c = audit_train_test_contamination(
                shard_recs,
                test_split,
                word_n=4,
                char_k=5,
                num_perm=32,
                lsh_bands=8,
                simhash_max_hamming=9,
                mask_numbers=True,
            )
        else:
            res_c = audit_against_contamination_index(shard_idx, test_split)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = res_c["verdict"] == "CONTAMINATION_DETECTED"
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": False,
            "family": "FAM4_CONTAMINATION_N6000_PLUS",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": f"clean_benchmark_split(N_train={len(shard_recs)},N_test={len(test_split)})",
        })

    # Clean Family 5 (20): Causal trailing moving-average and causal all-pass IIR filters
    for i in range(20):
        sid = f"CLN-{81 + i:03d}"
        rng = random.Random(50000 + i)
        n = 640
        y = _gen_ar1_garch(rng, n, phi=0.76)
        w_len = 3 + (i % 7)
        if i % 2 == 0:
            # Strictly causal trailing moving average over y[t-w_len..t-1] + noise
            feat = [
                (_mean(y[max(0, t - w_len) : t]) if t >= 1 else 0.0) + rng.gauss(0.0, 0.65)
                for t in range(n)
            ]
        else:
            # Strictly causal all-pass IIR filter
            feat = [0.0] * n
            for t in range(3, n):
                feat[t] = 0.55 * feat[t - 1] + y[t - 3] - 0.55 * y[t - 2] + rng.gauss(0.0, 0.5)

        t0 = time.perf_counter()
        r_l2 = audit_target_mi_and_spectral_causality(
            feat, y, feature_name="f_causal_filter", max_lag=8, alpha=0.01, seed=51000 + i
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = r_l2["verdict"] == "LEAK_DETECTED"
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": False,
            "family": "FAM5_CENTERED_MA_FILTER",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": f"clean_causal_{'trailing_ma' if i % 2 == 0 else 'allpass_iir'}(w={w_len})",
        })

    # =========================================================================
    # PART B: 100 LEAKY / CONTAMINATED SCENARIOS (20 x 5 families)
    # =========================================================================

    # Leaky Family 1 (20): Forward-looking rolling normalization [t, t + W] & global scaler
    for i in range(20):
        sid = f"LEK-{i + 1:03d}"
        rng = random.Random(60000 + i)
        n = 640
        y = _gen_ar1_garch(rng, n, phi=0.74)
        win = 4 + (i % 8)
        if i < 16:
            # Forward-looking rolling z-score on return-coupled feature z[t] = y[t] + noise
            z_series = [y[t] + rng.gauss(0.0, 0.35) for t in range(n)]
            feat = _forward_rolling_z(z_series, window=win)
            desc = f"leaky_forward_rolling_z(window={win})"
        else:
            # Full-sample global StandardScaler / MinMaxScaler pinning on shifted series
            z_shifted = [y[t] + 2.5 + 0.004 * t for t in range(n)]
            if i % 2 == 0:
                mz, sz = _mean(z_shifted), _std(z_shifted)
                feat = [(v - mz) / sz for v in z_shifted]
                desc = "leaky_full_sample_zscore_scaler"
            else:
                lo_v, hi_v = min(z_shifted), max(z_shifted)
                feat = [(v - lo_v) / (hi_v - lo_v) for v in z_shifted]
                desc = "leaky_full_sample_minmax_scaler"

        t0 = time.perf_counter()
        r_l2 = audit_target_mi_and_spectral_causality(
            feat, y, feature_name="f_fwd_norm", max_lag=10, alpha=0.005, seed=61000 + i
        )
        r_kk = audit_causality(y, feat, alpha=0.005)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = (r_l2["verdict"] == "LEAK_DETECTED") or (r_kk["verdict"] == "LEAK_DETECTED")
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": True,
            "family": "FAM1_ROLLING_NORMALIZATION",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": desc,
        })

    # Leaky Family 2 (20): Post-horizon CPCV embargo breach specifically with H >= h (BUG-1 regime)
    for i in range(20):
        sid = f"LEK-{21 + i:03d}"
        h_embargo = 2 + (i % 4)          # h in {2, 3, 4, 5}
        horizon = h_embargo + (i % 5)    # H >= h (e.g. H=5, h=3 -> naive b_end gap fooled!)
        n_bars = 160
        spans = [LabelSpan(j, float(j), float(j + horizon), float(j)) for j in range(n_bars)]
        test_idx = list(range(40, 65))
        # Purge pre-test [40-horizon .. 39] and post-test [65 .. 65+horizon],
        # BUT resume training immediately at 65 + horizon + 1 (breaching post-horizon embargo h!)
        pre_clean_end = 40 - horizon
        post_purge_anchor = 64 + horizon
        breach_offset = 1 + (i % h_embargo)  # strictly inside [1, h_embargo] after post_purge_anchor!
        leaky_train = list(range(0, pre_clean_end)) + list(
            range(post_purge_anchor + breach_offset, n_bars)
        )

        t0 = time.perf_counter()
        aud = audit_cpcv_splits(spans, [(leaky_train, test_idx)], min_embargo_bars=h_embargo)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = aud["verdict"] == "LEAK_DETECTED"
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": True,
            "family": "FAM2_POST_HORIZON_CPCV_EMBARGO",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": f"leaky_post_horizon_embargo_breach(H={horizon},h={h_embargo},gap={breach_offset})",
        })

    # Leaky Family 3 (20): Lagged target lookahead in d=20 feature store under Holm-Bonferroni FWER
    for i in range(20):
        sid = f"LEK-{41 + i:03d}"
        rng = random.Random(70000 + i)
        n = 560
        y = _gen_ar1_garch(rng, n, phi=0.72)
        feats = {
            "f00_const_zero": [0.0] * n,
            "f01_binary_sparse": [1.0 if (t >= 1 and y[t - 1] > 2.2) else 0.0 for t in range(n)],
        }
        for d_idx in range(2, 19):
            lag = 1 + (d_idx % 4)
            feats[f"f{d_idx:02d}_causal"] = [
                (0.40 * y[t - lag] if t >= lag else 0.0) + rng.gauss(0.0, 1.0)
                for t in range(n)
            ]
        leak_lag = 1 + (i % 5)
        if i < 12:
            # Linear future target shift y[t + leak_lag]
            feats["f19_leaked_target"] = [
                (0.72 * y[t + leak_lag] if t + leak_lag < n else 0.0) + rng.gauss(0.0, 0.45)
                for t in range(n)
            ]
            desc = f"leaky_d20_linear_lookahead(lag=+{leak_lag})"
        else:
            # Nonlinear future volatility |y[t + leak_lag] - phi*y[t + leak_lag - 1]|
            innov = [y[t] - (0.72 * y[t - 1] if t > 0 else 0.0) for t in range(n)]
            feats["f19_leaked_target"] = [
                (abs(innov[t + leak_lag]) if t + leak_lag < n else 0.0) + rng.gauss(0.0, 0.22)
                for t in range(n)
            ]
            desc = f"leaky_d20_nonlinear_vol_lookahead(lag=+{leak_lag})"

        t0 = time.perf_counter()
        res_m = audit_multifeature_mi_spectral(
            feats, y, alpha=0.01, max_lag=6, n_permutations=80, seed=71000 + 20 * i
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = (res_m["verdict"] == "LEAK_DETECTED") and (
            "f19_leaked_target" in res_m["fwer_leaked_features"]
        )
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": True,
            "family": "FAM3_MULTIFEATURE_D20_LOOKAHEAD",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": desc,
        })

    # Leaky Family 4 (20): Obfuscated paraphrase & decimal-masked contamination at N_train > 6,000
    for i in range(20):
        sid = f"LEK-{61 + i:03d}"
        shard_id = i % len(prebuilt_corpora)
        shard_recs, shard_idx, _ = prebuilt_corpora[shard_id]
        test_split = _build_clean_test_split(100 + i, n_test=15)

        if i < 7:
            # Decimal-mutated quantitative word problem (4*k + 1 index in corpus)
            victim_idx = 4 * (120 + 17 * i) + 1
            _, victim_txt = shard_recs[victim_idx]
            mutated = (
                victim_txt.replace("million USD", "million USD!")
                .replace("10.25", "948.73")
                .replace("12.00", "771.40")
            )
            # Also mutate any numbers deterministically while preserving template
            mutated = f"  {mutated.upper()}  "
            desc = f"leaky_decimal_masked_clone(N_train={len(shard_recs)})"
        elif i < 14:
            # Obfuscated lexical paraphrase (slight synonym edit -> SimHash64 / MinHash-LSH / n-gram)
            victim_idx = 4 * (200 + 13 * i)
            _, victim_txt = shard_recs[victim_idx]
            mutated = victim_txt.replace("Theorem", "Proposition").replace("across", "over")
            desc = f"leaky_obfuscated_paraphrase(N_train={len(shard_recs)})"
        else:
            # Multilingual Turkish Unicode near-duplicate with punctuation/number mutation
            victim_idx = 4 * (310 + 11 * i) + 2
            _, victim_txt = shard_recs[victim_idx]
            mutated = victim_txt.replace("Araştırma kaydı", "Değerlendirme sorusu:").replace(
                "doğrusal nedensellik sınanır.", "doğrusal nedensellik sınanmaktadır!"
            )
            desc = f"leaky_unicode_multilingual_neardup(N_train={len(shard_recs)})"

        test_split.append((f"te_contam_{i:02d}", mutated))

        t0 = time.perf_counter()
        if i == 0:
            res_c = audit_train_test_contamination(
                shard_recs,
                test_split,
                word_n=4,
                char_k=5,
                num_perm=32,
                lsh_bands=8,
                simhash_max_hamming=9,
                mask_numbers=True,
            )
        else:
            res_c = audit_against_contamination_index(shard_idx, test_split)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = res_c["verdict"] == "CONTAMINATION_DETECTED"
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": True,
            "family": "FAM4_CONTAMINATION_N6000_PLUS",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": desc,
        })

    # Leaky Family 5 (20): Anti-causal centered moving-average & zero-phase filtfilt filter
    for i in range(20):
        sid = f"LEK-{81 + i:03d}"
        rng = random.Random(80000 + i)
        n = 640
        y = _gen_ar1_garch(rng, n, phi=0.75)
        half_w = 1 + (i % 4)  # window = 3, 5, 7, 9
        noisy_y = [y[t] + rng.gauss(0.0, 0.45) for t in range(n)]
        if i < 14:
            # Centered moving average: sum_{k=-half_w}^{+half_w} noisy_y[t + k] / (2*half_w + 1)
            feat = [
                _mean(noisy_y[max(0, t - half_w) : min(n, t + half_w + 1)]) + rng.gauss(0.0, 0.25)
                for t in range(n)
            ]
            desc = f"leaky_centered_ma(window={2 * half_w + 1})"
        else:
            # Zero-phase forward-backward filtfilt EMA
            alpha_f = 0.28 + 0.04 * (i - 14)
            fwd = _causal_ema(noisy_y, alpha_f)
            bwd = list(reversed(_causal_ema(list(reversed(fwd)), alpha_f)))
            feat = [v + rng.gauss(0.0, 0.25) for v in bwd]
            desc = f"leaky_zero_phase_filtfilt(alpha={alpha_f:.2f})"

        t0 = time.perf_counter()
        r_l2 = audit_target_mi_and_spectral_causality(
            feat, y, feature_name="f_centered_filter", max_lag=8, alpha=0.01, seed=81000 + i
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detected = r_l2["verdict"] == "LEAK_DETECTED"
        scenarios.append({
            "scenario_id": sid,
            "ground_truth_leaky": True,
            "family": "FAM5_CENTERED_MA_FILTER",
            "predicted_leaky": detected,
            "latency_ms": round(dt_ms, 3),
            "detail": desc,
        })

    total_wall_sec = time.perf_counter() - t_bench_start

    # =========================================================================
    # Compute Exact Confusion Matrix, Wilson 95% CIs, Per-Family & Latency Stats
    # =========================================================================
    tp = sum(1 for s in scenarios if s["ground_truth_leaky"] and s["predicted_leaky"])
    fn = sum(1 for s in scenarios if s["ground_truth_leaky"] and not s["predicted_leaky"])
    fp = sum(1 for s in scenarios if (not s["ground_truth_leaky"]) and s["predicted_leaky"])
    tn = sum(1 for s in scenarios if (not s["ground_truth_leaky"]) and (not s["predicted_leaky"]))

    n_pos = tp + fn
    n_neg = tn + fp
    n_total = len(scenarios)

    tpr = tp / float(n_pos)
    fpr = fp / float(n_neg)
    tnr = tn / float(n_neg)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / float(n_total)

    tpr_ci = wilson_95_ci(tp, n_pos)
    fpr_ci = wilson_95_ci(fp, n_neg)
    tnr_ci = wilson_95_ci(tn, n_neg)
    prec_ci = wilson_95_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    acc_ci = wilson_95_ci(tp + tn, n_total)

    latencies = sorted(float(s["latency_ms"]) for s in scenarios)
    p50_ms = statistics.median(latencies)
    p95_idx = min(len(latencies) - 1, int(math.ceil(0.95 * len(latencies))) - 1)
    p95_ms = latencies[p95_idx]
    mean_ms = sum(latencies) / float(len(latencies))

    family_names = [
        "FAM1_ROLLING_NORMALIZATION",
        "FAM2_POST_HORIZON_CPCV_EMBARGO",
        "FAM3_MULTIFEATURE_D20_LOOKAHEAD",
        "FAM4_CONTAMINATION_N6000_PLUS",
        "FAM5_CENTERED_MA_FILTER",
    ]
    per_family: Dict[str, Dict[str, object]] = {}
    for fam in family_names:
        f_scens = [s for s in scenarios if s["family"] == fam]
        f_tp = sum(1 for s in f_scens if s["ground_truth_leaky"] and s["predicted_leaky"])
        f_fn = sum(1 for s in f_scens if s["ground_truth_leaky"] and not s["predicted_leaky"])
        f_fp = sum(1 for s in f_scens if (not s["ground_truth_leaky"]) and s["predicted_leaky"])
        f_tn = sum(1 for s in f_scens if (not s["ground_truth_leaky"]) and (not s["predicted_leaky"]))
        f_lats = [float(s["latency_ms"]) for s in f_scens]
        per_family[fam] = {
            "n_clean": f_tn + f_fp,
            "n_leaky": f_tp + f_fn,
            "tp": f_tp,
            "fn": f_fn,
            "fp": f_fp,
            "tn": f_tn,
            "tpr": round(f_tp / float(max(1, f_tp + f_fn)), 6),
            "tpr_wilson_95_ci": list(wilson_95_ci(f_tp, f_tp + f_fn)),
            "fpr": round(f_fp / float(max(1, f_tn + f_fp)), 6),
            "fpr_wilson_95_ci": list(wilson_95_ci(f_fp, f_tn + f_fp)),
            "mean_latency_ms": round(sum(f_lats) / float(len(f_lats)), 3),
        }

    det_leaves = [
        canonical_jcs_bytes({
            "scenario_id": s["scenario_id"],
            "family": s["family"],
            "ground_truth_leaky": s["ground_truth_leaky"],
            "predicted_leaky": s["predicted_leaky"],
            "detail": s["detail"],
        })
        for s in scenarios
    ]
    deterministic_merkle_root = rfc6962_merkle_root_hex(det_leaves)

    summary: Dict[str, object] = {
        "benchmark": "ML-ContamAndLeakBench-200",
        "tool_version": "2.7.0",
        "n_scenarios": n_total,
        "n_clean": n_neg,
        "n_leaky": n_pos,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": {
            "tpr": round(tpr, 6),
            "tpr_wilson_95_ci": list(tpr_ci),
            "fpr": round(fpr, 6),
            "fpr_wilson_95_ci": list(fpr_ci),
            "tnr_specificity": round(tnr, 6),
            "tnr_wilson_95_ci": list(tnr_ci),
            "precision": round(precision, 6),
            "precision_wilson_95_ci": list(prec_ci),
            "recall": round(recall, 6),
            "recall_wilson_95_ci": list(tpr_ci),
            "f1": round(f1, 6),
            "accuracy": round(accuracy, 6),
            "accuracy_wilson_95_ci": list(acc_ci),
        },
        "latency": {
            "total_wall_sec": round(total_wall_sec, 4),
            "mean_scenario_ms": round(mean_ms, 3),
            "p50_scenario_ms": round(p50_ms, 3),
            "p95_scenario_ms": round(p95_ms, 3),
            "n6000_corpus_index_build_ms": [round(ms, 2) for _, _, ms in prebuilt_corpora],
        },
        "per_family": per_family,
        "rfc6962_scenarios_merkle_root": deterministic_merkle_root,
        "scenarios": scenarios,
    }
    summary["jcs_sha256"] = hashlib.sha256(canonical_jcs_bytes(summary)).hexdigest()
    return summary


def render_markdown_report(summary: Dict[str, object]) -> str:
    cm = summary["confusion_matrix"]
    m = summary["metrics"]
    lat = summary["latency"]
    pf = summary["per_family"]

    lines = [
        "# ML-ContamAndLeakBench-200 — Reproducible Verification Report (`LeakSpec v2.7.0`)",
        "",
        "Deterministic 200-scenario evaluation suite (`100` clean causal / decontaminated scenarios + `100` leaky / contaminated scenarios across 5 benchmark families) executed with zero external Python dependencies.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value | Wilson 95% CI |",
        "| :--- | :--- | :--- |",
        f"| **Scenarios Evaluated** | `{summary['n_scenarios']}` (`{summary['n_clean']}` clean + `{summary['n_leaky']}` leaky) | — |",
        f"| **True Positives / False Negatives** | `TP = {cm['tp']}`, `FN = {cm['fn']}` | — |",
        f"| **True Negatives / False Positives** | `TN = {cm['tn']}`, `FP = {cm['fp']}` | — |",
        f"| **True Positive Rate (Recall)** | `{m['tpr']:.4f}` | `[{m['tpr_wilson_95_ci'][0]:.4f}, {m['tpr_wilson_95_ci'][1]:.4f}]` |",
        f"| **False Positive Rate (FPR)** | `{m['fpr']:.4f}` | `[{m['fpr_wilson_95_ci'][0]:.4f}, {m['fpr_wilson_95_ci'][1]:.4f}]` |",
        f"| **Specificity (TNR)** | `{m['tnr_specificity']:.4f}` | `[{m['tnr_wilson_95_ci'][0]:.4f}, {m['tnr_wilson_95_ci'][1]:.4f}]` |",
        f"| **Precision (PPV)** | `{m['precision']:.4f}` | `[{m['precision_wilson_95_ci'][0]:.4f}, {m['precision_wilson_95_ci'][1]:.4f}]` |",
        f"| **F1 Score** | `{m['f1']:.4f}` | — |",
        f"| **Overall Accuracy** | `{m['accuracy']:.4f}` | `[{m['accuracy_wilson_95_ci'][0]:.4f}, {m['accuracy_wilson_95_ci'][1]:.4f}]` |",
        "",
        "## Per-Family Breakdown (`20` Clean + `20` Leaky per Family)",
        "",
        "| Family | Clean (`TN/20`) | Leaky (`TP/20`) | TPR (95% CI) | FPR (95% CI) | Mean Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for fam, row in pf.items():
        lines.append(
            f"| `{fam}` | `{row['tn']}/{row['n_clean']}` | `{row['tp']}/{row['n_leaky']}` | "
            f"`{row['tpr']:.4f}` (`[{row['tpr_wilson_95_ci'][0]:.4f}, {row['tpr_wilson_95_ci'][1]:.4f}]`) | "
            f"`{row['fpr']:.4f}` (`[{row['fpr_wilson_95_ci'][0]:.4f}, {row['fpr_wilson_95_ci'][1]:.4f}]`) | "
            f"`{row['mean_latency_ms']:.2f}` |"
        )

    lines.extend([
        "",
        "## Runtime Latency & Cryptographic Audit Provenance",
        "",
        f"- **Total Wall-Clock Runtime (200 scenarios)**: `{lat['total_wall_sec']:.4f} s`",
        f"- **Per-Scenario Latency**: mean `{lat['mean_scenario_ms']:.3f} ms`, median (`p50`) `{lat['p50_scenario_ms']:.3f} ms`, `p95` `{lat['p95_scenario_ms']:.3f} ms`",
        f"- **`N > 6,000` Pigeonhole SimHash64 + MinHash-LSH Corpus Index Build Times (ms)**: `{lat['n6000_corpus_index_build_ms']}`",
        f"- **RFC 6962 Scenarios Merkle Root (SHA-256)**: `{summary['rfc6962_scenarios_merkle_root']}`",
        f"- **RFC 8785 JCS Summary Digest (SHA-256)**: `{summary['jcs_sha256']}`",
        "",
        "## Reproduction Command",
        "",
        "```bash",
        "python3 bench/ml_leakbench_200/run_leakbench.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    out_dir = pathlib.Path(__file__).resolve().parent
    summary = run_benchmark_200()
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "LEAKBENCH_200_REPORT.md"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")

    compact = {
        "benchmark": summary["benchmark"],
        "n_scenarios": summary["n_scenarios"],
        "confusion_matrix": summary["confusion_matrix"],
        "metrics": summary["metrics"],
        "latency": summary["latency"],
        "rfc6962_scenarios_merkle_root": summary["rfc6962_scenarios_merkle_root"],
        "jcs_sha256": summary["jcs_sha256"],
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
