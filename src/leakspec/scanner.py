# SPDX-License-Identifier: MIT
"""
Multivariate Feature-Store CSV Scanner, Holm-FWER / BH-FDR Gate, and SARIF 2.1.0 Exporter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Mapping, Sequence, Tuple

from .core import _mean, _std, audit_causality, mv_kk_joint_audit


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.lower() in {"nan", "na", "null", "none"}:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def load_feature_store_csv(
    csv_path: str,
    target_col: str,
    feature_cols: Sequence[str] | None = None,
    time_col: str | None = None,
    sort_by_time: bool = False,
) -> Dict[str, object]:
    """
    Loads a time-ordered multivariate feature-store CSV with strict hygiene checks:
      - Verifies timestamp monotonicity (or sorts when sort_by_time=True).
      - Drops rows with non-finite/missing numeric values and logs exact drop count.
      - Auto-discovers numeric feature columns when feature_cols is omitted.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        headers = list(reader.fieldnames)
        if target_col not in headers:
            raise ValueError(f"target_col '{target_col}' not found in headers: {headers}")
        if time_col and time_col not in headers:
            raise ValueError(f"time_col '{time_col}' not found in headers: {headers}")
        raw_rows = list(reader)

    if feature_cols:
        selected_cols = list(feature_cols)
        missing = [c for c in selected_cols if c not in headers]
        if missing:
            raise ValueError(f"feature_cols missing from CSV: {missing}")
    else:
        excluded = {target_col}
        if time_col:
            excluded.add(time_col)
        selected_cols = [h for h in headers if h not in excluded]

    monotonic_ok = True
    non_monotonic_events = 0
    if time_col:
        if sort_by_time:
            raw_rows.sort(key=lambda r: (r.get(time_col) or ""))
        prev_ts: str | None = None
        for r in raw_rows:
            cur = r.get(time_col) or ""
            if prev_ts is not None and cur < prev_ts:
                monotonic_ok = False
                non_monotonic_events += 1
            prev_ts = cur

    target_vals: List[float] = []
    feat_vals: Dict[str, List[float]] = {c: [] for c in selected_cols}
    dropped_rows = 0
    for r in raw_rows:
        tv = _parse_float(r.get(target_col))
        if tv is None:
            dropped_rows += 1
            continue
        parsed_row: List[float] = []
        ok = True
        for c in selected_cols:
            fv = _parse_float(r.get(c))
            if fv is None:
                ok = False
                break
            parsed_row.append(fv)
        if not ok:
            dropped_rows += 1
            continue
        target_vals.append(tv)
        for c, fv in zip(selected_cols, parsed_row):
            feat_vals[c].append(fv)

    return {
        "csv_path": csv_path,
        "target_col": target_col,
        "time_col": time_col,
        "feature_cols": selected_cols,
        "n_raw_rows": len(raw_rows),
        "n_valid_rows": len(target_vals),
        "n_dropped_rows": dropped_rows,
        "timestamp_monotonic": monotonic_ok,
        "non_monotonic_events": non_monotonic_events,
        "target": target_vals,
        "features": feat_vals,
    }


def holm_bonferroni_adjust(p_items: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """Step-down Holm-Bonferroni FWER adjusted p-values."""
    m = len(p_items)
    if m == 0:
        return {}
    ordered = sorted(p_items, key=lambda kv: kv[1])
    adj: Dict[str, float] = {}
    running_max = 0.0
    for rank, (key, pval) in enumerate(ordered):
        cand = min(1.0, pval * (m - rank))
        running_max = max(running_max, cand)
        adj[key] = running_max
    return adj


def benjamini_hochberg_fdr(p_items: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """Step-up Benjamini-Hochberg FDR q-values for feature triage."""
    m = len(p_items)
    if m == 0:
        return {}
    ordered = sorted(p_items, key=lambda kv: kv[1])
    q_rev = [0.0] * m
    running_min = 1.0
    for idx in range(m - 1, -1, -1):
        _, pval = ordered[idx]
        cand = min(1.0, pval * m / float(idx + 1))
        running_min = min(running_min, cand)
        q_rev[idx] = running_min
    return {ordered[i][0]: q_rev[i] for i in range(m)}


def audit_feature_store(
    target: Sequence[float],
    features: Mapping[str, Sequence[float]],
    alpha: float = 0.01,
    ar_p: int = 6,
    causal_lags: int = 10,
    acausal_lags: int = 16,
    sumrule_horizon: int = 24,
    enable_mv_kk: bool = True,
    source_label: str = "in_memory",
) -> Dict[str, object]:
    """
    Audits a multivariate feature store (d features against target) using:
      - Univariate 3-Layer KK-CAUSALSPEC per feature (IP-KK + NL2-KK + SUMRULE-KK + Pinning)
      - Layer 4 MV-KK Joint Cholesky-Whitened Hotelling-Martingale across active features
      - Store-level Holm-Bonferroni FWER gate (m = d_active + 1) + per-feature BH-FDR triage
    """
    active_features: Dict[str, Sequence[float]] = {}
    skipped_features: Dict[str, str] = {}
    for fname, series in features.items():
        if _std(series) < 1e-12:
            skipped_features[fname] = "CONSTANT_ZERO_VARIANCE"
        else:
            active_features[fname] = series

    raw_results: Dict[str, Dict[str, object]] = {}
    p_items_features: List[Tuple[str, float]] = []
    for fname, series in active_features.items():
        res = audit_causality(
            target,
            series,
            alpha=alpha,
            ar_p=ar_p,
            causal_lags=causal_lags,
            acausal_lags=acausal_lags,
            sumrule_horizon=sumrule_horizon,
        )
        raw_results[fname] = res
        p_items_features.append((fname, float(res["p_fwer_min"])))

    mv_res: Dict[str, object] = (
        mv_kk_joint_audit(
            target,
            active_features,
            ar_p=ar_p,
            causal_lags=causal_lags,
            acausal_lags=acausal_lags,
        )
        if enable_mv_kk and len(active_features) >= 2
        else {
            "active": False,
            "reason": "DISABLED_OR_SINGLE_FEATURE",
            "d_active": len(active_features),
            "p_mv_peak": 1.0,
            "p_mv_total": 1.0,
            "p_mv_min": 1.0,
            "best_lag": 0,
            "best_t2": 0.0,
            "q_total": 0.0,
            "top_suspects": [],
            "attribution_z_at_best_lag": {},
        }
    )

    store_p_items = list(p_items_features)
    if mv_res.get("active"):
        store_p_items.append(("__MV_KK_JOINT__", float(mv_res["p_mv_min"])))

    holm_store = holm_bonferroni_adjust(store_p_items)
    bh_features = benjamini_hochberg_fdr(p_items_features)

    feature_rows: List[Dict[str, object]] = []
    leaked_features: List[str] = []
    for fname in features.keys():
        if fname in skipped_features:
            feature_rows.append({
                "feature": fname,
                "status": "SKIPPED",
                "reason": skipped_features[fname],
                "verdict": "SKIPPED_CONSTANT",
                "p_univariate_fwer": 1.0,
                "p_store_holm": 1.0,
                "q_bh_fdr": 1.0,
                "dominant_trigger": "NONE",
                "estimated_leak_lag": 0,
                "dominant_z": 0.0,
            })
            continue
        res = raw_results[fname]
        p_uni = float(res["p_fwer_min"])
        p_holm = float(holm_store[fname])
        q_bh = float(bh_features[fname])
        pinned = bool(res["global_pinning"]["pinned"])
        is_leaked = (p_holm < alpha) or pinned
        if is_leaked:
            leaked_features.append(fname)
        feature_rows.append({
            "feature": fname,
            "status": "AUDITED",
            "verdict": "LEAK_DETECTED" if is_leaked else "CAUSAL_CLEAN",
            "p_univariate_fwer": round(p_uni, 8),
            "p_store_holm": round(p_holm, 8),
            "q_bh_fdr": round(q_bh, 8),
            "dominant_trigger": str(res["dominant_trigger"]),
            "estimated_leak_lag": int(res["estimated_leak_lag"]),
            "dominant_z": round(float(res["dominant_z"]), 3),
            "global_pin_mode": str(res["global_pinning"]["mode"]),
            "linear_kk_defect_ratio": round(float(res["linear_kk"]["kk_spectral"]["kk_defect_ratio"]), 6),
            "sumrule_z": round(float(res["linear_kk"]["z_sumrule"]), 3),
        })

    mv_p_holm = float(holm_store.get("__MV_KK_JOINT__", 1.0))
    mv_triggered = bool(mv_res.get("active") and (mv_p_holm < alpha))
    joint_only_detection = mv_triggered and (len(leaked_features) == 0)
    store_leaked = bool(leaked_features) or mv_triggered

    report: Dict[str, object] = {
        "schema_version": "kk-causalspec-v2.0",
        "source": source_label,
        "n_samples": len(target),
        "d_total_features": len(features),
        "d_active_features": len(active_features),
        "d_skipped_features": len(skipped_features),
        "alpha_store_fwer": alpha,
        "store_verdict": "LEAK_DETECTED" if store_leaked else "CAUSAL_CLEAN",
        "exit_code_recommended": 2 if store_leaked else 0,
        "leaked_feature_count": len(leaked_features),
        "leaked_features": leaked_features,
        "joint_only_stealth_leak": joint_only_detection,
        "mv_kk_joint": {
            **mv_res,
            "p_store_holm": round(mv_p_holm, 12),
            "triggered": mv_triggered,
        },
        "features": feature_rows,
    }
    canon = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(canon).hexdigest()
    return report


def audit_csv_file(
    csv_path: str,
    target_col: str,
    feature_cols: Sequence[str] | None = None,
    time_col: str | None = None,
    sort_by_time: bool = False,
    alpha: float = 0.01,
    ar_p: int = 6,
    causal_lags: int = 10,
    acausal_lags: int = 16,
    sumrule_horizon: int = 24,
    enable_mv_kk: bool = True,
) -> Dict[str, object]:
    """Loads CSV feature store and runs full 4-layer KK-CAUSALSPEC v2.0 multivariate audit."""
    loaded = load_feature_store_csv(
        csv_path=csv_path,
        target_col=target_col,
        feature_cols=feature_cols,
        time_col=time_col,
        sort_by_time=sort_by_time,
    )
    report = audit_feature_store(
        target=loaded["target"],
        features=loaded["features"],
        alpha=alpha,
        ar_p=ar_p,
        causal_lags=causal_lags,
        acausal_lags=acausal_lags,
        sumrule_horizon=sumrule_horizon,
        enable_mv_kk=enable_mv_kk,
        source_label=csv_path,
    )
    report["csv_metadata"] = {
        "target_col": loaded["target_col"],
        "time_col": loaded["time_col"],
        "n_raw_rows": loaded["n_raw_rows"],
        "n_valid_rows": loaded["n_valid_rows"],
        "n_dropped_rows": loaded["n_dropped_rows"],
        "timestamp_monotonic": loaded["timestamp_monotonic"],
        "non_monotonic_events": loaded["non_monotonic_events"],
    }
    # Re-seal canonical receipt including csv_metadata
    report_copy = {k: v for k, v in report.items() if k != "receipt_sha256"}
    canon = json.dumps(report_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(canon).hexdigest()
    return report


def build_sarif_report(audit_report: Mapping[str, object]) -> Dict[str, object]:
    """Builds a GitHub Code Scanning compliant SARIF 2.1.0 document from a store audit report."""
    source_uri = str(audit_report.get("source", "feature_store.csv"))
    results: List[Dict[str, object]] = []

    for row in audit_report.get("features", []):
        if row.get("verdict") == "LEAK_DETECTED":
            fname = row["feature"]
            trig = row["dominant_trigger"]
            lag = row["estimated_leak_lag"]
            p_h = row["p_store_holm"]
            q_bh = row["q_bh_fdr"]
            results.append({
                "ruleId": "LEAKSPEC-KK-UNIVARIATE",
                "level": "error",
                "message": {
                    "text": (
                        f"Lookahead leakage detected in feature '{fname}' "
                        f"(trigger={trig}, lag={lag}, p_holm={p_h}, q_bh={q_bh})."
                    )
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": source_uri},
                            "region": {"startLine": 1},
                        }
                    }
                ],
                "properties": {
                    "feature": fname,
                    "dominant_trigger": trig,
                    "estimated_leak_lag": lag,
                    "p_store_holm": p_h,
                    "q_bh_fdr": q_bh,
                },
            })

    mv = audit_report.get("mv_kk_joint", {})
    if mv.get("triggered"):
        suspects = mv.get("top_suspects", [])
        results.append({
            "ruleId": "LEAKSPEC-MV-KK-JOINT",
            "level": "error",
            "message": {
                "text": (
                    f"Multivariate joint lookahead detected by MV-KK "
                    f"(best_lag={mv.get('best_lag')}, T2={mv.get('best_t2')}, "
                    f"p_holm={mv.get('p_store_holm')}, top_suspects={suspects})."
                )
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": source_uri},
                        "region": {"startLine": 1},
                    }
                }
            ],
            "properties": {
                "best_lag": mv.get("best_lag"),
                "best_t2": mv.get("best_t2"),
                "top_suspects": suspects,
            },
        })

    return {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "LeakSpec KK-CAUSALSPEC",
                        "version": "2.0.0",
                        "rules": [
                            {
                                "id": "LEAKSPEC-KK-UNIVARIATE",
                                "shortDescription": {
                                    "text": "Univariate Kramers-Kronig spectral lookahead leakage in feature column"
                                },
                            },
                            {
                                "id": "LEAKSPEC-MV-KK-JOINT",
                                "shortDescription": {
                                    "text": "Multivariate Hotelling-Martingale joint/cancellation lookahead leakage"
                                },
                            },
                        ],
                    }
                },
                "results": results,
                "properties": {
                    "store_verdict": audit_report.get("store_verdict"),
                    "receipt_sha256": audit_report.get("receipt_sha256"),
                },
            }
        ],
    }


def _student_t_draw(rng: random.Random, df: int = 5) -> float:
    z = rng.gauss(0.0, 1.0)
    c2 = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(df))
    scale = math.sqrt((df - 2.0) / df) if df > 2 else 1.0
    return z / math.sqrt(c2 / df) * scale


def _gen_ar1_garch(rng: random.Random, n: int, phi: float = 0.88, heavy_tail: bool = True) -> List[float]:
    x = [0.0] * n
    h = 1.0
    prev_e = 0.0
    for t in range(n):
        h = 0.08 + 0.14 * (prev_e * prev_e) + 0.78 * h
        eps = _student_t_draw(rng, 5) if heavy_tail else rng.gauss(0.0, 1.0)
        e = math.sqrt(h) * eps
        x[t] = (phi * x[t - 1] if t > 0 else 0.0) + e
        prev_e = e
    return x


def _causal_ema(x: Sequence[float], alpha: float) -> List[float]:
    y = [0.0] * len(x)
    acc = 0.0
    for t, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        y[t] = acc
    return y


def _filtfilt_ema(x: Sequence[float], alpha: float) -> List[float]:
    fwd = _causal_ema(x, alpha)
    return list(reversed(_causal_ema(list(reversed(fwd)), alpha)))


def _expanding_z(xs: Sequence[float]) -> List[float]:
    out = [0.0] * len(xs)
    s1 = 0.0
    s2 = 0.0
    for i, v in enumerate(xs):
        s1 += v
        s2 += v * v
        if i >= 5:
            m = s1 / (i + 1)
            var = max(s2 / (i + 1) - m * m, 1e-6)
            out[i] = (v - m) / math.sqrt(var)
    return out


def generate_synthetic_d10_csv(
    csv_path: str,
    mode: str = "mixed",
    n_rows: int = 2000,
    seed: int = 20261125,
) -> Dict[str, object]:
    """
    Generates a realistic d=10 feature-store CSV for benchmark verification:
      - mode='mixed': 5 hard causal controls (f01..f05) + 5 distinct leak archetypes (f06..f10)
      - mode='stealth_joint': 8 causal features + 2 stealth cancellation-mode features (f09, f10)
      - mode='clean': 10 strictly causal correlated AR(1)+GARCH+Student-t features
    """
    rng = random.Random(seed)
    target = _gen_ar1_garch(rng, n_rows, phi=0.88, heavy_tail=True)
    eps = [rng.gauss(0.0, 0.45) for _ in range(n_rows)]
    features: Dict[str, List[float]] = {}
    expected_leaks: List[str] = []

    if mode == "mixed":
        # 5 Hard Causal Controls
        features["f01_causal_ema"] = [v + 0.3 * rng.gauss(0.0, 1.0) for v in _causal_ema(target, 0.25)]
        f_allpass = [0.0] * n_rows
        for t in range(3, n_rows):
            f_allpass[t] = 0.6 * f_allpass[t - 1] + target[t - 3] - 0.6 * target[t - 2] + 0.2 * eps[t]
        features["f02_causal_allpass"] = f_allpass
        f_garch = [0.0] * n_rows
        h_est = 1.0
        for t in range(1, n_rows):
            h_est = 0.08 + 0.14 * (target[t - 1] ** 2) + 0.78 * h_est
            f_garch[t] = (target[t - 1] / math.sqrt(h_est)) + 0.3 * rng.gauss(0.0, 0.6)
        features["f03_causal_garch_vol"] = f_garch
        x_shifted = [v + 3.5 for v in target]
        features["f04_causal_expanding_z"] = _expanding_z(x_shifted)
        features["f05_causal_momentum_lag3"] = [
            (target[t] - target[t - 3] if t >= 3 else 0.0) + 0.4 * rng.gauss(0.0, 1.0)
            for t in range(n_rows)
        ]

        # 5 Distinct Leak Archetypes
        features["f06_leak_shift_1bar"] = [
            0.8 * target[t] + (0.16 * target[t + 1] if t + 1 < n_rows else 0.0) + eps[t]
            for t in range(n_rows)
        ]
        features["f07_leak_filtfilt_ema"] = [
            v + 0.25 * e for v, e in zip(_filtfilt_ema(target, 0.35), eps)
        ]
        # Zero-linear-correlation nonlinear future-volatility sizing on target innovations
        innov = [target[t] - (0.88 * target[t - 1] if t > 0 else 0.0) for t in range(n_rows)]
        features["f08_leak_nl2_future_vol"] = [
            innov[t] * (1.0 + 0.95 * (abs(innov[t + 1]) if t + 1 < n_rows else 0.0)) + 0.25 * rng.gauss(0.0, 0.5)
            for t in range(n_rows)
        ]
        features["f09_leak_diffuse_24bar"] = [
            target[t] - (0.95 * _mean(target[t + 1 : min(n_rows, t + 25)]) if t + 1 < n_rows else 0.0) + 0.3 * eps[t]
            for t in range(n_rows)
        ]
        mx_s, sx_s = _mean(x_shifted), _std(x_shifted)
        features["f10_leak_global_scaler"] = [(v - mx_s) / sx_s for v in x_shifted]
        expected_leaks = [
            "f06_leak_shift_1bar",
            "f07_leak_filtfilt_ema",
            "f08_leak_nl2_future_vol",
            "f09_leak_diffuse_24bar",
            "f10_leak_global_scaler",
        ]
    elif mode == "stealth_joint":
        common_z = [rng.gauss(0.0, 8.0) for _ in range(n_rows)]
        for k in range(1, 9):
            alpha_k = 0.1 + 0.08 * k
            ema_k = _causal_ema(target, alpha_k)
            features[f"f0{k}_causal"] = [v + rng.gauss(0.0, 0.7) for v in ema_k]
        features["f09_stealth_leg_a"] = [
            common_z[t] + (0.18 * target[t + 1] if t + 1 < n_rows else 0.0) + rng.gauss(0.0, 0.25)
            for t in range(n_rows)
        ]
        features["f10_stealth_leg_b"] = [
            -common_z[t] + (0.18 * target[t + 1] if t + 1 < n_rows else 0.0) + rng.gauss(0.0, 0.25)
            for t in range(n_rows)
        ]
        expected_leaks = ["f09_stealth_leg_a", "f10_stealth_leg_b"]
    elif mode == "clean":
        common_z = [rng.gauss(0.0, 5.0) for _ in range(n_rows)]
        for k in range(1, 9):
            alpha_k = 0.08 + 0.09 * k
            ema_k = _causal_ema(target, alpha_k)
            features[f"f0{k}_causal"] = [v + rng.gauss(0.0, 0.7) for v in ema_k]
        features["f09_causal_pair_a"] = [common_z[t] + rng.gauss(0.0, 0.3) for t in range(n_rows)]
        features["f10_causal_pair_b"] = [-common_z[t] + rng.gauss(0.0, 0.3) for t in range(n_rows)]
        expected_leaks = []
    else:
        raise ValueError(f"Unsupported synthetic mode: {mode}")

    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    cols = ["ts", "target_ret"] + list(features.keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(n_rows):
            ts = (t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            row = [ts, f"{target[i]:.8f}"] + [f"{features[c][i]:.8f}" for c in features.keys()]
            writer.writerow(row)

    return {
        "csv_path": csv_path,
        "mode": mode,
        "n_rows": n_rows,
        "d_features": len(features),
        "expected_leaks": expected_leaks,
    }
