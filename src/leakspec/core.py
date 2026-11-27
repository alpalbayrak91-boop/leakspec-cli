# SPDX-License-Identifier: MIT
"""
KK-CAUSALSPEC v2.0 Core Spectral Causality Engine
=================================================
Pure Python 3.10+ (zero external dependencies, standard library only).

Layers:
  1. IP-KK (Dual Innovation-Prewhitened Linear Kramers-Kronig Hilbert Defect)
  2. NL2-KK (Channel-Whitened Scandolo-Bassani Nonlinear Envelope Scan)
  3. SUMRULE-KK (Low-Frequency f-Sum-Rule Tail + Global Pinning Forensic)
  4. MV-KK (Multivariate Cholesky-Whitened Hall-Heyde Hotelling-Martingale Joint Audit)
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Mapping, Sequence, Tuple


def fft_radix2(buf: Sequence[complex], inverse: bool = False) -> List[complex]:
    """Iterative Cooley-Tukey Radix-2 FFT for power-of-two lengths."""
    n = len(buf)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FFT length must be a positive power of two, got {n}")
    a = [complex(v) for v in buf]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    sign = 1.0 if inverse else -1.0
    length = 2
    while length <= n:
        ang = sign * 2.0 * math.pi / length
        wlen = complex(math.cos(ang), math.sin(ang))
        half = length >> 1
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1
    if inverse:
        inv_n = 1.0 / n
        a = [val * inv_n for val in a]
    return a


def discrete_kk_hilbert_freq(spectrum_real: Sequence[float]) -> List[float]:
    """Exact discrete Kramers-Kronig Hilbert operator H_KK[R] = DFT(i * sgn_M * IDFT(R))."""
    m_len = len(spectrum_real)
    half = m_len >> 1
    r_time = fft_radix2([complex(v, 0.0) for v in spectrum_real], inverse=True)
    signed = [0.0j] * m_len
    for k in range(1, half):
        signed[k] = 1j * r_time[k]
    for k in range(half + 1, m_len):
        signed[k] = -1j * r_time[k]
    h_freq = fft_radix2(signed, inverse=False)
    return [v.real for v in h_freq]


def kk_spectral_defect_from_impulse(h: Sequence[float]) -> Dict[str, float]:
    """Computes exact frequency-domain KK defect and Parseval parity against anti-causal energy."""
    m_len = len(h)
    half = m_len >> 1
    spec = fft_radix2([complex(v, 0.0) for v in h], inverse=False)
    real_p = [v.real for v in spec]
    imag_p = [v.imag for v in spec]
    hilb_r = discrete_kk_hilbert_freq(real_p)
    delta_kk = [imag_p[m] + hilb_r[m] for m in range(m_len)]
    e_kk_spec = sum(d * d for d in delta_kk) / (2.0 * m_len)
    e_acausal = sum(h[k] * h[k] for k in range(half + 1, m_len))
    e_causal = sum(h[k] * h[k] for k in range(0, half))
    e_total = e_causal + e_acausal + h[half] * h[half]
    defect_ratio = e_acausal / e_total if e_total > 1e-18 else 0.0
    return {
        "e_kk_spectral": e_kk_spec,
        "e_acausal_time": e_acausal,
        "parseval_abs_err": abs(e_kk_spec - e_acausal),
        "e_total": e_total,
        "kk_defect_ratio": defect_ratio,
    }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _validate_finite_real(val: object, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite real number, got {val!r}")
    fv = float(val)
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite (no NaN or Inf), got {val!r}")
    return fv


def _validate_int_ge(val: object, name: str, minimum: int) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {val!r}")
    return val


def _validate_finite_series(xs: Sequence[float], name: str) -> List[float]:
    out: List[float] = []
    for idx, v in enumerate(xs):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name}[{idx}] must be a finite real number, got {v!r}")
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"{name}[{idx}] must be finite (no NaN or Inf), got {v!r}")
        out.append(fv)
    return out


def _std(xs: Sequence[float], mu: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs) if mu is None else mu
    return math.sqrt(sum((v - m) ** 2 for v in xs) / len(xs))


def gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) via series + Lentz continued fraction (<1e-13)."""
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        return 0.0
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(300):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * 1e-13:
                break
        p = summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-13:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return max(0.0, min(1.0, q))


def chi2_survival(x: float, df: int) -> float:
    """Exact upper-tail survival probability P(Chi2_df >= x) for all integer df >= 1."""
    if x <= 0.0:
        return 1.0
    if df <= 0:
        return 0.0
    return gamma_q(0.5 * df, 0.5 * x)


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def fit_ar_levinson(x: Sequence[float], order: int) -> Tuple[List[float], float]:
    """Fits causal AR(order) via Yule-Walker + Levinson-Durbin with 1e-6 ridge damping."""
    n = len(x)
    mu = _mean(x)
    xc = [v - mu for v in x]
    if order <= 0 or n <= order + 4:
        return [], mu
    r = [0.0] * (order + 1)
    for k in range(order + 1):
        s = 0.0
        for i in range(n - k):
            s += xc[i] * xc[i + k]
        r[k] = s / n
    if r[0] <= 1e-15:
        return [0.0] * order, mu
    e = r[0] * (1.0 + 1e-6)
    a = [0.0] * (order + 1)
    for m in range(1, order + 1):
        num = r[m]
        for j in range(1, m):
            num -= a[j] * r[m - j]
        k_m = num / e if abs(e) > 1e-18 else 0.0
        k_m = max(-0.985, min(0.985, k_m))
        a_new = [0.0] * (order + 1)
        a_new[m] = k_m
        for j in range(1, m):
            a_new[j] = a[j] - k_m * a[m - j]
        a = a_new
        e *= (1.0 - k_m * k_m)
        if e <= 1e-15:
            break
    return a[1:], mu


def apply_causal_ar_filter(x: Sequence[float], phi: Sequence[float], mu: float) -> List[float]:
    """Applies strictly causal whitening filter A(L) = 1 - sum phi_j L^j to (x - mu)."""
    p = len(phi)
    xc = [v - mu for v in x]
    n = len(xc)
    out = [0.0] * n
    for t in range(n):
        val = xc[t]
        lim = p if t >= p else t
        for j in range(lim):
            val -= phi[j] * xc[t - 1 - j]
        out[t] = val
    return out


def solve_linear_system(a_mat: List[List[float]], b_vec: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b_vec)
    mat = [row[:] + [b_vec[i]] for i, row in enumerate(a_mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[piv][col]) < 1e-14:
            continue
        mat[col], mat[piv] = mat[piv], mat[col]
        inv_p = 1.0 / mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] *= inv_p
        for r in range(n):
            if r != col and mat[r][col] != 0.0:
                f = mat[r][col]
                for j in range(col, n + 1):
                    mat[r][j] -= f * mat[col][j]
    return [mat[i][n] for i in range(n)]


def invert_spd_cholesky(a_mat: Sequence[Sequence[float]]) -> List[List[float]]:
    """Invert symmetric positive-definite matrix via Cholesky decomposition."""
    n = len(a_mat)
    chol = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(chol[i][k] * chol[j][k] for k in range(j))
            if i == j:
                val = a_mat[i][i] - s
                chol[i][j] = math.sqrt(val if val > 1e-14 else 1e-14)
            else:
                chol[i][j] = (a_mat[i][j] - s) / chol[j][j]
    linv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        linv[i][i] = 1.0 / chol[i][i]
        for j in range(i):
            s = sum(chol[i][k] * linv[k][j] for k in range(j, i))
            linv[i][j] = -s / chol[i][i]
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = sum(linv[k][i] * linv[k][j] for k in range(j, n))
            inv[i][j] = s
            inv[j][i] = s
    return inv


def causal_ols_residual(u: Sequence[float], v: Sequence[float], causal_lags: int) -> Tuple[List[float], List[float]]:
    """Projects v[t] onto strictly causal lags u[t..t-causal_lags] with 1e-6 ridge."""
    n = len(u)
    k_dim = causal_lags + 1
    start = causal_lags
    if n <= start + k_dim + 4:
        return list(v), [0.0] * k_dim
    xtx = [[0.0] * k_dim for _ in range(k_dim)]
    xty = [0.0] * k_dim
    for t in range(start, n):
        vt = v[t]
        for i in range(k_dim):
            ui = u[t - i]
            xty[i] += ui * vt
            for j in range(i, k_dim):
                xtx[i][j] += ui * u[t - j]
    for i in range(k_dim):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
        xtx[i][i] += 1e-6 * (n - start)
    beta = solve_linear_system(xtx, xty)
    res = [0.0] * n
    for t in range(start, n):
        pred = 0.0
        for i in range(k_dim):
            pred += beta[i] * u[t - i]
        res[t] = v[t] - pred
    return res, beta


def _martingale_kk_channel_scan(
    x: Sequence[float],
    y: Sequence[float],
    ar_p: int = 6,
    causal_lags: int = 10,
    acausal_lags: int = 16,
    sumrule_horizon: int = 24,
) -> Dict[str, object]:
    """Dual-whitened linear Kramers-Kronig channel scan with Hall-Heyde self-normalized Z."""
    n = min(len(x), len(y))
    phi_x, mu_x = fit_ar_levinson(x[:n], ar_p)
    u_full = apply_causal_ar_filter(x[:n], phi_x, mu_x)
    mu_y = _mean(y[:n])
    v_full = apply_causal_ar_filter(y[:n], phi_x, mu_y)

    burn = ar_p + causal_lags + 2
    r_full, beta_causal = causal_ols_residual(u_full, v_full, causal_lags)
    phi_r, mu_r = fit_ar_levinson(r_full[burn:], ar_p)
    w_full = apply_causal_ar_filter(r_full, phi_r, mu_r)

    start = burn + ar_p
    max_tau = max(acausal_lags, sumrule_horizon)
    z_profile: List[float] = []
    h_acausal: List[float] = []
    u_slice = u_full[start:]
    var_u = max(sum(val * val for val in u_slice) / max(1, len(u_slice)), 1e-15)

    for tau in range(1, max_tau + 1):
        end = n - tau
        if end <= start + 10:
            z_profile.append(0.0)
            h_acausal.append(0.0)
            continue
        s1 = 0.0
        s2 = 0.0
        for t in range(start, end):
            prod = w_full[t] * u_full[t + tau]
            s1 += prod
            s2 += prod * prod
        z_val = s1 / math.sqrt(s2) if s2 > 1e-18 else 0.0
        z_profile.append(z_val)
        h_acausal.append((s1 / (end - start)) / var_u)

    m_len = 128
    h_cyc = [0.0] * m_len
    for k, b in enumerate(beta_causal):
        if k < m_len // 2:
            h_cyc[k] = b
    for tau in range(1, min(acausal_lags + 1, m_len // 2)):
        h_cyc[m_len - tau] = h_acausal[tau - 1]
    kk_spec = kk_spectral_defect_from_impulse(h_cyc)

    z_ac = z_profile[:acausal_lags]
    q_kk = sum(z * z for z in z_ac)
    p_q = chi2_survival(q_kk, acausal_lags)
    best_idx = max(range(acausal_lags), key=lambda i: abs(z_ac[i]))
    max_abs_z = abs(z_ac[best_idx])
    best_tau = best_idx + 1
    p_peak = min(1.0, acausal_lags * normal_two_sided_p(max_abs_z))

    h_sr = min(sumrule_horizon, len(z_profile))
    z_sumrule = sum(z_profile[:h_sr]) / math.sqrt(max(1, h_sr))
    p_sumrule = normal_two_sided_p(z_sumrule)

    return {
        "q_kk": q_kk,
        "p_q": p_q,
        "max_abs_z": max_abs_z,
        "best_lag_tau": -best_tau,
        "p_peak": p_peak,
        "z_sumrule": z_sumrule,
        "p_sumrule": p_sumrule,
        "kk_spectral": kk_spec,
        "z_profile_first8": [round(v, 3) for v in z_profile[:8]],
    }


def _global_pinning_forensic(x: Sequence[float], y: Sequence[float]) -> Dict[str, object]:
    """Detects full-sample StandardScaler (mean=0, std=1) or MinMaxScaler ([0, 1]) pinning."""
    n = len(y)
    half = n // 2
    if half < 20:
        return {"pinned": False, "mode": "NONE"}
    mu_all = _mean(y)
    sd_all = _std(y, mu_all)
    mu_h1 = _mean(y[:half])
    sd_h1 = _std(y[:half], mu_h1)
    mu_x = _mean(x)
    sd_x = _std(x, mu_x)

    zscore_pin = (
        abs(mu_all) < 1e-6
        and abs(sd_all - 1.0) < 1e-6
        and (abs(mu_h1) > 1e-3 or abs(sd_h1 - 1.0) > 1e-3)
        and (abs(mu_x) > 1e-3 or abs(sd_x - 1.0) > 1e-3)
    )
    y_min = min(y)
    y_max = max(y)
    argmin_idx = y.index(y_min)
    argmax_idx = y.index(y_max)
    minmax_pin = (
        abs(y_min) < 1e-7
        and abs(y_max - 1.0) < 1e-7
        and (argmin_idx >= half or argmax_idx >= half)
        and (min(x) < -1e-3 or max(x) > 1.0 + 1e-3)
    )
    mode = "GLOBAL_ZSCORE_PIN" if zscore_pin else ("GLOBAL_MINMAX_PIN" if minmax_pin else "NONE")
    return {
        "pinned": zscore_pin or minmax_pin,
        "mode": mode,
        "mu_all": mu_all,
        "sd_all": sd_all,
        "mu_first_half": mu_h1,
        "sd_first_half": sd_h1,
    }


def audit_causality(
    x: Sequence[float],
    y: Sequence[float],
    alpha: float = 0.01,
    ar_p: int = 6,
    causal_lags: int = 10,
    acausal_lags: int = 16,
    sumrule_horizon: int = 24,
) -> Dict[str, object]:
    """Full 3-Layer Univariate KK-CAUSALSPEC Audit (IP-KK + NL2-KK + SUMRULE-KK/Pinning)."""
    alpha_f = _validate_finite_real(alpha, "alpha")
    if not (0.0 < alpha_f < 1.0):
        raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha!r}")
    _validate_int_ge(ar_p, "ar_p", 0)
    _validate_int_ge(causal_lags, "causal_lags", 0)
    _validate_int_ge(acausal_lags, "acausal_lags", 1)
    _validate_int_ge(sumrule_horizon, "sumrule_horizon", 1)
    x_clean = _validate_finite_series(x, "x")
    y_clean = _validate_finite_series(y, "y")

    n = min(len(x_clean), len(y_clean))
    x_seq = x_clean[:n]
    y_seq = y_clean[:n]

    l1 = _martingale_kk_channel_scan(
        x_seq, y_seq, ar_p=ar_p, causal_lags=causal_lags, acausal_lags=acausal_lags, sumrule_horizon=sumrule_horizon
    )

    phi_x, mu_x = fit_ar_levinson(x_seq, ar_p)
    u = apply_causal_ar_filter(x_seq, phi_x, mu_x)
    v = apply_causal_ar_filter(y_seq, phi_x, _mean(y_seq))
    r, _ = causal_ols_residual(u, v, causal_lags)

    u_sq = [val * val for val in u]
    u_abs = [abs(val) for val in u]
    v_sq = [val * val for val in v]
    v_abs = [abs(val) for val in v]
    r_sq = [val * val for val in r]

    nl_channels = {
        "u2_to_v2": _martingale_kk_channel_scan(u_sq, v_sq, ar_p=ar_p, causal_lags=causal_lags, acausal_lags=acausal_lags, sumrule_horizon=sumrule_horizon),
        "uabs_to_vabs": _martingale_kk_channel_scan(u_abs, v_abs, ar_p=ar_p, causal_lags=causal_lags, acausal_lags=acausal_lags, sumrule_horizon=sumrule_horizon),
        "u2_to_v": _martingale_kk_channel_scan(u_sq, v, ar_p=ar_p, causal_lags=causal_lags, acausal_lags=acausal_lags, sumrule_horizon=sumrule_horizon),
        "u_to_r2": _martingale_kk_channel_scan(u, r_sq, ar_p=ar_p, causal_lags=causal_lags, acausal_lags=acausal_lags, sumrule_horizon=sumrule_horizon),
    }

    pin = _global_pinning_forensic(x_seq, y_seq)

    candidates = [
        ("L1_IP_KK_ENERGY", float(l1["p_q"]), int(l1["best_lag_tau"]), float(l1["max_abs_z"])),
        ("L1_IP_KK_PEAK", float(l1["p_peak"]), int(l1["best_lag_tau"]), float(l1["max_abs_z"])),
        ("L3_SUMRULE_KK", float(l1["p_sumrule"]), -sumrule_horizon, abs(float(l1["z_sumrule"]))),
    ]
    for cname, res in nl_channels.items():
        candidates.append((f"L2_NL2_{cname}_ENERGY", float(res["p_q"]), int(res["best_lag_tau"]), float(res["max_abs_z"])))
        candidates.append((f"L2_NL2_{cname}_PEAK", float(res["p_peak"]), int(res["best_lag_tau"]), float(res["max_abs_z"])))

    m_tests = len(candidates)
    best_test = min(candidates, key=lambda item: item[1])
    p_fwer = min(1.0, best_test[1] * m_tests)

    triggered_layers: List[str] = []
    for name, pval, _, _ in candidates:
        if pval * m_tests < alpha:
            triggered_layers.append(name)
    if pin["pinned"]:
        triggered_layers.append(f"L3_{pin['mode']}")

    verdict = "LEAK_DETECTED" if triggered_layers else "CAUSAL_CLEAN"
    payload: Dict[str, object] = {
        "verdict": verdict,
        "n_samples": n,
        "alpha_fwer": alpha,
        "p_fwer_min": p_fwer,
        "dominant_trigger": f"L3_{pin['mode']}" if (pin["pinned"] and p_fwer >= alpha) else best_test[0],
        "estimated_leak_lag": 0 if (pin["pinned"] and p_fwer >= alpha) else best_test[2],
        "dominant_z": best_test[3],
        "triggered_layers": triggered_layers,
        "linear_kk": l1,
        "nl2_summary": {
            k: {
                "p_q": round(float(v["p_q"]), 6),
                "p_peak": round(float(v["p_peak"]), 6),
                "best_lag": int(v["best_lag_tau"]),
                "max_z": round(float(v["max_abs_z"]), 3),
            }
            for k, v in nl_channels.items()
        },
        "global_pinning": pin,
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def mv_kk_joint_audit(
    target: Sequence[float],
    features: Mapping[str, Sequence[float]],
    ar_p: int = 6,
    causal_lags: int = 10,
    acausal_lags: int = 16,
) -> Dict[str, object]:
    """
    Layer 4 MV-KK: Multivariate Cholesky-Whitened Hall-Heyde Hotelling-Martingale Audit.
    Detects stealth joint / difference-mode lookahead leakage across d features that
    remains invisible to univariate feature-by-feature screening.
    """
    _validate_int_ge(ar_p, "ar_p", 0)
    _validate_int_ge(causal_lags, "causal_lags", 0)
    _validate_int_ge(acausal_lags, "acausal_lags", 1)
    target_clean = _validate_finite_series(target, "target")
    features_clean: Dict[str, List[float]] = {
        fname: _validate_finite_series(fseries, f"features[{fname}]")
        for fname, fseries in features.items()
    }

    names = list(features_clean.keys())
    d = len(names)
    n = len(target_clean)
    burn = ar_p + causal_lags + 2
    start = burn + ar_p
    if d == 0 or n <= start + acausal_lags + d + 8:
        return {
            "active": False,
            "reason": "INSUFFICIENT_DIMENSIONS_OR_SAMPLES",
            "d_active": d,
            "p_mv_peak": 1.0,
            "p_mv_total": 1.0,
            "p_mv_min": 1.0,
            "best_lag": 0,
            "best_t2": 0.0,
            "q_total": 0.0,
            "attribution_z_at_best_lag": {},
        }

    phi_x, mu_x = fit_ar_levinson(target_clean, ar_p)
    u = apply_causal_ar_filter(target_clean, phi_x, mu_x)

    w_cols: List[List[float]] = []
    for name in names:
        y = features_clean[name]
        v = apply_causal_ar_filter(y, phi_x, _mean(y))
        r, _ = causal_ols_residual(u, v, causal_lags)
        phi_r, mu_r = fit_ar_levinson(r[burn:], ar_p)
        w = apply_causal_ar_filter(r, phi_r, mu_r)
        sd = _std(w[start:])
        scale = 1.0 / sd if sd > 1e-12 else 0.0
        w_cols.append([val * scale for val in w])

    t_len = n - start
    sigma = [[0.0] * d for _ in range(d)]
    for i in range(d):
        wi = w_cols[i]
        for j in range(i, d):
            wj = w_cols[j]
            s = sum(wi[t] * wj[t] for t in range(start, n)) / t_len
            sigma[i][j] = s
            sigma[j][i] = s
    for i in range(d):
        sigma[i][i] += 1e-5
    sigma_inv = invert_spd_cholesky(sigma)

    t2_by_lag: List[float] = []
    best_tau = 1
    best_t2 = -1.0
    best_attr: Dict[str, float] = {}

    for tau in range(1, acausal_lags + 1):
        end = n - tau
        t_eff = end - start
        score = [0.0] * d
        v_mat = [[0.0] * d for _ in range(d)]
        for t in range(start, end):
            ut = u[t + tau]
            g = [w_cols[i][t] * ut for i in range(d)]
            for i in range(d):
                gi = g[i]
                score[i] += gi
                for j in range(i, d):
                    v_mat[i][j] += gi * g[j]
        tr_v = sum(v_mat[i][i] for i in range(d)) / max(1, d)
        ridge_v = 1e-6 * max(tr_v, 1e-12)
        for i in range(d):
            v_mat[i][i] += ridge_v
            for j in range(i):
                v_mat[i][j] = v_mat[j][i]
        v_inv = invert_spd_cholesky(v_mat)
        t2_raw = 0.0
        for i in range(d):
            row_s = sum(v_inv[i][j] * score[j] for j in range(d))
            t2_raw += score[i] * row_s
        t2_adj = t2_raw * max(0.5, (t_eff - d) / float(t_eff))
        t2_by_lag.append(t2_adj)
        if t2_adj > best_t2:
            best_t2 = t2_adj
            best_tau = tau
            beta = [sum(sigma_inv[i][j] * (score[j] / t_eff) for j in range(d)) for i in range(d)]
            tmp = [[sum(sigma_inv[i][k] * v_mat[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
            attr_z: Dict[str, float] = {}
            for i in range(d):
                var_bi = sum(tmp[i][k] * sigma_inv[k][i] for k in range(d)) / (t_eff * t_eff)
                attr_z[names[i]] = round(beta[i] / math.sqrt(max(var_bi, 1e-18)), 3)
            best_attr = attr_z

    p_peak = min(1.0, acausal_lags * chi2_survival(best_t2, d))
    q_total = sum(t2_by_lag)
    p_total = chi2_survival(q_total, d * acausal_lags)
    p_mv_min = min(1.0, 2.0 * min(p_peak, p_total))

    ranked_attr = sorted(best_attr.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_suspects = [k for k, z in ranked_attr if abs(z) >= 3.2]

    return {
        "active": True,
        "d_active": d,
        "p_mv_peak": p_peak,
        "p_mv_total": p_total,
        "p_mv_min": p_mv_min,
        "best_lag": -best_tau,
        "best_t2": round(best_t2, 3),
        "df_peak": d,
        "q_total": round(q_total, 3),
        "df_total": d * acausal_lags,
        "top_suspects": top_suspects,
        "attribution_z_at_best_lag": best_attr,
    }
