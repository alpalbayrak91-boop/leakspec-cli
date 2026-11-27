# SPDX-License-Identifier: MIT
"""
Layer 1 (v2.7.0): Lopez de Prado (2018) Combinatorial Purged Cross-Validation (CPCV),
Exact Post-Horizon Effective Embargo Auditor (post_purge_anchor = max_test_label_end),
Combinatorial Backtest Path Reconstruction, and Bailey & Lopez de Prado (2017)
Probability of Backtest Overfitting (PBO) Rank-Logit Estimator.

Zero external dependencies (Pure Python 3.10+ standard library).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def _is_finite_real(val: object) -> bool:
    return (not isinstance(val, bool)) and isinstance(val, (int, float)) and math.isfinite(float(val))


def _require_finite_real(val: object, name: str) -> float:
    if not _is_finite_real(val):
        raise ValueError(f"{name} must be a finite real number (no NaN, Inf, or bool), got {val!r}")
    return float(val)


def _require_int(val: object, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an integer (not bool), got {val!r}")
    return val


@dataclass(frozen=True)
class LabelSpan:
    """Point-in-time observation and forward label evaluation span [t_start, t_end]."""

    obs_index: int
    t_start: float
    t_end: float
    t_avail: Optional[float] = None

    def __post_init__(self) -> None:
        _require_int(self.obs_index, "obs_index")
        _require_finite_real(self.t_start, "t_start")
        _require_finite_real(self.t_end, "t_end")
        if self.t_avail is not None:
            _require_finite_real(self.t_avail, "t_avail")

    def overlaps(self, other: "LabelSpan") -> bool:
        """True iff closed intervals [t_start, t_end] intersect."""
        return (self.t_start <= other.t_end) and (other.t_start <= self.t_end)


@dataclass(frozen=True)
class CPCVSplit:
    """Single purged and post-horizon embargoed CPCV(N, k) split."""

    split_id: int
    test_groups: Tuple[int, ...]
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    purged_indices: Tuple[int, ...]
    embargoed_indices: Tuple[int, ...]


def cpcv_num_splits(n_groups: int, n_test_groups: int) -> int:
    """Total number of CPCV splits: C(N, k)."""
    _require_int(n_groups, "n_groups")
    _require_int(n_test_groups, "n_test_groups")
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError(
            f"Require 1 <= n_test_groups < n_groups and n_groups >= 2; "
            f"got N={n_groups}, k={n_test_groups}"
        )
    return math.comb(n_groups, n_test_groups)


def cpcv_num_backtest_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of combinatorial backtest paths: phi(N, k) = (k / N) * C(N, k) = C(N - 1, k - 1)."""
    _require_int(n_groups, "n_groups")
    _require_int(n_test_groups, "n_test_groups")
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError(
            f"Require 1 <= n_test_groups < n_groups and n_groups >= 2; "
            f"got N={n_groups}, k={n_test_groups}"
        )
    return math.comb(n_groups - 1, n_test_groups - 1)


def reconstruct_cpcv_paths(
    n_groups: int,
    n_test_groups: int,
    splits: Sequence[CPCVSplit],
) -> Dict[str, object]:
    """
    Lopez de Prado (2018, Ch. 12) Combinatorial Backtest Path Assembly.
    Maps each split's k test groups onto phi(N, k) = C(N - 1, k - 1) complete,
    chronologically ordered out-of-sample backtest paths without collision.
    """
    num_paths = cpcv_num_backtest_paths(n_groups, n_test_groups)
    expected_splits = cpcv_num_splits(n_groups, n_test_groups)
    if len(splits) != expected_splits:
        raise ValueError(f"Expected {expected_splits} splits, got {len(splits)}")

    path_matrix: List[List[Optional[int]]] = [[None] * num_paths for _ in range(n_groups)]
    next_slot = [0] * n_groups

    for sp in splits:
        for g in sp.test_groups:
            slot = next_slot[g]
            if slot >= num_paths:
                raise RuntimeError(f"Group {g} exceeded path capacity {num_paths}")
            path_matrix[g][slot] = sp.split_id
            next_slot[g] += 1

    paths: List[Dict[int, int]] = []
    for p in range(num_paths):
        mapping: Dict[int, int] = {}
        for g in range(n_groups):
            sid = path_matrix[g][p]
            if sid is None:
                raise RuntimeError(f"Incomplete backtest path {p} at group {g}")
            mapping[g] = sid
        paths.append(mapping)

    return {
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "num_splits": expected_splits,
        "num_backtest_paths": num_paths,
        "paths": paths,
        "coverage_verified": all(len(p) == n_groups for p in paths) and len(paths) == num_paths,
    }


def _partition_groups(n_samples: int, n_groups: int) -> List[List[int]]:
    _require_int(n_samples, "n_samples")
    _require_int(n_groups, "n_groups")
    if n_groups < 1:
        raise ValueError(f"n_groups ({n_groups}) must be >= 1")
    if n_samples < n_groups:
        raise ValueError(f"n_samples ({n_samples}) must be >= n_groups ({n_groups})")
    base = n_samples // n_groups
    rem = n_samples % n_groups
    groups: List[List[int]] = []
    cursor = 0
    for g in range(n_groups):
        size = base + (1 if g < rem else 0)
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _contiguous_blocks(sorted_indices: Sequence[int]) -> List[Tuple[int, int]]:
    if not sorted_indices:
        return []
    blocks: List[Tuple[int, int]] = []
    start = prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            blocks.append((start, prev))
            start = prev = idx
    blocks.append((start, prev))
    return blocks


def compute_post_purge_anchor(
    spans: Sequence[LabelSpan],
    b_end: int,
    max_test_label_end: float,
    test_set: set[int],
) -> int:
    """
    Computes the post-horizon purge anchor index >= b_end corresponding to
    post_purge_anchor = max_test_label_end (Lopez de Prado 2018, Section 7.4).

    When label horizon H >= embargo_bars, counting embargo bars from b_end
    would collapse post-test embargo into the already-purged horizon [b_end + 1, anchor].
    Anchoring at the last observation whose t_start <= max_test_label_end guarantees
    a strictly positive post-horizon embargo buffer after purging completes.
    """
    post_purge_anchor = b_end
    n_samples = len(spans)
    for j in range(b_end + 1, n_samples):
        if j in test_set:
            continue
        if spans[j].t_start <= max_test_label_end:
            post_purge_anchor = j
    return post_purge_anchor


_post_purge_anchor = compute_post_purge_anchor


def generate_cpcv_splits(
    spans: Sequence[LabelSpan],
    n_groups: int = 6,
    n_test_groups: int = 2,
    pct_embargo: float = 0.01,
    embargo_bars: Optional[int] = None,
    embargo_time: float = 0.0,
) -> Dict[str, object]:
    """
    Generates Lopez de Prado (2018) CPCV(N, k) splits with:
      1. Exact span-level purging across all test-block envelopes [t_min, max_test_label_end],
      2. Post-horizon effective embargo starting strictly after post_purge_anchor = max_test_label_end,
      3. Complete combinatorial backtest path reconstruction.
    """
    # Validate CPCV(N, k) group combinatorics and scalar embargo parameters at entry
    total_splits = cpcv_num_splits(n_groups, n_test_groups)
    total_paths = cpcv_num_backtest_paths(n_groups, n_test_groups)

    pct_embargo_f = _require_finite_real(pct_embargo, "pct_embargo")
    if not (0.0 <= pct_embargo_f < 0.5):
        raise ValueError(f"pct_embargo must be in [0.0, 0.5), got {pct_embargo!r}")

    embargo_time_f = _require_finite_real(embargo_time, "embargo_time")
    if embargo_time_f < 0.0:
        raise ValueError(f"embargo_time must be >= 0.0, got {embargo_time!r}")

    n_samples = len(spans)
    if n_samples < n_groups:
        raise ValueError(f"Need at least {n_groups} samples, got {n_samples}")

    if embargo_bars is not None:
        h_embargo = _require_int(embargo_bars, "embargo_bars")
        if h_embargo < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars!r}")
    else:
        h_embargo = int(math.ceil(n_samples * pct_embargo_f))

    for i, sp in enumerate(spans):
        _require_int(sp.obs_index, f"spans[{i}].obs_index")
        ts_val = _require_finite_real(sp.t_start, f"spans[{i}].t_start")
        te_val = _require_finite_real(sp.t_end, f"spans[{i}].t_end")
        if sp.t_avail is not None:
            _require_finite_real(sp.t_avail, f"spans[{i}].t_avail")
        if te_val < ts_val:
            raise ValueError(f"Invalid span at index {i}: t_end ({te_val}) < t_start ({ts_val})")

    embargo_time = embargo_time_f

    groups = _partition_groups(n_samples, n_groups)
    combos = list(itertools.combinations(range(n_groups), n_test_groups))
    splits: List[CPCVSplit] = []

    for split_id, test_grp_tuple in enumerate(combos):
        test_set: set[int] = set()
        for g in test_grp_tuple:
            test_set.update(groups[g])
        test_sorted = sorted(test_set)
        test_blocks = _contiguous_blocks(test_sorted)

        block_envelopes: List[Tuple[int, int, float, float, int]] = []
        for b_start, b_end in test_blocks:
            t_min = min(spans[i].t_start for i in range(b_start, b_end + 1))
            max_test_label_end = max(spans[i].t_end for i in range(b_start, b_end + 1))
            post_purge_anchor = compute_post_purge_anchor(
                spans, b_end, max_test_label_end, test_set
            )
            block_envelopes.append(
                (b_start, b_end, t_min, max_test_label_end, post_purge_anchor)
            )

        purged_set: set[int] = set()
        for idx in range(n_samples):
            if idx in test_set:
                continue
            sp = spans[idx]
            for _, _, t_min, max_test_label_end, _ in block_envelopes:
                if sp.t_start <= max_test_label_end and t_min <= sp.t_end:
                    purged_set.add(idx)
                    break

        embargo_idx_set: set[int] = set()
        for _, _, _, max_test_label_end, post_purge_anchor in block_envelopes:
            upper_bar = min(n_samples, post_purge_anchor + h_embargo + 1)
            for cand in range(post_purge_anchor + 1, upper_bar):
                if cand not in test_set and cand not in purged_set:
                    embargo_idx_set.add(cand)
            if embargo_time > 0.0:
                for j in range(post_purge_anchor + 1, n_samples):
                    if j in test_set or j in purged_set:
                        continue
                    if (spans[j].t_start - max_test_label_end) < embargo_time:
                        embargo_idx_set.add(j)
                    elif spans[j].t_start >= max_test_label_end + embargo_time:
                        break

        train_clean = [
            idx
            for idx in range(n_samples)
            if idx not in test_set and idx not in purged_set and idx not in embargo_idx_set
        ]
        splits.append(
            CPCVSplit(
                split_id=split_id,
                test_groups=test_grp_tuple,
                train_indices=tuple(train_clean),
                test_indices=tuple(test_sorted),
                purged_indices=tuple(sorted(purged_set)),
                embargoed_indices=tuple(sorted(embargo_idx_set)),
            )
        )

    path_info = reconstruct_cpcv_paths(n_groups, n_test_groups, splits)
    return {
        "n_samples": n_samples,
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "num_splits": total_splits,
        "num_backtest_paths": total_paths,
        "embargo_bars": h_embargo,
        "embargo_time": embargo_time,
        "post_horizon_embargo": True,
        "backtest_paths": path_info["paths"],
        "path_coverage_verified": path_info["coverage_verified"],
        "splits": splits,
    }


def audit_cpcv_splits(
    spans: Sequence[LabelSpan],
    splits: Sequence[Tuple[Sequence[int], Sequence[int]]],
    min_embargo_bars: int = 1,
    min_embargo_time: float = 0.0,
) -> Dict[str, object]:
    """
    Audits user-supplied (train_indices, test_indices) splits against:
      - LS-TIME-001: Temporal monotonicity / point-in-time availability / non-finite timestamp
      - LS-TIME-002: Purging violation (overlapping train/test spans or shared index)
      - LS-TIME-003: Post-horizon effective embargo violation after post_purge_anchor
    """
    _require_int(min_embargo_bars, "min_embargo_bars")
    if min_embargo_bars < 0:
        raise ValueError(f"min_embargo_bars must be >= 0, got {min_embargo_bars!r}")
    min_embargo_time_f = _require_finite_real(min_embargo_time, "min_embargo_time")
    if min_embargo_time_f < 0.0:
        raise ValueError(f"min_embargo_time must be >= 0.0, got {min_embargo_time!r}")
    min_embargo_time = min_embargo_time_f

    n_samples = len(spans)
    findings: List[Dict[str, object]] = []
    nonfinite_indices: set[int] = set()

    for i, sp in enumerate(spans):
        obs_id = sp.obs_index if (not isinstance(sp.obs_index, bool) and isinstance(sp.obs_index, int)) else i
        finite_ok = (
            _is_finite_real(sp.t_start)
            and _is_finite_real(sp.t_end)
            and (sp.t_avail is None or _is_finite_real(sp.t_avail))
        )
        if not finite_ok:
            nonfinite_indices.add(i)
            findings.append({
                "rule_id": "LS-TIME-001",
                "severity": "error",
                "split_id": -1,
                "obs_index": obs_id,
                "detail": (
                    f"Non-finite or non-real timestamp at index {i}: "
                    f"t_start={sp.t_start!r}, t_end={sp.t_end!r}, t_avail={sp.t_avail!r}."
                ),
            })
            continue
        if sp.t_end < sp.t_start:
            findings.append({
                "rule_id": "LS-TIME-001",
                "severity": "error",
                "split_id": -1,
                "obs_index": obs_id,
                "detail": f"Invalid label span at index {i}: t_end ({sp.t_end}) < t_start ({sp.t_start}).",
            })
        if sp.t_avail is not None and sp.t_avail > sp.t_start:
            findings.append({
                "rule_id": "LS-TIME-001",
                "severity": "error",
                "split_id": -1,
                "obs_index": obs_id,
                "detail": (
                    f"Point-in-time violation at index {i}: feature t_avail ({sp.t_avail}) "
                    f"> decision t_start ({sp.t_start})."
                ),
            })
        if i > 0 and (i - 1) not in nonfinite_indices and sp.t_start < spans[i - 1].t_start:
            findings.append({
                "rule_id": "LS-TIME-001",
                "severity": "error",
                "split_id": -1,
                "obs_index": obs_id,
                "detail": (
                    f"Non-monotonic timestamp ordering at index {i}: "
                    f"t_start ({sp.t_start}) < prev ({spans[i - 1].t_start})."
                ),
            })

    total_purge_violations = 0
    total_embargo_violations = 0
    split_summaries: List[Dict[str, object]] = []

    if nonfinite_indices:
        return {
            "layer": "L1_CPCV_EMBARGO",
            "n_samples": n_samples,
            "n_splits_audited": len(splits),
            "min_embargo_bars": min_embargo_bars,
            "min_embargo_time": min_embargo_time,
            "post_horizon_embargo": True,
            "total_purge_violations": 0,
            "total_embargo_violations": 0,
            "verdict": "LEAK_DETECTED",
            "splits": [],
            "findings": findings,
        }

    for split_id, (train_idx_raw, test_idx_raw) in enumerate(splits):
        train_set = {int(i) for i in train_idx_raw}
        test_set = {int(i) for i in test_idx_raw}
        test_sorted = sorted(test_set)
        train_sorted = sorted(train_set)

        direct_overlap = sorted(train_set & test_set)
        purge_violations: List[Tuple[int, int]] = [(idx, idx) for idx in direct_overlap]
        embargo_violations: List[Tuple[int, int]] = []

        test_blocks = _contiguous_blocks(test_sorted)
        block_meta: List[Tuple[int, int, float, float, int]] = []
        for b_start, b_end in test_blocks:
            t_min = min(spans[i].t_start for i in range(b_start, b_end + 1))
            max_test_label_end = max(spans[i].t_end for i in range(b_start, b_end + 1))
            post_purge_anchor = compute_post_purge_anchor(
                spans, b_end, max_test_label_end, test_set
            )
            block_meta.append(
                (b_start, b_end, t_min, max_test_label_end, post_purge_anchor)
            )

        for tr in train_sorted:
            if tr in test_set:
                continue
            sp_tr = spans[tr]
            for b_start, b_end, t_min, max_test_label_end, post_purge_anchor in block_meta:
                if sp_tr.t_start <= max_test_label_end and t_min <= sp_tr.t_end:
                    witness = b_start
                    for te in range(b_start, b_end + 1):
                        if sp_tr.overlaps(spans[te]):
                            witness = te
                            break
                    purge_violations.append((tr, witness))
                elif tr > post_purge_anchor:
                    eff_bar_gap = tr - post_purge_anchor
                    time_gap = sp_tr.t_start - max_test_label_end
                    if (min_embargo_bars > 0 and eff_bar_gap <= min_embargo_bars) or (
                        min_embargo_time > 0.0 and time_gap < min_embargo_time
                    ):
                        embargo_violations.append((tr, post_purge_anchor))

        total_purge_violations += len(purge_violations)
        total_embargo_violations += len(embargo_violations)

        if purge_violations:
            sample = purge_violations[:5]
            findings.append({
                "rule_id": "LS-TIME-002",
                "severity": "error",
                "split_id": split_id,
                "count": len(purge_violations),
                "witness_pairs": sample,
                "detail": (
                    f"Split {split_id}: {len(purge_violations)} CPCV purging violation(s) "
                    f"(train/test label-span overlap, sample train->test pairs={sample})."
                ),
            })

        if embargo_violations:
            sample = embargo_violations[:5]
            findings.append({
                "rule_id": "LS-TIME-003",
                "severity": "warning",
                "split_id": split_id,
                "count": len(embargo_violations),
                "witness_pairs": sample,
                "detail": (
                    f"Split {split_id}: {len(embargo_violations)} post-horizon CPCV embargo violation(s) "
                    f"(train obs within {min_embargo_bars} bar(s) after purged horizon, sample={sample})."
                ),
            })

        split_summaries.append({
            "split_id": split_id,
            "n_train": len(train_sorted),
            "n_test": len(test_sorted),
            "direct_overlap_count": len(direct_overlap),
            "purge_violation_count": len(purge_violations),
            "embargo_violation_count": len(embargo_violations),
            "clean": (len(purge_violations) == 0 and len(embargo_violations) == 0),
        })

    verdict = "LEAK_DETECTED" if findings else "CPCV_CLEAN"
    return {
        "layer": "L1_CPCV_EMBARGO",
        "n_samples": n_samples,
        "n_splits_audited": len(splits),
        "min_embargo_bars": min_embargo_bars,
        "min_embargo_time": min_embargo_time,
        "post_horizon_embargo": True,
        "total_purge_violations": total_purge_violations,
        "total_embargo_violations": total_embargo_violations,
        "verdict": verdict,
        "splits": split_summaries,
        "findings": findings,
    }


def estimate_cpcv_pbo(
    is_scores_by_split: Sequence[Sequence[float]],
    oos_scores_by_split: Sequence[Sequence[float]],
) -> Dict[str, object]:
    """
    Bailey, Borwein, Lopez de Prado & Zhu (2017) Probability of Backtest Overfitting (PBO).

    Given S CPCV splits across M >= 2 candidate strategy configurations:
      - In split s, select in-sample optimal strategy m* = argmax_m IS[s][m].
      - Compute symmetric 1-based Weibull plotting-position rank in OOS[s]:
            rank = less + 0.5 * (equal + 1)
            omega_s = rank / (M + 1) in (0, 1)
      - Logit lambda_s = ln(omega_s / (1 - omega_s)), which is 0.0 at exact median
        and satisfies lambda(min) = -lambda(max).
      - PBO = fraction of splits where lambda_s < 0.0 (IS winner underperforms OOS median).
    """
    if not is_scores_by_split or len(is_scores_by_split) != len(oos_scores_by_split):
        raise ValueError("IS and OOS score matrices must be non-empty with equal split count.")
    n_splits = len(is_scores_by_split)
    n_strats = len(is_scores_by_split[0])
    if n_strats < 2:
        raise ValueError("Require at least 2 candidate strategies to estimate PBO.")

    logits: List[float] = []
    relative_ranks: List[float] = []
    overfit_count = 0
    for s in range(n_splits):
        is_row = is_scores_by_split[s]
        oos_row = oos_scores_by_split[s]
        if len(is_row) != n_strats or len(oos_row) != n_strats:
            raise ValueError(f"Inconsistent strategy count at split {s}.")
        is_clean = [_require_finite_real(v, f"is_scores_by_split[{s}][{m}]") for m, v in enumerate(is_row)]
        oos_clean = [_require_finite_real(v, f"oos_scores_by_split[{s}][{m}]") for m, v in enumerate(oos_row)]
        best_m = max(range(n_strats), key=lambda m: is_clean[m])
        best_oos = oos_clean[best_m]
        less = sum(1 for v in oos_clean if v < best_oos)
        equal = sum(1 for v in oos_clean if v == best_oos)
        mid_rank_1based = less + 0.5 * (equal + 1.0)
        omega = mid_rank_1based / float(n_strats + 1)
        omega = max(1e-6, min(1.0 - 1e-6, omega))
        lam = math.log(omega / (1.0 - omega))
        relative_ranks.append(round(omega, 6))
        logits.append(round(lam, 6))
        if lam < 0.0:
            overfit_count += 1

    pbo = overfit_count / float(n_splits)
    return {
        "n_splits": n_splits,
        "n_strategies": n_strats,
        "pbo": round(pbo, 6),
        "overfit_splits": overfit_count,
        "mean_logit": round(sum(logits) / float(n_splits), 6),
        "relative_ranks": relative_ranks,
        "logits": logits,
    }
