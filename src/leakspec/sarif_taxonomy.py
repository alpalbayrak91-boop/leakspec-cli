# SPDX-License-Identifier: MIT
"""
Layers 4 & 5 (v2.7.0): OASIS SARIF v2.1.0 Rule Taxonomy (LS-TIME-001..008),
Order-Invariant GitHub Code Scanning Fingerprints (primaryLocationLineHash),
RFC 8785 JCS Canonical Serialization + RFC 6962 Domain-Separated SHA-256 Merkle
Audit Digests, 14-Theorem Pure-Init Lean 4 Causal-Mask Certificate Emitter, and
FWER-Gated 5-Layer Unified Audit Pipeline.

Zero external dependencies (Pure Python 3.10+ standard library).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .contamination_lsh import audit_train_test_contamination
from .cpcv_embargo import LabelSpan, audit_cpcv_splits
from .target_mi_spectral import audit_multifeature_mi_spectral

OASIS_SARIF_210_SCHEMA_URI = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
)
_SARIF_SCHEMA_KEY = "\x24schema"
_REPO_URI = "https://github.com/alpalbayrak91-boop/leakspec-cli"
EMPTY_RFC6962_MERKLE_ROOT = hashlib.sha256(b"").hexdigest()


# ============================================================================
# RFC 8785 JSON Canonicalization Scheme (JCS) + RFC 6962 SHA-256 Merkle Digest
# ============================================================================


def _format_jcs_float(val: float) -> str:
    """Serialize an IEEE-754 float strictly per RFC 8785 Section 3.2.2.3."""
    if not math.isfinite(val):
        raise ValueError("RFC 8785 JCS forbids NaN and Infinity")
    if val == 0.0:
        return "0"

    sign = "-" if val < 0.0 else ""
    av = abs(val)
    raw = repr(av)
    if "e" in raw or "E" in raw:
        mant, exp_str = raw.lower().split("e", 1)
        exp10 = int(exp_str)
    else:
        mant = raw
        exp10 = 0

    if "." in mant:
        int_part, frac_part = mant.split(".", 1)
        digits = int_part + frac_part
        exp10 -= len(frac_part)
    else:
        digits = mant

    digits = digits.lstrip("0") or "0"
    stripped = digits.rstrip("0")
    exp10 += len(digits) - len(stripped)
    digits = stripped or "0"

    k = len(digits)
    n = exp10 + k

    if k <= n <= 21:
        return sign + digits + ("0" * (n - k))
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + ("0" * (-n)) + digits

    exp_val = n - 1
    exp_sign = "+" if exp_val >= 0 else "-"
    if k == 1:
        return sign + digits + "e" + exp_sign + str(abs(exp_val))
    return sign + digits[0] + "." + digits[1:] + "e" + exp_sign + str(abs(exp_val))


def canonical_jcs_dumps(obj: Any) -> str:
    """Serialize JSON values strictly per RFC 8785 (UTF-16-BE key sort, ECMAScript numbers)."""
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _format_jcs_float(obj)
    if isinstance(obj, str):
        obj.encode("utf-8")
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical_jcs_dumps(item) for item in obj) + "]"
    if isinstance(obj, Mapping):
        entries: List[Tuple[bytes, str, Any]] = []
        for key, val in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"RFC 8785 JCS requires string object keys, got {type(key).__name__}")
            entries.append((key.encode("utf-16-be"), key, val))
        entries.sort(key=lambda item: item[0])
        return (
            "{"
            + ",".join(
                f"{canonical_jcs_dumps(k)}:{canonical_jcs_dumps(v)}"
                for _, k, v in entries
            )
            + "}"
        )
    raise TypeError(f"Unsupported type for RFC 8785 JCS serialization: {type(obj).__name__}")


def canonical_jcs_bytes(obj: Any) -> bytes:
    """UTF-8 encoded RFC 8785 canonical JSON bytes."""
    return canonical_jcs_dumps(obj).encode("utf-8")


def rfc6962_leaf_hash(payload: bytes) -> bytes:
    """RFC 6962 Section 2.1 leaf hash: SHA-256(0x00 || payload)."""
    return hashlib.sha256(b"\x00" + payload).digest()


def rfc6962_node_hash(left: bytes, right: bytes) -> bytes:
    """RFC 6962 Section 2.1 interior node hash: SHA-256(0x01 || left || right)."""
    return hashlib.sha256(b"\x01" + left + right).digest()


def _largest_pow2_lt(n: int) -> int:
    k = 1
    while (k << 1) < n:
        k <<= 1
    return k


def _rfc6962_mth(leaves: Sequence[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return rfc6962_leaf_hash(leaves[0])
    k = _largest_pow2_lt(n)
    return rfc6962_node_hash(_rfc6962_mth(leaves[:k]), _rfc6962_mth(leaves[k:]))


def rfc6962_merkle_root_hex(leaves: Sequence[bytes]) -> str:
    """Computes RFC 6962 Section 2.1 SHA-256 Merkle tree root hex digest over byte payloads."""
    return _rfc6962_mth(leaves).hex()


# ============================================================================
# OASIS SARIF v2.1.0 Rule Taxonomy (LS-TIME-001..008)
# ============================================================================

SARIF_RULE_TAXONOMY: Tuple[Dict[str, object], ...] = (
    {
        "id": "LS-TIME-001",
        "name": "PointInTimeMonotonicityViolation",
        "helpUri": f"{_REPO_URI}#ls-time-001",
        "shortDescription": {
            "text": "Timestamp ordering inversion or point-in-time availability leak (t_avail > t_query)."
        },
        "fullDescription": {
            "text": "Detects non-monotonic timestamps, inverted label spans (t_end < t_start), or features whose availability timestamp exceeds the decision query timestamp."
        },
        "defaultConfiguration": {"level": "error"},
        "properties": {
            "tags": ["leakspec", "temporal", "point-in-time", "layer-1"],
            "precision": "very-high",
            "security-severity": "8.5",
        },
    },
    {
        "id": "LS-TIME-002",
        "name": "CPCVPurgingViolation",
        "helpUri": f"{_REPO_URI}#ls-time-002",
        "shortDescription": {
            "text": "Lopez de Prado (2018) CPCV purging failure: overlapping train and test label spans."
        },
        "fullDescription": {
            "text": "Training observations whose evaluation horizon [t_start, t_end] overlaps any test fold interval leak future outcome information across the split boundary."
        },
        "defaultConfiguration": {"level": "error"},
        "properties": {
            "tags": ["leakspec", "cpcv", "purging", "quant", "layer-1"],
            "precision": "very-high",
            "security-severity": "8.8",
        },
    },
    {
        "id": "LS-TIME-003",
        "name": "CPCVEmbargoViolation",
        "helpUri": f"{_REPO_URI}#ls-time-003",
        "shortDescription": {
            "text": "Lopez de Prado (2018) post-horizon CPCV embargo failure inside buffer window."
        },
        "fullDescription": {
            "text": "Training observations immediately following a test block's purged label horizon (post_purge_anchor) without an effective embargo buffer suffer from serial-correlation leakage."
        },
        "defaultConfiguration": {"level": "warning"},
        "properties": {
            "tags": ["leakspec", "cpcv", "embargo", "quant", "layer-1"],
            "precision": "high",
            "security-severity": "6.5",
        },
    },
    {
        "id": "LS-TIME-004",
        "name": "TargetLeakageNormalizedMI",
        "helpUri": f"{_REPO_URI}#ls-time-004",
        "shortDescription": {
            "text": "Normalized target-leakage Mutual Information (Westfall-Young NMI) future-lag anomaly."
        },
        "fullDescription": {
            "text": "Innovation-prewhitened feature exhibits statistically significant nonlinear dependence with future target realizations beyond the max-lag permutation null."
        },
        "defaultConfiguration": {"level": "error"},
        "properties": {
            "tags": ["leakspec", "mutual-information", "nonlinear-leakage", "layer-2"],
            "precision": "high",
            "security-severity": "8.2",
        },
    },
    {
        "id": "LS-TIME-005",
        "name": "SpectralCausalityKramersKronigViolation",
        "helpUri": f"{_REPO_URI}#ls-time-005",
        "shortDescription": {
            "text": "Anti-causal spectral energy / Kramers-Kronig Hilbert dispersion causality violation."
        },
        "fullDescription": {
            "text": "Prewhitened cross-spectrum reveals non-zero anti-causal impulse response energy at future lags, violating causal analyticity (Toll 1956)."
        },
        "defaultConfiguration": {"level": "error"},
        "properties": {
            "tags": ["leakspec", "kramers-kronig", "spectral-causality", "layer-2"],
            "precision": "high",
            "security-severity": "8.6",
        },
    },
    {
        "id": "LS-TIME-006",
        "name": "BenchmarkExactOrNgramContamination",
        "helpUri": f"{_REPO_URI}#ls-time-006",
        "shortDescription": {
            "text": "Exact canonical hash or canonical n-gram train-test benchmark contamination."
        },
        "fullDescription": {
            "text": "Evaluation or benchmark records share identical canonical hashes or high canonical word n-gram containment with training data."
        },
        "defaultConfiguration": {"level": "error"},
        "properties": {
            "tags": ["leakspec", "contamination", "ngram", "benchmark", "layer-3"],
            "precision": "very-high",
            "security-severity": "9.0",
        },
    },
    {
        "id": "LS-TIME-007",
        "name": "BenchmarkNearDuplicateLSHContamination",
        "helpUri": f"{_REPO_URI}#ls-time-007",
        "shortDescription": {
            "text": "Near-duplicate train-test contamination via Pigeonhole SimHash64 or MinHash-LSH."
        },
        "fullDescription": {
            "text": "Near-duplicate or paraphrased evaluation samples collide in MinHash-LSH bands (Broder 1997) or fall within Pigeonhole Multi-Block SimHash64 Hamming threshold (Charikar 2002)."
        },
        "defaultConfiguration": {"level": "warning"},
        "properties": {
            "tags": ["leakspec", "contamination", "simhash", "minhash-lsh", "layer-3"],
            "precision": "high",
            "security-severity": "7.2",
        },
    },
    {
        "id": "LS-TIME-008",
        "name": "FormalCausalMaskInvariantViolation",
        "helpUri": f"{_REPO_URI}#ls-time-008",
        "shortDescription": {
            "text": "Formal causal-mask upper-triangular invariant failure (Lean 4 certificate check)."
        },
        "fullDescription": {
            "text": "A model or operator weight/mask matrix contains non-zero entries for future indices (s > t), violating the machine-checked Pure-Init Lean 4 causal-mask subalgebra invariant."
        },
        "defaultConfiguration": {"level": "error"},
        "properties": {
            "tags": ["leakspec", "formal-verification", "lean4", "causal-mask", "layer-5"],
            "precision": "very-high",
            "security-severity": "9.5",
        },
    },
)

_RULE_INDEX_MAP: Dict[str, int] = {
    str(rule["id"]): idx for idx, rule in enumerate(SARIF_RULE_TAXONOMY)
}

# Exact byte-for-byte mirror of formal/LeakSpecCausalMask.lean (Lean 4.36.0 --trust=0, 0 warnings, 0 Classical.choice)
LEAN4_CAUSAL_MASK_THEOREM_SOURCE = r"""-- SPDX-License-Identifier: MIT
-- LeakSpec v2.7.0 Layer 5: Constructive Pure-Init Lean 4 Proofs
-- Covers:
--   1. Strictly Causal Linear Operator Subalgebra (Non-anticipation, Composition, Addition, Scaling)
--   2. Lopez de Prado (2018) Purged + Post-Horizon Embargo Soundness & Monotonicity
--   3. Combinatorial Purged Cross-Validation (CPCV) Path-Counting Identity
--   4. Kramers-Kronig Anti-Causal Impulse Spectral Energy Vanishing & Non-Negativity
--   5. Pigeonhole Multi-Block SimHash64 Sublinear Exact-Recall Soundness
--   6. RFC 6962 Domain-Separated Merkle Audit Digest Collision & Tamper Soundness
-- Pure Lean 4 Init only, 0 external dependencies, 0 incomplete proofs, 0 Classical.choice.

set_option warningAsError true
set_option autoImplicit false

namespace LeakSpec

-- ============================================================================
-- 1. Strictly Causal Operator Subalgebra (LS-TIME-008)
-- ============================================================================

/-- A discrete linear operator kernel K(t, s) is strictly causal iff K(t, s) = 0 for all s > t. -/
def IsCausalKernel (K : Nat -> Nat -> Int) : Prop :=
  forall t s : Nat, t < s -> K t s = 0

/-- Finite prefix convolution sum_{s=0}^{n} K(t, s) * x(s). -/
def applyKernel (K : Nat -> Nat -> Int) (x : Nat -> Int) (t : Nat) : Nat -> Int
  | 0 => K t 0 * x 0
  | n + 1 => applyKernel K x t n + K t (n + 1) * x (n + 1)

/-- Theorem 1 (LS-TIME-008): Non-anticipation / Causal Mask Invariance.
    If K is strictly causal and two input streams agree up to time t,
    their finite-horizon operator outputs at time t are identical for every horizon N. -/
theorem causal_mask_non_anticipation
    (K : Nat -> Nat -> Int) (x x' : Nat -> Int) (t : Nat)
    (hc : IsCausalKernel K)
    (hagree : forall s : Nat, s <= t -> x s = x' s) :
    forall N : Nat, applyKernel K x t N = applyKernel K x' t N := by
  intro N
  induction N with
  | zero =>
    have h0 : 0 <= t := Nat.zero_le t
    simp [applyKernel, hagree 0 h0]
  | succ n ih =>
    simp only [applyKernel]
    rw [ih]
    refine Decidable.byCases (p := n + 1 <= t) (fun hle => ?_) (fun hnot => ?_)
    · rw [hagree (n + 1) hle]
    · have hgt : t < n + 1 := Nat.lt_of_not_ge hnot
      rw [hc t (n + 1) hgt]
      simp

/-- Finite-horizon composition of two discrete linear kernels: sum_{u=0}^{N} K2(t, u) * K1(u, s). -/
def composeKernel (K2 K1 : Nat -> Nat -> Int) : Nat -> Nat -> Nat -> Int
  | 0, t, s => K2 t 0 * K1 0 s
  | n + 1, t, s => composeKernel K2 K1 n t s + K2 t (n + 1) * K1 (n + 1) s

/-- Theorem 2 (LS-TIME-008 Composition Closure):
    Composing any two strictly causal kernels preserves strict causality at all horizons. -/
theorem causal_kernel_compose_closed
    (K2 K1 : Nat -> Nat -> Int)
    (h2 : IsCausalKernel K2)
    (h1 : IsCausalKernel K1) :
    forall N : Nat, IsCausalKernel (composeKernel K2 K1 N) := by
  intro N t s hts
  induction N with
  | zero =>
    simp only [composeKernel]
    have h0s : 0 < s := Nat.lt_of_le_of_lt (Nat.zero_le t) hts
    rw [h1 0 s h0s]
    simp
  | succ n ih =>
    simp only [composeKernel]
    rw [ih]
    refine Decidable.byCases (p := n + 1 <= t) (fun hle => ?_) (fun hnot => ?_)
    · have hns : n + 1 < s := Nat.lt_of_le_of_lt hle hts
      rw [h1 (n + 1) s hns]
      simp
    · have htn : t < n + 1 := Nat.lt_of_not_ge hnot
      rw [h2 t (n + 1) htn]
      simp

/-- Theorem 3 (LS-TIME-008 Additive Closure / Residual Skip Connections):
    Pointwise sum of two strictly causal kernels is strictly causal. -/
theorem causal_kernel_add_closed
    (K1 K2 : Nat -> Nat -> Int)
    (h1 : IsCausalKernel K1)
    (h2 : IsCausalKernel K2) :
    IsCausalKernel (fun t s => K1 t s + K2 t s) := by
  intro t s hts
  dsimp only
  rw [h1 t s hts, h2 t s hts]
  simp

/-- Theorem 4 (LS-TIME-008 Scalar Closure / Attention Head Scaling):
    Scaling a strictly causal kernel by any integer factor preserves strict causality. -/
theorem causal_kernel_scale_closed
    (c : Int) (K : Nat -> Nat -> Int)
    (hc : IsCausalKernel K) :
    IsCausalKernel (fun t s => c * K t s) := by
  intro t s hts
  dsimp only
  rw [hc t s hts]
  simp

-- ============================================================================
-- 2. Purged & Post-Horizon Embargo Soundness + CPCV Combinatorics (LS-TIME-002/003)
-- ============================================================================

/-- Closed discrete observation and label evaluation span [start, stop]. -/
structure Span where
  start : Nat
  stop : Nat

/-- Two closed spans overlap iff start_a <= stop_b and start_b <= stop_a. -/
def Span.overlaps (a b : Span) : Prop :=
  a.start <= b.stop /\ b.start <= a.stop

/-- Lopez de Prado (2018) Purged + Embargoed separation predicate with embargo h. -/
def PurgedEmbargoed (tr te : Span) (h : Nat) : Prop :=
  tr.stop < te.start \/ te.stop + h < tr.start

/-- Theorem 5 (LS-TIME-002 / LS-TIME-003): Purged-Embargo implies strict non-overlap. -/
theorem purged_embargo_implies_no_overlap
    (tr te : Span) (h : Nat) (hpe : PurgedEmbargoed tr te h) :
    Not (Span.overlaps tr te) := by
  intro hov
  rcases hov with ⟨h1, h2⟩
  rcases hpe with hbefore | hafter
  · exact Nat.not_le_of_gt hbefore h2
  · have hlt : te.stop < tr.start := Nat.lt_of_le_of_lt (Nat.le_add_right te.stop h) hafter
    exact Nat.not_le_of_gt hlt h1

/-- Theorem 6 (LS-TIME-003 Embargo Monotonicity):
    Enforcing a larger post-horizon embargo h2 >= h1 implies every smaller embargo h1 holds. -/
theorem purged_embargo_monotone
    (tr te : Span) (h1 h2 : Nat)
    (hle : h1 <= h2)
    (hpe2 : PurgedEmbargoed tr te h2) :
    PurgedEmbargoed tr te h1 := by
  rcases hpe2 with hbefore | hafter
  · exact Or.inl hbefore
  · right
    have hadd : te.stop + h1 <= te.stop + h2 := Nat.add_le_add_left hle te.stop
    exact Nat.lt_of_le_of_lt hadd hafter

/-- Theorem 7 (LS-TIME-002 / LS-TIME-003 Post-Horizon Envelope Soundness):
    If a test block has maximum label horizon t_max >= te.stop (post_purge_anchor)
    and a post-test training span satisfies t_max + h < tr.start, then PurgedEmbargoed tr te h
    and strict interval non-overlap both hold simultaneously even when H >= h. -/
theorem purged_embargo_post_horizon_sound
    (tr te : Span) (t_max h : Nat)
    (henv : te.stop <= t_max)
    (hpost : t_max + h < tr.start) :
    PurgedEmbargoed tr te h /\ Not (Span.overlaps tr te) := by
  have hpe : PurgedEmbargoed tr te h := by
    right
    have hle : te.stop + h <= t_max + h := Nat.add_le_add_right henv h
    exact Nat.lt_of_le_of_lt hle hpost
  exact ⟨hpe, purged_embargo_implies_no_overlap tr te h hpe⟩

/-- Exact binomial coefficient C(n, k) in pure Lean 4. -/
def binom : Nat -> Nat -> Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

private theorem binom_one : forall n : Nat, binom n 1 = n := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [binom]
    rw [ih]
    omega

/-- Theorem 8 (LS-TIME-002 Lopez de Prado Eq. 12.1 CPCV Path-Counting Identity):
    (n + 1) * C(n, k) = (k + 1) * C(n + 1, k + 1), certifying N * phi(N, k) = k * C(N, k). -/
theorem cpcv_path_counting_identity :
    forall n k : Nat, (n + 1) * binom n k = (k + 1) * binom (n + 1) (k + 1) := by
  intro n
  induction n with
  | zero =>
    intro k
    cases k with
    | zero => rfl
    | succ k' => simp [binom]
  | succ n ih =>
    intro k
    cases k with
    | zero =>
      simp only [binom, Nat.zero_add, binom_one]
      omega
    | succ k' =>
      have hrec : binom (n + 2) (k' + 2) = binom (n + 1) (k' + 1) + binom (n + 1) (k' + 2) := rfl
      have hrec_n : binom (n + 1) (k' + 1) = binom n k' + binom n (k' + 1) := rfl
      have ih1 := ih k'
      have ih2 := ih (k' + 1)
      rw [hrec, Nat.mul_add, ← ih2]
      have hsplit : (k' + 2) * binom (n + 1) (k' + 1)
                  = (k' + 1) * binom (n + 1) (k' + 1) + binom (n + 1) (k' + 1) :=
        Nat.succ_mul (k' + 1) (binom (n + 1) (k' + 1))
      rw [hsplit, ← ih1, hrec_n]
      have hlhs : (n + 2) * (binom n k' + binom n (k' + 1))
                = (n + 1) * binom n k' + binom n k' + ((n + 1) * binom n (k' + 1) + binom n (k' + 1)) := by
        rw [Nat.mul_add, Nat.succ_mul (n + 1) (binom n k'), Nat.succ_mul (n + 1) (binom n (k' + 1))]
      rw [hlhs]
      omega

-- ============================================================================
-- 3. Kramers-Kronig Anti-Causal Impulse Spectral Energy (LS-TIME-005)
-- ============================================================================

/-- Anti-causal impulse energy over future lags 1..L: sum_{k=1}^{L} f(k)^2. -/
def acausalEnergy (f : Nat -> Int) : Nat -> Int
  | 0 => 0
  | k + 1 => acausalEnergy f k + f (k + 1) * f (k + 1)

/-- Theorem 9 (LS-TIME-005): Zero anti-causal spectral energy for causal impulse response. -/
theorem acausal_energy_zero_of_causal_impulse
    (f : Nat -> Int) (hc : forall k : Nat, 1 <= k -> f k = 0) :
    forall L : Nat, acausalEnergy f L = 0 := by
  intro L
  induction L with
  | zero => rfl
  | succ k ih =>
    simp only [acausalEnergy]
    rw [ih, hc (k + 1) (Nat.succ_le_succ (Nat.zero_le k))]
    simp

private theorem int_sq_nonneg (a : Int) : 0 <= a * a := by
  cases a with
  | ofNat n =>
    exact Int.natCast_nonneg (n * n)
  | negSucc n =>
    have h : Int.negSucc n * Int.negSucc n = Int.ofNat ((n + 1) * (n + 1)) := rfl
    rw [h]
    exact Int.natCast_nonneg ((n + 1) * (n + 1))

/-- Theorem 10 (LS-TIME-005 Non-negativity): Anti-causal spectral energy is always >= 0. -/
theorem acausal_energy_nonneg (f : Nat -> Int) :
    forall L : Nat, 0 <= acausalEnergy f L := by
  intro L
  induction L with
  | zero =>
    exact Int.le_refl 0
  | succ k ih =>
    simp only [acausalEnergy]
    exact Int.add_nonneg ih (int_sq_nonneg (f (k + 1)))

-- ============================================================================
-- 4. Pigeonhole Multi-Block SimHash64 Sublinear Exact Recall (LS-TIME-006/007)
-- ============================================================================

/-- Sum of per-block Hamming distances across disjoint SimHash bit-slices. -/
def blockDiffSum : List Nat -> Nat
  | [] => 0
  | d :: ds => d + blockDiffSum ds

/-- Theorem 11 (LS-TIME-007): XOR self-annihilation for identical canonical fingerprints. -/
theorem xor_self_zero (a : Nat) : Nat.xor a a = 0 := by
  have h : a ^^^ a = 0 := Nat.xor_self a
  exact h

/-- Theorem 12 (Pigeonhole Lower Bound): If every disjoint SimHash block has >= 1 differing bit,
    the total Hamming distance across all blocks is at least the number of blocks. -/
theorem pigeonhole_sum_ge_length_of_pos :
    forall ds : List Nat, (forall d : Nat, d ∈ ds -> 1 <= d) -> ds.length <= blockDiffSum ds := by
  intro ds
  induction ds with
  | nil =>
    intro _
    simp [blockDiffSum]
  | cons d tail ih =>
    intro hpos
    have hd : 1 <= d := hpos d (List.Mem.head tail)
    have htail : tail.length <= blockDiffSum tail :=
      ih (fun x hx => hpos x (List.Mem.tail d hx))
    simp only [List.length_cons, blockDiffSum]
    omega

/-- Constructive zero-block witness helper: if blockDiffSum ds < ds.length, some block is 0. -/
private theorem exists_zero_of_sum_lt_length :
    forall ds : List Nat, blockDiffSum ds < ds.length -> Exists (fun d : Nat => d ∈ ds /\ d = 0) := by
  intro ds
  induction ds with
  | nil =>
    intro hlt
    simp [blockDiffSum] at hlt
  | cons d tail ih =>
    intro hlt
    cases d with
    | zero =>
      exact ⟨0, List.Mem.head tail, rfl⟩
    | succ m =>
      simp only [blockDiffSum, List.length_cons] at hlt
      have htail_lt : blockDiffSum tail < tail.length := by omega
      rcases ih htail_lt with ⟨w, hw_mem, hw_zero⟩
      exact ⟨w, List.Mem.tail (m + 1) hw_mem, hw_zero⟩

/-- Theorem 13 (LS-TIME-007 Pigeonhole Multi-Block SimHash64 Exact Recall):
    Partitioning 64-bit SimHash into (k + 1) disjoint blocks guarantees that any pair with
    total Hamming distance <= k matches identically (0 bit differences) in at least one block. -/
theorem simhash_pigeonhole_zero_block
    (ds : List Nat) (k : Nat)
    (hlen : ds.length = k + 1)
    (hham : blockDiffSum ds <= k) :
    Exists (fun d : Nat => d ∈ ds /\ d = 0) := by
  have hlt : blockDiffSum ds < ds.length := by omega
  exact exists_zero_of_sum_lt_length ds hlt

-- ============================================================================
-- 5. RFC 6962 Domain-Separated Merkle Audit Digest Soundness (Layer 4)
-- ============================================================================

/-- Abstract collision-resistant tagged hash modeling SHA-256(tag || left || right). -/
structure TaggedHash where
  digest : Nat -> Nat -> Nat -> Nat

/-- Collision-resistance injectivity assumption for tagged SHA-256. -/
def TaggedHash.Injective (H : TaggedHash) : Prop :=
  forall t1 a1 b1 t2 a2 b2 : Nat,
    H.digest t1 a1 b1 = H.digest t2 a2 b2 -> t1 = t2 /\ a1 = a2 /\ b1 = b2

/-- RFC 6962 Section 2.1 leaf hash (0x00 domain tag) and interior node hash (0x01 domain tag). -/
def merkleLeaf (H : TaggedHash) (x : Nat) : Nat := H.digest 0 x 0
def merkleNode (H : TaggedHash) (l r : Nat) : Nat := H.digest 1 l r

/-- Theorem 14 (RFC 6962 Domain Separation & Tamper Detection):
    Leaf hashes never collide with interior node hashes, and modifying either finding payload
    changes the two-finding Merkle root. -/
theorem rfc6962_merkle_domain_and_tamper_sound
    (H : TaggedHash) (hinj : TaggedHash.Injective H)
    (x l r a1 a2 b1 b2 : Nat) :
    merkleLeaf H x ≠ merkleNode H l r /\
    (merkleNode H (merkleLeaf H a1) (merkleLeaf H b1) =
      merkleNode H (merkleLeaf H a2) (merkleLeaf H b2) -> a1 = a2 /\ b1 = b2) := by
  constructor
  · intro heq
    have htags := hinj 0 x 0 1 l r heq
    rcases htags with ⟨ht, _, _⟩
    omega
  · intro hroot
    have hnodes := hinj 1 (merkleLeaf H a1) (merkleLeaf H b1) 1 (merkleLeaf H a2) (merkleLeaf H b2) hroot
    rcases hnodes with ⟨_, hl, hr⟩
    have hla := hinj 0 a1 0 0 a2 0 hl
    have hra := hinj 0 b1 0 0 b2 0 hr
    rcases hla with ⟨_, ha, _⟩
    rcases hra with ⟨_, hb, _⟩
    exact ⟨ha, hb⟩

end LeakSpec

#print axioms LeakSpec.causal_mask_non_anticipation
#print axioms LeakSpec.causal_kernel_compose_closed
#print axioms LeakSpec.causal_kernel_add_closed
#print axioms LeakSpec.causal_kernel_scale_closed
#print axioms LeakSpec.purged_embargo_implies_no_overlap
#print axioms LeakSpec.purged_embargo_monotone
#print axioms LeakSpec.purged_embargo_post_horizon_sound
#print axioms LeakSpec.cpcv_path_counting_identity
#print axioms LeakSpec.acausal_energy_zero_of_causal_impulse
#print axioms LeakSpec.acausal_energy_nonneg
#print axioms LeakSpec.xor_self_zero
#print axioms LeakSpec.pigeonhole_sum_ge_length_of_pos
#print axioms LeakSpec.simhash_pigeonhole_zero_block
#print axioms LeakSpec.rfc6962_merkle_domain_and_tamper_sound
"""


def emit_lean4_causal_mask_theorem() -> str:
    """Returns the verified 14-theorem zero-dependency Pure-Init Lean 4 formal source."""
    return LEAN4_CAUSAL_MASK_THEOREM_SOURCE


def verify_causal_mask_matrix(
    mask_matrix: Sequence[Sequence[float]],
    tol: float = 1e-12,
    operator_name: str = "causal_operator",
) -> Dict[str, object]:
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(float(tol)) or float(tol) < 0.0:
        raise ValueError(f"tol must be a finite non-negative real number, got {tol!r}")
    tol_f = float(tol)

    n_rows = len(mask_matrix)
    violations: List[Tuple[int, int, object]] = []
    nonfinite_count = 0
    max_future_weight = 0.0

    for t, row in enumerate(mask_matrix):
        for s, raw_entry in enumerate(row):
            if isinstance(raw_entry, bool) or not isinstance(raw_entry, (int, float)) or not math.isfinite(float(raw_entry)):
                nonfinite_count += 1
                violations.append((t, s, repr(raw_entry)))
                continue
            if s > t:
                fval = float(raw_entry)
                val = abs(fval)
                if val > max_future_weight:
                    max_future_weight = val
                if val > tol_f:
                    violations.append((t, s, fval))

    findings: List[Dict[str, object]] = []
    if violations:
        sample = violations[:5]
        findings.append({
            "rule_id": "LS-TIME-008",
            "severity": "error",
            "operator": operator_name,
            "violation_count": len(violations),
            "nonfinite_entry_count": nonfinite_count,
            "max_future_weight": round(max_future_weight, 8),
            "witness_entries": sample,
            "detail": (
                f"Formal causal-mask violation in '{operator_name}': {len(violations)} "
                f"invalid or non-zero future entry/entries (nonfinite_or_bool={nonfinite_count}, "
                f"max_future_abs={max_future_weight:.6g}, sample={sample})."
            ),
        })

    lean_sha256 = hashlib.sha256(LEAN4_CAUSAL_MASK_THEOREM_SOURCE.encode("utf-8")).hexdigest()
    return {
        "layer": "L5_FORMAL_CAUSAL_MASK",
        "operator": operator_name,
        "matrix_rows": n_rows,
        "violation_count": len(violations),
        "nonfinite_entry_count": nonfinite_count,
        "max_future_weight": round(max_future_weight, 8),
        "lean4_theorem_count": 14,
        "lean4_theorem_sha256": lean_sha256,
        "verdict": "LEAK_DETECTED" if findings else "CAUSAL_MASK_VERIFIED",
        "findings": findings,
    }


def _stable_finding_key(f: Mapping[str, object]) -> str:
    """Order-invariant identity key for GitHub Code Scanning partialFingerprints."""
    parts: List[str] = []
    for field in ("feature", "test_id", "train_id", "operator", "split_id", "obs_index"):
        if field in f:
            parts.append(f"{field}={f[field]}")
    return ";".join(parts) if parts else "global"


def build_sarif_210_report(
    findings: Sequence[Mapping[str, object]],
    artifact_uri: str = "feature_store.csv",
    tool_version: str = "2.7.0",
    extra_properties: Optional[Mapping[str, object]] = None,
    include_schema_key: bool = True,
) -> Dict[str, object]:
    """
    Builds an OASIS SARIF v2.1.0 compliant document with LS-TIME-001..008 rules,
    helpUri, ruleIndex, order-invariant GitHub Code Scanning partialFingerprints,
    and RFC 8785 JCS + RFC 6962 SHA-256 Merkle audit digests.
    """
    sarif_results: List[Dict[str, object]] = []
    finding_leaf_hashes: List[bytes] = []

    for f in findings:
        rule_id = str(f.get("rule_id", "LS-TIME-001"))
        rule_index = _RULE_INDEX_MAP.get(rule_id, 0)
        level = str(f.get("severity", "error"))
        if level not in ("error", "warning", "note"):
            level = "error"
        msg = str(f.get("detail", f"LeakSpec violation {rule_id}"))
        props = {k: v for k, v in f.items() if k not in ("rule_id", "severity", "detail")}

        # Order-invariant primaryLocationLineHash (independent of list index; closes BUG-4)
        fp_material = f"{rule_id}|{artifact_uri}|{_stable_finding_key(f)}|{msg}".encode("utf-8")
        line_hash = hashlib.sha256(fp_material).hexdigest()[:32]
        result_entry: Dict[str, object] = {
            "ruleId": rule_id,
            "ruleIndex": rule_index,
            "level": level,
            "message": {"text": msg},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": artifact_uri},
                        "region": {"startLine": 1},
                    }
                }
            ],
            "partialFingerprints": {
                "primaryLocationLineHash": line_hash,
            },
            "properties": props,
        }
        sarif_results.append(result_entry)
        finding_leaf_hashes.append(canonical_jcs_bytes(result_entry))

    # Sort leaves canonically so RFC 6962 Merkle root is also order-invariant across finding permutations
    ordered_leaves = sorted(finding_leaf_hashes)
    findings_merkle_root = rfc6962_merkle_root_hex(ordered_leaves)

    run_props: Dict[str, object] = {
        "taxonomy": "LS-TIME-001..008",
        "total_findings": len(sarif_results),
        "lean4_theorem_count": 14,
        "lean4_theorem_sha256": hashlib.sha256(
            LEAN4_CAUSAL_MASK_THEOREM_SOURCE.encode("utf-8")
        ).hexdigest(),
        "rfc6962_findings_merkle_root": findings_merkle_root,
        "canonicalization_scheme": "RFC-8785-JCS+RFC-6962-SHA256",
    }
    if extra_properties:
        run_props.update(extra_properties)

    sarif_doc: Dict[str, object] = {}
    if include_schema_key:
        sarif_doc[_SARIF_SCHEMA_KEY] = OASIS_SARIF_210_SCHEMA_URI
    sarif_doc["version"] = "2.1.0"
    sarif_doc["runs"] = [
        {
            "tool": {
                "driver": {
                    "name": "LeakSpec-CLI 5-Layer Temporal & Contamination Auditor",
                    "version": tool_version,
                    "informationUri": _REPO_URI,
                    "rules": list(SARIF_RULE_TAXONOMY),
                }
            },
            "results": sarif_results,
            "properties": run_props,
        }
    ]
    jcs_digest = hashlib.sha256(canonical_jcs_bytes(sarif_doc)).hexdigest()
    sarif_doc["runs"][0]["properties"]["jcs_sha256"] = jcs_digest
    sarif_doc["runs"][0]["properties"]["receipt_sha256"] = jcs_digest
    return sarif_doc


def run_five_layer_audit(
    spans: Sequence[LabelSpan],
    splits: Sequence[Tuple[Sequence[int], Sequence[int]]],
    target: Sequence[float],
    features: Mapping[str, Sequence[float]],
    train_records: Sequence[Tuple[str, str]],
    test_records: Sequence[Tuple[str, str]],
    causal_mask_matrix: Sequence[Sequence[float]],
    artifact_uri: str = "pipeline_audit.json",
    min_embargo_bars: int = 2,
    alpha: float = 0.01,
    fwer_strict: bool = True,
    mask_numbers: bool = False,
) -> Dict[str, object]:
    """
    Executes all 5 LeakSpec v2.7.0 layers with strict Holm-Bonferroni FWER gating
    and seals the report with RFC 8785 JCS + RFC 6962 SHA-256 Merkle audit digests.
    """
    l1 = audit_cpcv_splits(spans=spans, splits=splits, min_embargo_bars=min_embargo_bars)
    l2_multi = audit_multifeature_mi_spectral(features=features, target=target, alpha=alpha)
    l2_results: Dict[str, Dict[str, object]] = l2_multi["per_feature"]
    fwer_set = set(l2_multi["fwer_leaked_features"])

    l2_findings: List[Dict[str, object]] = []
    for fname, res in l2_results.items():
        if (not fwer_strict) or (fname in fwer_set):
            l2_findings.extend(res["findings"])

    l3 = audit_train_test_contamination(
        train_records=train_records,
        test_records=test_records,
        mask_numbers=mask_numbers,
    )
    l5 = verify_causal_mask_matrix(causal_mask_matrix)

    all_findings: List[Dict[str, object]] = []
    all_findings.extend(l1["findings"])
    all_findings.extend(l2_findings)
    all_findings.extend(l3["findings"])
    all_findings.extend(l5["findings"])

    verdict = "LEAK_DETECTED" if all_findings else "ALL_5_LAYERS_CLEAN"
    layer_merkle_root = rfc6962_merkle_root_hex([
        canonical_jcs_bytes(l1),
        canonical_jcs_bytes(l2_multi),
        canonical_jcs_bytes(l3),
        canonical_jcs_bytes(l5),
    ])
    sarif = build_sarif_210_report(
        findings=all_findings,
        artifact_uri=artifact_uri,
        extra_properties={
            "overall_verdict": verdict,
            "fwer_strict": fwer_strict,
            "fwer_leaked_features": l2_multi["fwer_leaked_features"],
            "rfc6962_layers_merkle_root": layer_merkle_root,
        },
    )
    payload: Dict[str, object] = {
        "schema_version": "leakspec-v2.7-5layer",
        "overall_verdict": verdict,
        "exit_code": 2 if all_findings else 0,
        "layer_1_cpcv": l1,
        "layer_2_mi_spectral": l2_results,
        "layer_2_multiplicity": {
            "holm_adjusted_p": l2_multi["holm_adjusted_p"],
            "bh_fdr_q": l2_multi["bh_fdr_q"],
            "fwer_leaked_features": l2_multi["fwer_leaked_features"],
        },
        "layer_3_contamination": l3,
        "layer_5_causal_mask": l5,
        "rfc6962_layers_merkle_root": layer_merkle_root,
        "sarif_2_1_0": sarif,
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_jcs_bytes(payload)).hexdigest()
    return payload
