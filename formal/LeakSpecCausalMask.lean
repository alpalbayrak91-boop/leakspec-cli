-- SPDX-License-Identifier: MIT
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
