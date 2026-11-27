# SPDX-License-Identifier: MIT
"""
Comprehensive unittest suite for LeakSpec 5-Layer Modules:
  - Layer 1: Lopez de Prado (2018) CPCV + Purging + Post-Horizon Embargo
  - Layer 2: Normalized Target-Leakage MI + Kramers-Kronig Spectral Causality
  - Layer 3: Pigeonhole SimHash64 + MinHash-LSH + Canonical n-gram Contamination
  - Layer 4: OASIS SARIF 2.1.0 Rule Taxonomy (LS-TIME-001..008)
  - Layer 5: Pure-Init Lean 4 Causal-Mask Certificate
"""

from __future__ import annotations

import random
import unittest

from leakspec.contamination_lsh import (
    audit_train_test_contamination,
    canonicalize_text,
    hamming_distance64,
    simhash64,
)
from leakspec.cpcv_embargo import (
    LabelSpan,
    audit_cpcv_splits,
    cpcv_num_backtest_paths,
    cpcv_num_splits,
    generate_cpcv_splits,
)
from leakspec.sarif_taxonomy import (
    SARIF_RULE_TAXONOMY,
    emit_lean4_causal_mask_theorem,
    run_five_layer_audit,
    verify_causal_mask_matrix,
)
from leakspec.target_mi_spectral import audit_target_mi_and_spectral_causality


def _gen_ar1(seed: int, n: int = 800, phi: float = 0.75) -> list[float]:
    rng = random.Random(seed)
    y = [0.0] * n
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.gauss(0.0, 1.0)
    return y


class TestV25FiveLayer(unittest.TestCase):
    def test_layer1_cpcv_combinatorics_and_audit(self) -> None:
        self.assertEqual(cpcv_num_splits(6, 2), 15)
        self.assertEqual(cpcv_num_backtest_paths(6, 2), 5)
        self.assertEqual(cpcv_num_splits(10, 2), 45)
        self.assertEqual(cpcv_num_backtest_paths(10, 2), 9)

        spans = [
            LabelSpan(obs_index=i, t_start=float(i), t_end=float(i + 3), t_avail=float(i))
            for i in range(120)
        ]
        gen = generate_cpcv_splits(spans, n_groups=6, n_test_groups=2, embargo_bars=3)
        self.assertEqual(gen["num_splits"], 15)
        self.assertEqual(gen["num_backtest_paths"], 5)

        raw_splits = [(s.train_indices, s.test_indices) for s in gen["splits"]]
        audit_clean = audit_cpcv_splits(spans, raw_splits, min_embargo_bars=3)
        self.assertEqual(audit_clean["verdict"], "CPCV_CLEAN")
        self.assertEqual(audit_clean["total_purge_violations"], 0)
        self.assertEqual(audit_clean["total_embargo_violations"], 0)

        naive_train = list(range(0, 40)) + list(range(60, 120))
        naive_test = list(range(40, 60))
        audit_bad = audit_cpcv_splits(spans, [(naive_train, naive_test)], min_embargo_bars=3)
        rule_ids = {f["rule_id"] for f in audit_bad["findings"]}
        self.assertIn("LS-TIME-002", rule_ids)
        self.assertIn("LS-TIME-003", rule_ids)

        bad_spans = list(spans)
        bad_spans[10] = LabelSpan(obs_index=10, t_start=10.0, t_end=13.0, t_avail=11.5)
        audit_pit = audit_cpcv_splits(bad_spans, [], min_embargo_bars=1)
        self.assertTrue(any(f["rule_id"] == "LS-TIME-001" for f in audit_pit["findings"]))

    def test_layer2_nmi_and_spectral_causality(self) -> None:
        y = _gen_ar1(seed=42, n=900, phi=0.80)
        rng = random.Random(43)

        x_causal = [
            (0.5 * y[t - 1] - 0.3 * y[t - 2] if t >= 2 else 0.0) + rng.gauss(0.0, 0.8)
            for t in range(len(y))
        ]
        res_clean = audit_target_mi_and_spectral_causality(
            x_causal, y, feature_name="f_causal", max_lag=6, n_permutations=60
        )
        self.assertEqual(res_clean["verdict"], "CAUSAL_CLEAN")

        x_leaked = [
            (0.65 * y[t + 2] if t + 2 < len(y) else 0.0) + rng.gauss(0.0, 0.5)
            for t in range(len(y))
        ]
        res_leaked = audit_target_mi_and_spectral_causality(
            x_leaked, y, feature_name="f_future_leak", max_lag=6, n_permutations=60
        )
        rids = {f["rule_id"] for f in res_leaked["findings"]}
        self.assertIn("LS-TIME-004", rids)
        self.assertIn("LS-TIME-005", rids)
        self.assertEqual(res_leaked["nmi"]["best_future_lag"], 2)
        self.assertEqual(res_leaked["spectral_causality"]["best_acausal_lag"], 2)

    def test_layer3_contamination_lsh_simhash_ngram(self) -> None:
        self.assertEqual(canonicalize_text("  Hello, World! 123.45 ", mask_numbers=True), "hello world num")
        h1 = simhash64(["alpha", "beta", "gamma", "delta"])
        self.assertEqual(hamming_distance64(h1, h1), 0)

        train = [
            ("tr_01", "Compute the combinatorial purged cross validation embargo window for financial time series."),
            ("tr_02", "Kramers Kronig dispersion relations link real and imaginary parts of causal susceptibility."),
            ("tr_03", "Zero dependency static analysis outputs OASIS SARIF 2.1.0 security alerts for CI pipelines."),
        ]
        test = [
            ("te_exact", "compute the COMBINATORIAL purged cross-validation embargo window for financial time series!"),
            ("te_neardup", "Kramers Kronig dispersion relations connect real and imaginary parts of causal susceptibility."),
            ("te_clean", "Bilinear attention routing optimizes sparse transformer memory bandwidth on edge accelerators."),
        ]
        res = audit_train_test_contamination(
            train, test, word_n=4, char_k=5, num_perm=64, lsh_bands=16, simhash_max_hamming=12
        )
        self.assertEqual(res["verdict"], "CONTAMINATION_DETECTED")
        self.assertEqual(res["exact_matches"], 1)
        self.assertEqual(res["contaminated_test_count"], 2)
        flagged_ids = {f["test_id"] for f in res["findings"]}
        self.assertIn("te_exact", flagged_ids)
        self.assertIn("te_neardup", flagged_ids)
        self.assertNotIn("te_clean", flagged_ids)

    def test_layer4_and_layer5_sarif_and_causal_mask(self) -> None:
        self.assertEqual(len(SARIF_RULE_TAXONOMY), 8)
        ids = [r["id"] for r in SARIF_RULE_TAXONOMY]
        self.assertEqual(ids, [f"LS-TIME-00{i}" for i in range(1, 9)])

        clean_mat = [
            [1.0, 0.0, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.2, 0.3, 0.5, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        ]
        m_ok = verify_causal_mask_matrix(clean_mat)
        self.assertEqual(m_ok["verdict"], "CAUSAL_MASK_VERIFIED")

        bad_mat = [
            [1.0, 0.25, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.2, 0.3, 0.5, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        ]
        m_bad = verify_causal_mask_matrix(bad_mat)
        self.assertEqual(m_bad["verdict"], "LEAK_DETECTED")
        self.assertEqual(m_bad["findings"][0]["rule_id"], "LS-TIME-008")

        spans = [LabelSpan(i, float(i), float(i + 2), float(i)) for i in range(60)]
        y = _gen_ar1(seed=99, n=400, phi=0.7)
        x_bad = [(0.7 * y[t + 1] if t + 1 < len(y) else 0.0) for t in range(len(y))]
        unified = run_five_layer_audit(
            spans=spans,
            splits=[(list(range(0, 20)) + list(range(30, 60)), list(range(20, 30)))],
            target=y,
            features={"f_leak1": x_bad},
            train_records=[("tr1", "exact duplicate benchmark prompt for leakspec validation")],
            test_records=[("te1", "exact duplicate benchmark prompt for leakspec validation")],
            causal_mask_matrix=bad_mat,
            min_embargo_bars=2,
        )
        self.assertEqual(unified["overall_verdict"], "LEAK_DETECTED")
        self.assertEqual(unified["exit_code"], 2)
        sarif = unified["sarif_2_1_0"]
        self.assertEqual(sarif["version"], "2.1.0")
        self.assertEqual(len(sarif["runs"][0]["tool"]["driver"]["rules"]), 8)
        found_rules = {r["ruleId"] for r in sarif["runs"][0]["results"]}
        self.assertTrue(
            {"LS-TIME-002", "LS-TIME-003", "LS-TIME-004", "LS-TIME-005", "LS-TIME-006", "LS-TIME-008"}.issubset(
                found_rules
            )
        )
        self.assertGreater(len(emit_lean4_causal_mask_theorem()), 500)


if __name__ == "__main__":
    unittest.main()
