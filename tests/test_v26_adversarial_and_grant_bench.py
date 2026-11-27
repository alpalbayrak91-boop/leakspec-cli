# SPDX-License-Identifier: MIT
"""
Adversarial Audit & Grant-Readiness Verification Suite for LeakSpec v2.7.0.
"""

from __future__ import annotations

import math
import random
import unittest

from leakspec.cli import main as cli_main
from leakspec.contamination_lsh import audit_train_test_contamination
from leakspec.cpcv_embargo import (
    LabelSpan,
    generate_cpcv_splits,
    reconstruct_cpcv_paths,
)
from leakspec.sarif_taxonomy import (
    OASIS_SARIF_210_SCHEMA_URI,
    SARIF_RULE_TAXONOMY,
    build_sarif_210_report,
    emit_lean4_causal_mask_theorem,
)
from leakspec.target_mi_spectral import (
    _fft_radix2,
    audit_multifeature_mi_spectral,
    audit_target_mi_and_spectral_causality,
    discrete_kramers_kronig_hilbert_defect,
)


class TestV26AdversarialAndGrant(unittest.TestCase):
    def test_cpcv_path_reconstruction_6_2_and_10_2(self) -> None:
        for n_groups, k in ((6, 2), (10, 2)):
            spans = [LabelSpan(i, float(i), float(i + 2), float(i)) for i in range(n_groups * 20)]
            gen = generate_cpcv_splits(spans, n_groups=n_groups, n_test_groups=k, embargo_bars=2)
            self.assertTrue(gen["path_coverage_verified"])
            paths = gen["backtest_paths"]
            self.assertEqual(len(paths), math.comb(n_groups - 1, k - 1))
            for path in paths:
                self.assertEqual(set(path.keys()), set(range(n_groups)))
            rec = reconstruct_cpcv_paths(n_groups, k, gen["splits"])
            self.assertTrue(rec["coverage_verified"])

    def test_true_hilbert_kk_defect_zero_on_causal_and_positive_on_acausal(self) -> None:
        n_fft = 32
        h_causal = [0.0j] * n_fft
        h_causal[0] = 1.0 + 0.0j
        h_causal[1] = 0.5 + 0.0j
        h_causal[2] = 0.25 + 0.0j
        H_c = _fft_radix2(h_causal, invert=False)
        defect_c = discrete_kramers_kronig_hilbert_defect(H_c)
        self.assertLess(defect_c, 1e-12)

        h_acausal = [0.0j] * n_fft
        h_acausal[0] = 0.2 + 0.0j
        h_acausal[n_fft - 2] = 0.8 + 0.0j
        H_a = _fft_radix2(h_acausal, invert=False)
        defect_a = discrete_kramers_kronig_hilbert_defect(H_a)
        self.assertGreater(defect_a, 0.5)

    def test_nonlinear_volatility_leak_caught_by_envelope_nmi(self) -> None:
        rng = random.Random(20261126)
        n = 900
        y = [rng.gauss(0.0, 1.0) for _ in range(n)]
        x_vol = [(abs(y[t + 2]) if t + 2 < n else 0.0) + rng.gauss(0.0, 0.25) for t in range(n)]
        res = audit_target_mi_and_spectral_causality(
            x_vol, y, feature_name="f_vol_leak", max_lag=5, n_permutations=80
        )
        self.assertTrue(res["nmi"]["triggered"])
        self.assertEqual(res["nmi"]["best_future_lag"], 2)
        self.assertTrue(any(f["rule_id"] == "LS-TIME-004" for f in res["findings"]))

    def test_multifeature_holm_fwer_clean_d10(self) -> None:
        rng = random.Random(777)
        n = 500
        y = [0.0] * n
        for t in range(1, n):
            y[t] = 0.65 * y[t - 1] + rng.gauss(0.0, 1.0)
        feats = {
            f"f_clean_{j}": [(0.3 * y[t - 1] if t >= 1 else 0.0) + rng.gauss(0.0, 1.0) for t in range(n)]
            for j in range(10)
        }
        res = audit_multifeature_mi_spectral(feats, y, alpha=0.01, max_lag=5, n_permutations=60)
        self.assertEqual(res["verdict"], "CAUSAL_CLEAN")
        self.assertEqual(res["fwer_leaked_features"], [])

    def test_number_masked_template_clone_contamination(self) -> None:
        train = [("tr_math", "Solve the quadratic equation 3 x^2 + 12 x + 9 = 0 for real roots.")]
        test = [("te_clone", "Solve the quadratic equation 7 x^2 + 28 x + 21 = 0 for real roots.")]
        res = audit_train_test_contamination(train, test, mask_numbers=True)
        self.assertEqual(res["verdict"], "CONTAMINATION_DETECTED")
        self.assertEqual(res["exact_matches"], 1)

    def test_sarif_schema_fingerprints_and_cli_exit_codes(self) -> None:
        sarif = build_sarif_210_report(
            [{"rule_id": "LS-TIME-008", "severity": "error", "detail": "Upper-triangular entry nonzero"}],
            artifact_uri="model_mask.json",
        )
        schema_key = "\x24schema"
        self.assertEqual(sarif[schema_key], OASIS_SARIF_210_SCHEMA_URI)
        self.assertEqual(sarif["version"], "2.1.0")
        res0 = sarif["runs"][0]["results"][0]
        self.assertEqual(res0["ruleIndex"], 7)
        self.assertIn("primaryLocationLineHash", res0["partialFingerprints"])
        self.assertEqual(len(SARIF_RULE_TAXONOMY), 8)
        self.assertIn("causal_kernel_compose_closed", emit_lean4_causal_mask_theorem())

        self.assertEqual(cli_main(["cpcv-bench", "--n-samples", "120", "--n-groups", "6", "--n-test-groups", "2"]), 0)
        self.assertEqual(cli_main(["audit-v25", "--mode", "clean", "--n-rows", "300"]), 0)
        self.assertEqual(cli_main(["audit-v25", "--mode", "adversarial", "--n-rows", "300"]), 2)
        self.assertEqual(cli_main(["audit-v27", "--mode", "clean", "--n-rows", "300"]), 0)
        self.assertEqual(cli_main(["audit-v27", "--mode", "adversarial", "--n-rows", "300"]), 2)


if __name__ == "__main__":
    unittest.main()
