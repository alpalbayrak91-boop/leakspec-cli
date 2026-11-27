# SPDX-License-Identifier: MIT
"""
Forensic Regression, Cryptographic Provenance, and OPSEC Hygiene Suite for LeakSpec v2.7.0.

Verifies:
  - BUG-1: Post-horizon effective embargo (post_purge_anchor = max_test_label_end) when H >= h + symmetric PBO
  - BUG-2a: Tie-aware mid-rank quantile binning (H(constant) == 0.0, 0 false positives on constant/binary sparse)
  - BUG-2b: Phipson-Smyth auto-floor (eff_permutations >= ceil(2/alpha) + 19), Westfall-Young max-lag null, and p_gtest
  - BUG-3a: Pigeonhole Multi-Block SimHash64 exact O(1) candidate lookup at N > 6,000 without 5,000-row cutoff
  - BUG-3b: Decimal-before-punctuation numeric masking & full Unicode isalnum() canonicalization
  - BUG-4: Order-invariant SARIF v2.1.0 partialFingerprints (primaryLocationLineHash), helpUri, and fwer_strict=True
  - Layer 4/5: RFC 8785 JCS + RFC 6962 SHA-256 Merkle digests & exact parity with formal/LeakSpecCausalMask.lean
  - Repo-wide OPSEC & hygiene: 0 ASCII-36 bytes, 0 literal helper workarounds, 0 local paths, 0 internal org names
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import types
import unittest

import leakspec
from leakspec.contamination_lsh import (
    audit_against_contamination_index,
    audit_train_test_contamination,
    build_contamination_index,
    canonicalize_text,
)
from leakspec.core import audit_causality, mv_kk_joint_audit
from leakspec.cpcv_embargo import (
    LabelSpan,
    audit_cpcv_splits,
    compute_post_purge_anchor,
    cpcv_num_backtest_paths,
    cpcv_num_splits,
    estimate_cpcv_pbo,
    generate_cpcv_splits,
)
from leakspec.sarif_taxonomy import (
    EMPTY_RFC6962_MERKLE_ROOT,
    SARIF_RULE_TAXONOMY,
    build_sarif_210_report,
    canonical_jcs_bytes,
    canonical_jcs_dumps,
    emit_lean4_causal_mask_theorem,
    rfc6962_leaf_hash,
    rfc6962_merkle_root_hex,
    rfc6962_node_hash,
    run_five_layer_audit,
    verify_causal_mask_matrix,
)
from leakspec.scanner import audit_feature_store
from leakspec.target_mi_spectral import (
    audit_multifeature_mi_spectral,
    audit_target_mi_and_spectral_causality,
    normalized_mutual_information,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestV27ForensicRegression(unittest.TestCase):
    def test_bug1_post_horizon_effective_embargo_and_symmetric_pbo(self) -> None:
        # Horizon H = 5 >= embargo_bars h = 3 (the exact BUG-1 collapse regime)
        spans = [LabelSpan(i, float(i), float(i + 5), float(i)) for i in range(120)]
        anchor = compute_post_purge_anchor(spans, 19, 24.0, set(range(0, 20)))
        self.assertEqual(anchor, 24)

        gen = generate_cpcv_splits(spans, n_groups=6, n_test_groups=2, embargo_bars=3)
        non_empty = sum(1 for s in gen["splits"] if len(s.embargoed_indices) > 0)
        self.assertEqual(non_empty, 14)

        raw_splits = [(s.train_indices, s.test_indices) for s in gen["splits"]]
        aud_clean = audit_cpcv_splits(spans, raw_splits, min_embargo_bars=3)
        self.assertEqual(aud_clean["verdict"], "CPCV_CLEAN")
        self.assertEqual(aud_clean["total_purge_violations"], 0)
        self.assertEqual(aud_clean["total_embargo_violations"], 0)

        # Breach post-horizon embargo at index 26 (purged [20..24] removed, so 26 - 19 = 7 > 3,
        # yet 26 - 24 = 2 <= 3 -> MUST trigger LS-TIME-003!)
        leaky_train = list(range(26, 120))
        leaky_test = list(range(0, 20))
        aud_breach = audit_cpcv_splits(spans, [(leaky_train, leaky_test)], min_embargo_bars=3)
        self.assertEqual(aud_breach["verdict"], "LEAK_DETECTED")
        self.assertEqual(aud_breach["total_purge_violations"], 0)
        self.assertGreater(aud_breach["total_embargo_violations"], 0)
        self.assertTrue(any(f["rule_id"] == "LS-TIME-003" for f in aud_breach["findings"]))

        # Bailey et al. (2017) PBO: complete overfit -> 1.0, complete OOS winner -> 0.0, median -> logit 0.0
        pbo_overfit = estimate_cpcv_pbo(
            is_scores_by_split=[[2.1, 0.5, 0.4], [1.9, 0.3, 0.2], [0.4, 2.0, 0.3]],
            oos_scores_by_split=[[-0.5, 0.8, 0.6], [-0.2, 0.7, 0.5], [0.9, -0.4, 0.5]],
        )
        self.assertEqual(pbo_overfit["pbo"], 1.0)

        pbo_skill = estimate_cpcv_pbo(
            is_scores_by_split=[[2.0, 0.1], [0.2, 1.8]],
            oos_scores_by_split=[[1.5, -0.3], [-0.2, 1.4]],
        )
        self.assertEqual(pbo_skill["pbo"], 0.0)
        self.assertGreater(pbo_skill["mean_logit"], 0.5)

        pbo_median = estimate_cpcv_pbo(
            is_scores_by_split=[[3.0, 1.0, 0.0]],
            oos_scores_by_split=[[0.5, 1.0, 0.0]],
        )
        self.assertAlmostEqual(float(pbo_median["logits"][0]), 0.0, places=6)

    def test_bug2_tie_aware_binning_perm_floor_and_gtest(self) -> None:
        res_const = normalized_mutual_information([0.0] * 500, [0.0] * 500, n_bins=8)
        self.assertEqual(res_const["h_x"], 0.0)
        self.assertEqual(res_const["mi_bits"], 0.0)
        self.assertEqual(res_const["nmi_sym"], 0.0)
        self.assertEqual(res_const["p_gtest"], 1.0)

        rng = random.Random(20261127)
        y = [rng.gauss(0.0, 1.0) for _ in range(650)]
        # Constant and binary sparse clean features must never false-positive
        aud_const = audit_target_mi_and_spectral_causality(
            [0.0] * len(y), y, feature_name="f_const", max_lag=5, n_permutations=40, alpha=0.01
        )
        self.assertEqual(aud_const["verdict"], "CAUSAL_CLEAN")
        self.assertGreaterEqual(aud_const["nmi"]["effective_permutations"], 219)

        sparse_bin = [1.0 if (t >= 1 and y[t - 1] > 2.1) else 0.0 for t in range(len(y))]
        aud_sparse = audit_target_mi_and_spectral_causality(
            sparse_bin, y, feature_name="f_sparse", max_lag=5, n_permutations=40, alpha=0.01
        )
        self.assertEqual(aud_sparse["verdict"], "CAUSAL_CLEAN")

        # Small caller n_permutations=50 with alpha=0.01 auto-escalates above 219 and triggers LS-TIME-004 + p_gtest
        x_leak = [(0.8 * y[t + 2] if t + 2 < len(y) else 0.0) + rng.gauss(0.0, 0.3) for t in range(len(y))]
        aud_leak = audit_target_mi_and_spectral_causality(
            x_leak, y, feature_name="f_leak", max_lag=5, n_permutations=50, alpha=0.01
        )
        self.assertGreaterEqual(aud_leak["nmi"]["effective_permutations"], 219)
        self.assertTrue(aud_leak["nmi"]["triggered"])
        self.assertLess(aud_leak["nmi"]["p_gtest"], 1e-6)

    def test_bug3_decimal_unicode_and_pigeonhole_above_5000_rows(self) -> None:
        self.assertEqual(canonicalize_text("  Hello, World! 123.45 ", mask_numbers=True), "hello world num")
        self.assertEqual(canonicalize_text("Fee: 1,250,000.75 USD!", mask_numbers=True), "fee num usd")
        self.assertIn("çağdaş", canonicalize_text("Çağdaş şifreleme 99.5!", mask_numbers=True))

        # Build N = 5,200 (> 5,000!) corpus where target near-duplicate differs only slightly
        # and verify Pigeonhole Multi-Block SimHash64 catches it in O(1) bucket time.
        train = [
            (f"tr_{i:04d}", f"Orthogonal spline wavelet packet basis index {i} on dyadic grid {i + 3}.")
            for i in range(5200)
        ]
        train.append((
            "tr_target_5200",
            "Çağdaş spektral nedensellik denetimi çok değişkenli finansal zaman serisi modellerinde gelecek sızıntısını önler ve doğrular.",
        ))
        idx = build_contamination_index(
            train,
            word_n=5,
            char_k=5,
            num_perm=16,
            lsh_bands=4,
            simhash_max_hamming=10,
            mask_numbers=True,
        )
        self.assertEqual(len(idx.sh_slices), 11)
        test = [(
            "te_neardup",
            "Çağdaş spektral nedensellik denetimi çok değişkenli finansal zaman serisi modellerinde gelecek sızıntısını önler ve kanıtlar!",
        )]
        res = audit_against_contamination_index(
            idx, test, jaccard_threshold=0.98, ngram_containment_threshold=0.98
        )
        # With containment/jaccard set strict (0.98), detection relies strictly on Pigeonhole SimHash64!
        self.assertEqual(res["pigeonhole_blocks"], 11)
        self.assertEqual(res["verdict"], "CONTAMINATION_DETECTED")
        self.assertEqual(res["findings"][0]["rule_id"], "LS-TIME-007")
        self.assertEqual(res["findings"][0]["train_id"], "tr_target_5200")

        # Also check direct audit_train_test_contamination consistency
        res_small = audit_train_test_contamination(
            train[-20:], test, num_perm=16, lsh_bands=4, simhash_max_hamming=10,
            jaccard_threshold=0.98, ngram_containment_threshold=0.98, mask_numbers=True
        )
        self.assertEqual(res_small["findings"][0]["train_id"], "tr_target_5200")

    def test_bug4_order_invariant_sarif_helpuri_jcs_and_rfc6962(self) -> None:
        for rule in SARIF_RULE_TAXONOMY:
            self.assertIn("helpUri", rule)
            self.assertTrue(str(rule["helpUri"]).startswith("https://github.com/"))

        f_a = {"rule_id": "LS-TIME-002", "severity": "error", "split_id": 0, "detail": "Split 0 purge leak"}
        f_b = {"rule_id": "LS-TIME-008", "severity": "error", "operator": "attn", "detail": "Mask entry (0,1) nonzero"}
        s1 = build_sarif_210_report([f_a, f_b])
        s2 = build_sarif_210_report([f_b, f_a])

        fp1 = s1["runs"][0]["results"][1]["partialFingerprints"]["primaryLocationLineHash"]
        fp2 = s2["runs"][0]["results"][0]["partialFingerprints"]["primaryLocationLineHash"]
        self.assertEqual(fp1, fp2)

        # Sorted leaf RFC 6962 Merkle root is also order-invariant across finding permutations
        mr1 = s1["runs"][0]["properties"]["rfc6962_findings_merkle_root"]
        mr2 = s2["runs"][0]["properties"]["rfc6962_findings_merkle_root"]
        self.assertEqual(mr1, mr2)
        self.assertNotEqual(mr1, EMPTY_RFC6962_MERKLE_ROOT)

        # Verify RFC 8785 JCS -0.0 -> "0" and key ordering, plus RFC 6962 domain separation
        self.assertEqual(canonical_jcs_dumps({"b": -0.0, "a": 1}), '{"a":1,"b":0}')
        self.assertEqual(rfc6962_merkle_root_hex([]), EMPTY_RFC6962_MERKLE_ROOT)
        self.assertNotEqual(rfc6962_leaf_hash(b"x"), rfc6962_node_hash(b"x", b""))

        # Verify fwer_strict=True gates unadjusted single-feature noise in run_five_layer_audit
        spans = [LabelSpan(i, float(i), float(i + 2), float(i)) for i in range(60)]
        gen = generate_cpcv_splits(spans, n_groups=6, n_test_groups=2, embargo_bars=2)
        clean_splits = [(s.train_indices, s.test_indices) for s in gen["splits"]]
        rng = random.Random(12345)
        y = [rng.gauss(0.0, 1.0) for _ in range(300)]
        feats = {f"f_{k}": [rng.gauss(0.0, 1.0) for _ in range(300)] for k in range(8)}
        rep = run_five_layer_audit(
            spans=spans,
            splits=clean_splits,
            target=y,
            features=feats,
            train_records=[("tr1", "Alpha beta gamma delta epsilon zeta.")],
            test_records=[("te1", "Unrelated hardware routing latency verification.")],
            causal_mask_matrix=[[1.0, 0.0], [0.5, 0.5]],
            min_embargo_bars=2,
            fwer_strict=True,
        )
        self.assertEqual(rep["overall_verdict"], "ALL_5_LAYERS_CLEAN")
        self.assertEqual(rep["exit_code"], 0)

    def test_lean4_source_mirror_and_metadata_sync(self) -> None:
        self.assertEqual(leakspec.__version__, "2.7.0")
        lean_file = REPO_ROOT / "formal" / "LeakSpecCausalMask.lean"
        self.assertTrue(lean_file.is_file())
        lean_disk = lean_file.read_text(encoding="utf-8")
        self.assertEqual(lean_disk, emit_lean4_causal_mask_theorem())

        # Must contain all required constructive theorems and zero incomplete-proof markers
        self.assertNotIn("sor" + "ry", lean_disk.lower())
        self.assertIn("set_option warningAsError true", lean_disk)
        self.assertIn("set_option autoImplicit false", lean_disk)
        required_theorems = (
            "causal_mask_non_anticipation",
            "causal_kernel_compose_closed",
            "causal_kernel_add_closed",
            "causal_kernel_scale_closed",
            "purged_embargo_implies_no_overlap",
            "purged_embargo_monotone",
            "purged_embargo_post_horizon_sound",
            "cpcv_path_counting_identity",
            "acausal_energy_zero_of_causal_impulse",
            "acausal_energy_nonneg",
            "pigeonhole_sum_ge_length_of_pos",
            "simhash_pigeonhole_zero_block",
            "rfc6962_merkle_domain_and_tamper_sound",
        )
        for thm in required_theorems:
            self.assertIn(f"theorem {thm}", lean_disk)
            self.assertIn(f"#print axioms LeakSpec.{thm}", lean_disk)

        pub_theorems = re.findall(r"(?m)^theorem\s+([A-Za-z0-9_]+)\b", lean_disk)
        self.assertGreaterEqual(len(pub_theorems), 11)

        # Metadata files must be present and version-aligned to 2.7.0 with ORCID 0009-0008-9737-4477
        pyproj = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('version = "2.7.0"', pyproj)
        cff = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn('version: "2.7.0"', cff)
        self.assertIn("0009-0008-9737-4477", cff)
        zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertEqual(zenodo["version"], "2.7.0")
        self.assertEqual(zenodo["creators"][0]["orcid"], "0009-0008-9737-4477")
        funding = json.loads((REPO_ROOT / "funding.json").read_text(encoding="utf-8"))
        self.assertEqual(funding["projects"][0]["guid"], "leakspec-cli")

    def test_leakbench_200_summary_and_cryptographic_merkle_integrity(self) -> None:
        summary_path = REPO_ROOT / "bench" / "ml_leakbench_200" / "summary.json"
        report_path = REPO_ROOT / "bench" / "ml_leakbench_200" / "LEAKBENCH_200_REPORT.md"
        self.assertTrue(summary_path.is_file())
        self.assertTrue(report_path.is_file())

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["benchmark"], "ML-ContamAndLeakBench-200")
        self.assertEqual(summary["tool_version"], "2.7.0")
        self.assertEqual(summary["n_scenarios"], 200)
        self.assertEqual(summary["n_clean"], 100)
        self.assertEqual(summary["n_leaky"], 100)
        self.assertEqual(len(summary["scenarios"]), 200)

        # Verify RFC 6962 deterministic Merkle root over all 200 scenarios
        leaves = [
            leakspec.canonical_jcs_bytes({
                "scenario_id": s["scenario_id"],
                "family": s["family"],
                "ground_truth_leaky": s["ground_truth_leaky"],
                "predicted_leaky": s["predicted_leaky"],
                "detail": s["detail"],
            })
            for s in summary["scenarios"]
        ]
        self.assertEqual(
            rfc6962_merkle_root_hex(leaves),
            summary["rfc6962_scenarios_merkle_root"],
        )
        self.assertGreaterEqual(summary["metrics"]["f1"], 0.98)
        self.assertLessEqual(summary["metrics"]["fpr"], 0.02)

    def test_defect1_target_mi_spectral_and_core_nan_inf_bool_rejected(self) -> None:
        rng = random.Random(901)
        n = 120
        y = [rng.gauss(0.0, 1.0) for _ in range(n)]
        leaked = [(0.9 * y[t + 1] if t + 1 < n else 0.0) for t in range(n)]

        for bad_val in (float("nan"), float("inf"), float("-inf"), True, False):
            bad_x = list(leaked)
            bad_x[17] = bad_val  # type: ignore[call-overload]
            bad_y = list(y)
            bad_y[23] = bad_val  # type: ignore[call-overload]
            bad_short_y = list(y[:8])
            bad_short_y[3] = bad_val  # type: ignore[call-overload]

            with self.assertRaises(ValueError):
                normalized_mutual_information(bad_x, y)
            with self.assertRaises(ValueError):
                normalized_mutual_information(leaked[:8], bad_short_y)
            with self.assertRaises(ValueError):
                audit_target_mi_and_spectral_causality(bad_x, y)
            with self.assertRaises(ValueError):
                audit_target_mi_and_spectral_causality(leaked, bad_y)
            with self.assertRaises(ValueError):
                audit_multifeature_mi_spectral({"f_ok": leaked, "f_bad": bad_x}, y)
            with self.assertRaises(ValueError):
                audit_multifeature_mi_spectral({"f_ok": leaked}, bad_y)
            with self.assertRaises(ValueError):
                audit_causality(bad_x, y)
            with self.assertRaises(ValueError):
                audit_causality(y, bad_x)
            with self.assertRaises(ValueError):
                mv_kk_joint_audit(y, {"f1": leaked, "f2": bad_x})
            with self.assertRaises(ValueError):
                audit_feature_store(y, {"f1": leaked, "f2": bad_x})

        for bad_alpha in (float("nan"), float("inf"), 0.0, 1.0, -0.05, 1.2, True):
            with self.assertRaises(ValueError):
                audit_target_mi_and_spectral_causality(leaked, y, alpha=bad_alpha)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                audit_multifeature_mi_spectral({"f": leaked}, y, alpha=bad_alpha)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                audit_causality(y, leaked, alpha=bad_alpha)  # type: ignore[arg-type]

        for bad_thr in (float("nan"), float("inf"), -0.01, 1.5, True):
            with self.assertRaises(ValueError):
                audit_target_mi_and_spectral_causality(leaked, y, nmi_threshold=bad_thr)  # type: ignore[arg-type]
        for bad_z in (float("nan"), float("inf"), float("-inf"), -0.1, False):
            with self.assertRaises(ValueError):
                audit_target_mi_and_spectral_causality(leaked, y, spectral_z_threshold=bad_z)  # type: ignore[arg-type]

    def test_defect2_estimate_cpcv_pbo_nan_inf_bool_rejected(self) -> None:
        for bad_val in (float("nan"), float("inf"), float("-inf"), True, False):
            with self.assertRaises(ValueError):
                estimate_cpcv_pbo(
                    [[1.0, bad_val], [0.5, 0.2]],  # type: ignore[list-item]
                    [[0.3, 0.4], [0.1, 0.8]],
                )
            with self.assertRaises(ValueError):
                estimate_cpcv_pbo(
                    [[1.0, 0.2], [0.5, 0.2]],
                    [[0.3, 0.4], [bad_val, 0.8]],  # type: ignore[list-item]
                )

    def test_defect3_labelspan_and_embargo_params_nan_inf_bool_rejected(self) -> None:
        for bad_ts in (float("nan"), float("inf"), float("-inf"), True, False):
            with self.assertRaises(ValueError):
                LabelSpan(0, bad_ts, 5.0, 0.0)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                LabelSpan(0, 0.0, bad_ts, 0.0)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                LabelSpan(0, 0.0, 5.0, bad_ts)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LabelSpan(True, 0.0, 5.0, 0.0)  # type: ignore[arg-type]

        # Bypassed / duck-typed non-finite timestamp passed to audit_cpcv_splits -> LS-TIME-001 + JCS-safe
        spans_valid = [LabelSpan(i, float(i), float(i + 2), float(i)) for i in range(24)]
        bypassed = list(spans_valid)
        bypassed[3] = types.SimpleNamespace(obs_index=3, t_start=float("nan"), t_end=5.0, t_avail=3.0)  # type: ignore[call-overload]
        aud_nan = audit_cpcv_splits(bypassed, [(list(range(0, 12)), list(range(12, 24)))])
        self.assertEqual(aud_nan["verdict"], "LEAK_DETECTED")
        self.assertTrue(any(f["rule_id"] == "LS-TIME-001" for f in aud_nan["findings"]))
        self.assertGreater(len(canonical_jcs_bytes(aud_nan)), 10)

        for bad_pct in (float("nan"), float("inf"), float("-inf"), -0.01, 0.5, 0.9, True):
            with self.assertRaises(ValueError):
                generate_cpcv_splits(spans_valid, n_groups=4, n_test_groups=1, pct_embargo=bad_pct)  # type: ignore[arg-type]
        for bad_time in (float("nan"), float("inf"), float("-inf"), -0.1, True):
            with self.assertRaises(ValueError):
                generate_cpcv_splits(spans_valid, n_groups=4, n_test_groups=1, embargo_time=bad_time)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                audit_cpcv_splits(spans_valid, [], min_embargo_time=bad_time)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            generate_cpcv_splits(spans_valid, n_groups=4, n_test_groups=1, embargo_bars=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            audit_cpcv_splits(spans_valid, [], min_embargo_bars=True)  # type: ignore[arg-type]

    def test_defect4_generate_cpcv_splits_zero_and_invalid_groups_entry_validation(self) -> None:
        spans = [LabelSpan(i, float(i), float(i + 1), float(i)) for i in range(20)]
        for bad_n, bad_k in ((0, 0), (1, 0), (1, 1), (4, 0), (4, -1), (4, 4), (4, 5), (-2, 1), (True, 1), (4, True)):
            with self.assertRaises(ValueError):
                generate_cpcv_splits([], n_groups=bad_n, n_test_groups=bad_k)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                generate_cpcv_splits(spans, n_groups=bad_n, n_test_groups=bad_k)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                cpcv_num_splits(bad_n, bad_k)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                cpcv_num_backtest_paths(bad_n, bad_k)  # type: ignore[arg-type]

    def test_defect5_verify_causal_mask_matrix_nan_inf_bool_fail_closed(self) -> None:
        for bad_tol in (float("nan"), float("inf"), float("-inf"), -1e-9, True, False):
            with self.assertRaises(ValueError):
                verify_causal_mask_matrix([[1.0, 0.0], [0.5, 0.5]], tol=bad_tol)  # type: ignore[arg-type]

        bad_matrices = (
            [[1.0, float("nan")], [0.5, 0.5]],
            [[float("nan"), 0.0], [0.5, 0.5]],
            [[1.0, 0.0], [float("-inf"), 0.5]],
            [[1.0, float("inf")], [0.5, 0.5]],
            [[True, 0.0], [0.5, 0.5]],
        )
        for mat in bad_matrices:
            res = verify_causal_mask_matrix(mat)  # type: ignore[arg-type]
            self.assertEqual(res["verdict"], "LEAK_DETECTED")
            self.assertGreaterEqual(res["violation_count"], 1)
            self.assertGreaterEqual(res["nonfinite_entry_count"], 1)
            self.assertEqual(res["findings"][0]["rule_id"], "LS-TIME-008")
            sarif = build_sarif_210_report(res["findings"])
            self.assertEqual(len(sarif["runs"][0]["results"]), 1)
            self.assertGreater(len(canonical_jcs_bytes(sarif)), 20)

    def test_repo_wide_zero_ascii36_zero_local_paths_zero_internal_names(self) -> None:
        forbidden_chr_call = ("chr" + "(36)").encode("utf-8")
        forbidden_host_path = "/" + "home" + "/" + "alp"
        forbidden_internal = "alp" + "lab"

        ascii36_hits = []
        chr_call_hits = []
        host_path_hits = []
        internal_hits = []

        for path in sorted(REPO_ROOT.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if (
                rel.startswith(".git/")
                or "__pycache__/" in rel
                or ".pytest_cache/" in rel
                or ".egg-info/" in rel
            ):
                continue
            raw = path.read_bytes()
            if raw.count(36) > 0:
                ascii36_hits.append((rel, raw.count(36)))
            if forbidden_chr_call in raw:
                chr_call_hits.append(rel)
            txt = raw.decode("utf-8", errors="ignore")
            if forbidden_host_path in txt:
                host_path_hits.append(rel)
            if forbidden_internal in txt.lower():
                internal_hits.append(rel)

        self.assertEqual(ascii36_hits, [], f"ASCII-36 byte found: {ascii36_hits}")
        self.assertEqual(chr_call_hits, [], f"chr-36 workaround found: {chr_call_hits}")
        self.assertEqual(host_path_hits, [], f"Local host path leak found: {host_path_hits}")
        self.assertEqual(internal_hits, [], f"Internal org name leak found: {internal_hits}")


if __name__ == "__main__":
    unittest.main()
