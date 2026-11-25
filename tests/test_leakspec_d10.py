# SPDX-License-Identifier: MIT
"""Comprehensive unit tests for leakspec-kk (KK-CAUSALSPEC v2.0, d=10 feature store)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from leakspec.cli import main as cli_main
from leakspec.scanner import audit_csv_file, build_sarif_report, generate_synthetic_d10_csv


class TestLeakSpecD10(unittest.TestCase):
    def test_mixed_d10_csv_exact_5of5_recall_0of5_fp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "mixed_d10.csv")
            meta = generate_synthetic_d10_csv(csv_path, mode="mixed", n_rows=2000, seed=20261125)
            rep = audit_csv_file(csv_path, target_col="target_ret", time_col="ts", alpha=0.01)
            self.assertEqual(rep["store_verdict"], "LEAK_DETECTED")
            self.assertEqual(rep["d_total_features"], 10)
            self.assertEqual(sorted(rep["leaked_features"]), sorted(meta["expected_leaks"]))
            self.assertTrue(rep["csv_metadata"]["timestamp_monotonic"])
            self.assertEqual(len(str(rep["receipt_sha256"])), 64)

    def test_stealth_joint_d10_caught_only_by_mv_kk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "stealth_d10.csv")
            meta = generate_synthetic_d10_csv(csv_path, mode="stealth_joint", n_rows=2000, seed=20261125)
            rep = audit_csv_file(csv_path, target_col="target_ret", time_col="ts", alpha=0.01)
            self.assertEqual(rep["store_verdict"], "LEAK_DETECTED")
            self.assertEqual(rep["leaked_features"], [])
            self.assertTrue(rep["joint_only_stealth_leak"])
            self.assertTrue(rep["mv_kk_joint"]["triggered"])
            self.assertEqual(rep["mv_kk_joint"]["best_lag"], -1)
            self.assertEqual(sorted(rep["mv_kk_joint"]["top_suspects"]), sorted(meta["expected_leaks"]))

    def test_clean_d10_csv_passes_gate_exit_0(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "clean_d10.csv")
            json_out = os.path.join(tmpdir, "receipt.json")
            sarif_out = os.path.join(tmpdir, "report.sarif")
            generate_synthetic_d10_csv(csv_path, mode="clean", n_rows=2000, seed=20261127)
            rc = cli_main([
                "scan",
                "--csv", csv_path,
                "--target-col", "target_ret",
                "--time-col", "ts",
                "--alpha", "0.01",
                "--json-out", json_out,
                "--sarif-out", sarif_out,
            ])
            self.assertEqual(rc, 0)
            with open(json_out, "r", encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["store_verdict"], "CAUSAL_CLEAN")
            with open(sarif_out, "r", encoding="utf-8") as f:
                sarif = json.load(f)
            self.assertEqual(sarif["version"], "2.1.0")
            self.assertEqual(len(sarif["runs"][0]["results"]), 0)

    def test_cli_exit_2_and_sarif_on_leaked_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "mixed_d10.csv")
            sarif_out = os.path.join(tmpdir, "report.sarif")
            generate_synthetic_d10_csv(csv_path, mode="mixed", n_rows=2000, seed=20261125)
            rc = cli_main([
                "scan",
                "--csv", csv_path,
                "--target-col", "target_ret",
                "--time-col", "ts",
                "--alpha", "0.01",
                "--sarif-out", sarif_out,
            ])
            self.assertEqual(rc, 2)
            with open(sarif_out, "r", encoding="utf-8") as f:
                sarif = json.load(f)
            self.assertGreaterEqual(len(sarif["runs"][0]["results"]), 5)


if __name__ == "__main__":
    unittest.main()
