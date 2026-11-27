# SPDX-License-Identifier: MIT
"""
LeakSpec CLI (v2.7.0) - 5-Layer Temporal Leakage, Post-Horizon CPCV Embargo & PBO,
Westfall-Young Target-MI / Kramers-Kronig Spectral Causality, Pigeonhole Multi-Block
SimHash64 & MinHash-LSH Contamination, OASIS SARIF v2.1.0 (LS-TIME-001..008) with
RFC 8785 JCS + RFC 6962 Merkle Digests, and Pure-Init Lean 4 Formal Auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from typing import Sequence

from .cpcv_embargo import (
    LabelSpan,
    audit_cpcv_splits,
    estimate_cpcv_pbo,
    generate_cpcv_splits,
)
from .sarif_taxonomy import (
    emit_lean4_causal_mask_theorem,
    run_five_layer_audit,
)
from .scanner import (
    audit_csv_file,
    build_sarif_report,
    generate_synthetic_d10_csv,
)

WORKFLOW_TEMPLATE = """name: LeakSpec Causal Feature-Store Gate

on:
  pull_request:
    paths:
      - "data/**/*.csv"
      - "features/**/*.py"
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  security-events: write

jobs:
  causal-lookahead-audit:
    name: LeakSpec v2.7.0 5-Layer Temporal & Contamination Audit
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install leakspec-kk
        run: python3 -m pip install --no-deps .
      - name: Run multivariate feature-store lookahead gate
        run: |
          mkdir -p reports/leakspec
          leakspec scan --csv data/feature_store_d10.csv --target-col target_ret --time-col ts --alpha 0.01 --json-out reports/leakspec/receipt.json --sarif-out reports/leakspec/leakspec.sarif
      - name: Upload SHA-256 audit receipt
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: leakspec-causal-audit-receipt
          path: reports/leakspec/
      - name: Upload SARIF annotations
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: reports/leakspec/leakspec.sarif
"""


def _cmd_scan(args: argparse.Namespace) -> int:
    fcols = [c.strip() for c in args.feature_cols.split(",")] if args.feature_cols else None
    report = audit_csv_file(
        csv_path=args.csv,
        target_col=args.target_col,
        feature_cols=fcols,
        time_col=args.time_col,
        sort_by_time=args.sort_by_time,
        alpha=args.alpha,
        ar_p=args.ar_p,
        causal_lags=args.causal_lags,
        acausal_lags=args.acausal_lags,
        sumrule_horizon=args.sumrule_horizon,
        enable_mv_kk=not args.disable_mv_kk,
    )
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if args.sarif_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.sarif_out)), exist_ok=True)
        sarif = build_sarif_report(report)
        with open(args.sarif_out, "w", encoding="utf-8") as f:
            json.dump(sarif, f, indent=2)
    print(json.dumps(report, indent=2))
    return int(report["exit_code_recommended"])


def _cmd_benchmark(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        mixed_csv = args.write_csv or os.path.join(tmpdir, "feature_store_d10_mixed.csv")
        stealth_csv = os.path.join(tmpdir, "feature_store_d10_stealth.csv")
        clean_csv = os.path.join(tmpdir, "feature_store_d10_clean.csv")

        meta_mixed = generate_synthetic_d10_csv(mixed_csv, mode="mixed", n_rows=args.n_rows, seed=args.seed)
        meta_stealth = generate_synthetic_d10_csv(stealth_csv, mode="stealth_joint", n_rows=args.n_rows, seed=args.seed + 1)
        meta_clean = generate_synthetic_d10_csv(clean_csv, mode="clean", n_rows=args.n_rows, seed=args.seed + 2)

        rep_mixed = audit_csv_file(mixed_csv, target_col="target_ret", time_col="ts", alpha=args.alpha)
        rep_stealth = audit_csv_file(stealth_csv, target_col="target_ret", time_col="ts", alpha=args.alpha)
        rep_clean = audit_csv_file(clean_csv, target_col="target_ret", time_col="ts", alpha=args.alpha)

        out = {
            "benchmark": "KK-CAUSALSPEC v2.7.0 d=10 Feature-Store CSV Suite",
            "mixed_d10": {
                "expected_leaks": meta_mixed["expected_leaks"],
                "detected_leaks": rep_mixed["leaked_features"],
                "exact_match": sorted(meta_mixed["expected_leaks"]) == sorted(rep_mixed["leaked_features"]),
                "store_verdict": rep_mixed["store_verdict"],
                "receipt_sha256": rep_mixed["receipt_sha256"],
            },
            "stealth_joint_d10": {
                "expected_pair": meta_stealth["expected_leaks"],
                "univariate_leaks": rep_stealth["leaked_features"],
                "joint_only_stealth_leak": rep_stealth["joint_only_stealth_leak"],
                "mv_kk_triggered": rep_stealth["mv_kk_joint"]["triggered"],
                "mv_kk_best_lag": rep_stealth["mv_kk_joint"]["best_lag"],
                "mv_kk_best_t2": rep_stealth["mv_kk_joint"]["best_t2"],
                "mv_kk_top_suspects": rep_stealth["mv_kk_joint"]["top_suspects"],
                "store_verdict": rep_stealth["store_verdict"],
            },
            "clean_d10": {
                "expected_leaks": meta_clean["expected_leaks"],
                "detected_leaks": rep_clean["leaked_features"],
                "mv_kk_triggered": rep_clean["mv_kk_joint"]["triggered"],
                "store_verdict": rep_clean["store_verdict"],
            },
        }
        print(json.dumps(out, indent=2))
        return 0


def _cmd_cpcv_bench(args: argparse.Namespace) -> int:
    spans = [
        LabelSpan(obs_index=i, t_start=float(i), t_end=float(i + args.horizon), t_avail=float(i))
        for i in range(args.n_samples)
    ]
    gen = generate_cpcv_splits(
        spans,
        n_groups=args.n_groups,
        n_test_groups=args.n_test_groups,
        embargo_bars=args.embargo_bars,
    )
    raw_splits = [(s.train_indices, s.test_indices) for s in gen["splits"]]
    audit_clean = audit_cpcv_splits(spans, raw_splits, min_embargo_bars=args.embargo_bars)
    rng = random.Random(20261127)
    s_cnt = int(gen["num_splits"])
    is_scores = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(s_cnt)]
    oos_scores = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(s_cnt)]
    pbo_info = estimate_cpcv_pbo(is_scores, oos_scores)
    out = {
        "benchmark": "Lopez de Prado (2018) Post-Horizon CPCV + PBO Verification",
        "n_samples": args.n_samples,
        "n_groups": args.n_groups,
        "n_test_groups": args.n_test_groups,
        "horizon": args.horizon,
        "num_splits": gen["num_splits"],
        "num_backtest_paths": gen["num_backtest_paths"],
        "path_coverage_verified": gen["path_coverage_verified"],
        "post_horizon_embargo": gen["post_horizon_embargo"],
        "embargo_bars": args.embargo_bars,
        "audit_verdict": audit_clean["verdict"],
        "total_purge_violations": audit_clean["total_purge_violations"],
        "total_embargo_violations": audit_clean["total_embargo_violations"],
        "null_pbo_estimate": pbo_info["pbo"],
    }
    print(json.dumps(out, indent=2))
    return 0 if audit_clean["verdict"] == "CPCV_CLEAN" else 2


def _cmd_audit_five_layer(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    n = args.n_rows
    y = [0.0] * n
    for t in range(1, n):
        y[t] = 0.75 * y[t - 1] + rng.gauss(0.0, 1.0)
    x_clean = [(0.5 * y[t - 1] if t >= 1 else 0.0) + rng.gauss(0.0, 0.8) for t in range(n)]
    spans = [LabelSpan(i, float(i), float(i + 5), float(i)) for i in range(120)]

    if args.mode == "clean":
        gen = generate_cpcv_splits(spans, n_groups=6, n_test_groups=2, embargo_bars=3)
        splits = [(s.train_indices, s.test_indices) for s in gen["splits"]]
        features = {"f_causal": x_clean, "f_const_zero": [0.0] * n}
        train_recs = [("tr1", "Combinatorial purged cross validation prevents backtest overfitting in quant research.")]
        test_recs = [("te1", "Sparse attention routing lowers GPU memory bandwidth in transformer inference.")]
        mask = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]
    else:
        splits = [(list(range(0, 40)) + list(range(60, 120)), list(range(40, 60)))]
        x_leaked = [(0.7 * y[t + 2] if t + 2 < n else 0.0) + rng.gauss(0.0, 0.4) for t in range(n)]
        features = {"f_causal": x_clean, "f_leaked_t2": x_leaked}
        train_recs = [("tr1", "Combinatorial purged cross validation prevents backtest overfitting in quant research.")]
        test_recs = [("te1", "combinatorial purged cross-validation prevents backtest overfitting in quant research!")]
        mask = [[1.0, 0.15, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]

    report = run_five_layer_audit(
        spans=spans,
        splits=splits,
        target=y,
        features=features,
        train_records=train_recs,
        test_records=test_recs,
        causal_mask_matrix=mask,
        artifact_uri=args.artifact_uri,
        min_embargo_bars=3,
    )
    if args.sarif_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.sarif_out)), exist_ok=True)
        with open(args.sarif_out, "w", encoding="utf-8") as f:
            json.dump(report["sarif_2_1_0"], f, indent=2)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    summary = {
        "schema_version": report["schema_version"],
        "mode": args.mode,
        "overall_verdict": report["overall_verdict"],
        "exit_code": report["exit_code"],
        "sarif_rules_count": len(report["sarif_2_1_0"]["runs"][0]["tool"]["driver"]["rules"]),
        "sarif_results_count": len(report["sarif_2_1_0"]["runs"][0]["results"]),
        "rfc6962_layers_merkle_root": report["rfc6962_layers_merkle_root"],
        "receipt_sha256": report["sarif_2_1_0"]["runs"][0]["properties"]["receipt_sha256"],
    }
    print(json.dumps(summary, indent=2))
    return int(report["exit_code"])


def _cmd_emit_lean(args: argparse.Namespace) -> int:
    src = emit_lean4_causal_mask_theorem()
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(src)
        print(json.dumps({"written": args.out, "bytes": len(src), "sha256": sha}, indent=2))
    else:
        print(src)
    return 0


def _cmd_init_ci(args: argparse.Namespace) -> int:
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(WORKFLOW_TEMPLATE)
    print(json.dumps({"written": out_path, "bytes": len(WORKFLOW_TEMPLATE)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leakspec",
        description="LeakSpec CLI v2.7.0 - 5-Layer Temporal Leakage, CPCV, Contamination & Formal Auditor",
    )
    parser.add_argument("--version", action="version", version="leakspec 2.7.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Audit a time-ordered multivariate feature-store CSV")
    p_scan.add_argument("--csv", required=True, help="Input feature-store CSV file path")
    p_scan.add_argument("--target-col", required=True, help="Exogenous driver or target column name")
    p_scan.add_argument("--feature-cols", default=None, help="Optional comma-separated feature column list")
    p_scan.add_argument("--time-col", default=None, help="Optional timestamp column for monotonicity verification")
    p_scan.add_argument("--sort-by-time", action="store_true", help="Sort rows by time-col before auditing")
    p_scan.add_argument("--alpha", type=float, default=0.01, help="Store-wide Holm-FWER significance threshold")
    p_scan.add_argument("--ar-p", type=int, default=6, help="Causal AR whitening order")
    p_scan.add_argument("--causal-lags", type=int, default=10, help="Causal OLS projection horizon")
    p_scan.add_argument("--acausal-lags", type=int, default=16, help="Anti-causal future lag horizon")
    p_scan.add_argument("--sumrule-horizon", type=int, default=24, help="f-sum-rule accumulation horizon")
    p_scan.add_argument("--disable-mv-kk", action="store_true", help="Disable Layer 4 multivariate MV-KK check")
    p_scan.add_argument("--json-out", default=None, help="Path to write canonical SHA-256 JSON receipt")
    p_scan.add_argument("--sarif-out", default=None, help="Path to write SARIF 2.1.0 annotations")
    p_scan.set_defaults(func=_cmd_scan)

    p_bench = sub.add_parser("benchmark", help="Run synthetic d=10 CSV verification suite")
    p_bench.add_argument("--n-rows", type=int, default=2000, help="Rows per synthetic CSV")
    p_bench.add_argument("--alpha", type=float, default=0.01, help="Store-wide Holm-FWER threshold")
    p_bench.add_argument("--seed", type=int, default=20261125, help="Deterministic RNG seed")
    p_bench.add_argument("--write-csv", default=None, help="Optional path to keep generated mixed d=10 CSV")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_cpcv = sub.add_parser("cpcv-bench", help="Run Lopez de Prado post-horizon CPCV + PBO benchmark")
    p_cpcv.add_argument("--n-samples", type=int, default=240, help="Number of time bars")
    p_cpcv.add_argument("--n-groups", type=int, default=6, help="Number of chronological groups N")
    p_cpcv.add_argument("--n-test-groups", type=int, default=2, help="Number of test groups k")
    p_cpcv.add_argument("--horizon", type=int, default=5, help="Forward label span horizon H")
    p_cpcv.add_argument("--embargo-bars", type=int, default=3, help="Post-horizon embargo buffer in bars")
    p_cpcv.set_defaults(func=_cmd_cpcv_bench)

    for cmd_name in ("audit-v25", "audit-v27"):
        p_aud = sub.add_parser(cmd_name, help="Run unified 5-layer v2.7.0 audit (clean or adversarial mode)")
        p_aud.add_argument("--mode", choices=["clean", "adversarial"], default="adversarial")
        p_aud.add_argument("--n-rows", type=int, default=600)
        p_aud.add_argument("--seed", type=int, default=20261126)
        p_aud.add_argument("--artifact-uri", default="pipeline_audit.json")
        p_aud.add_argument("--json-out", default=None)
        p_aud.add_argument("--sarif-out", default=None)
        p_aud.set_defaults(func=_cmd_audit_five_layer)

    p_lean = sub.add_parser("emit-lean", help="Emit verified Pure-Init Lean 4 causal-mask & CPCV proofs")
    p_lean.add_argument("--out", default=None, help="Optional output .lean file path")
    p_lean.set_defaults(func=_cmd_emit_lean)

    p_ci = sub.add_parser("init-ci", help="Write GitHub Actions workflow YAML template")
    p_ci.add_argument("--out", default=".github/workflows/leakspec-feature-gate.yml", help="Output YAML path")
    p_ci.set_defaults(func=_cmd_init_ci)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
