# leakspec-kk (KK-CAUSALSPEC v2.0)

Zero-dependency Pure Python 3.10+ black-box lookahead-leakage auditor for time-series models, quant backtests, and multivariate ML feature stores.

## 4-Layer Spectral Causality Architecture

1. **Layer 1 (`IP-KK`)**: Dual innovation-prewhitened linear Kramers-Kronig Hilbert dispersion defect with Hall-Heyde martingale self-normalized Z-statistics (immune to AR(1)+GARCH+heavy-tail false positives and minimum-phase Bode traps).
2. **Layer 2 (`NL2-KK`)**: Channel-whitened Scandolo-Bassani nonlinear envelope scan (`u^2`, `|u|`, `v^2`, `|v|`, `r^2`) detecting zero-linear-correlation future-volatility and future-regime leakage under GARCH(1,1).
3. **Layer 3 (`SUMRULE-KK` & Global Pinning)**: Low-frequency f-sum-rule coherent accumulation (`sqrt(H)` SNR boost for diffuse 24-bar future-mean leakage) plus full-sample `StandardScaler`/`MinMaxScaler` pinning forensics.
4. **Layer 4 (`MV-KK`)**: Multivariate Cholesky-whitened Hotelling-Martingale joint dispersion test across all `d` features simultaneously, catching stealth difference/cancellation-mode leakage buried below univariate detection thresholds, with heteroskedasticity-robust sandwich Wald feature attribution.

## Multiplicity Control

- **Store-Level CI/CD Gate**: Holm-Bonferroni FWER across all `d` features plus Layer 4 `MV-KK` (`m = d + 1`).
- **Per-Feature Triage**: Benjamini-Hochberg FDR (`q_bh`) for ordered diagnostics.

## Quickstart

```bash
pip install .
leakspec scan --csv data/feature_store_d10.csv --target-col target_ret --time-col ts --alpha 0.01 --json-out reports/receipt.json --sarif-out reports/leakspec.sarif
```

Exit codes:
- `0`: `CAUSAL_CLEAN`
- `2`: `LEAK_DETECTED` (blocks PR in CI/CD)
- `1`: CLI or input CSV error

## Tier Reference (USD)

- Open-Source CLI (`leakspec-kk` MIT): 0 USD
- Quant Pro CI/CD Pack (multi-store badge + compliance PDF): 49 USD/month
- Fund & Enterprise Audit SLA: 199 USD/month
