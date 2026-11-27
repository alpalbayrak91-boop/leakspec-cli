# Contributing to LeakSpec

Thank you for your interest in contributing! LeakSpec is an independent, OSI-licensed (MIT) open-source project.

## Developer Certificate of Origin (DCO 1.1)

No CLA is required. Please sign off every commit with `git commit -s` so each commit includes a `Signed-off-by: Your Name <email>` trailer confirming you have the right to submit the contribution under the MIT License.

## Development & Verification Workflow

1. Fork the repository and create a topic branch (`git checkout -b fix/my-change`).
2. Run the deterministic standard-library test suite locally before opening a pull request:
   ```bash
   PYTHONPATH=src python3 -W error::DeprecationWarning -m unittest discover -s tests -p "test_*.py" -v
   ```
3. If modifying `formal/LeakSpecCausalMask.lean`, ensure standalone Pure-Init Lean 4 compilation succeeds with `--trust=0 -DwarningAsError=true` and zero `Classical.choice` axioms, and sync `LEAN4_CAUSAL_MASK_THEOREM_SOURCE` in `src/leakspec/sarif_taxonomy.py`.
4. Never commit secrets, API keys, private credentials, or local machine paths.
5. Write `USD` explicitly for any cost or compute credit figures.

## Reporting Issues

Open a GitHub Issue at https://github.com/alpalbayrak91-boop/leakspec-cli/issues with minimal reproduction steps. For security vulnerabilities, use GitHub Security Advisories.
