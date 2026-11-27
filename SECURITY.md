# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `2.7.x` | Yes |
| `< 2.7.0` | Upgrade to `2.7.0` |

## Reporting a Vulnerability

Please report security vulnerabilities responsibly via GitHub Security Advisories at https://github.com/alpalbayrak91-boop/leakspec-cli/security/advisories. Do not open public issues containing exploit payloads against production CI pipelines.

## Security & Determinism Properties

- Pure Python 3.10+ standard library implementation with zero external runtime dependencies (eliminating third-party supply-chain risk).
- Offline deterministic CSV, LSH, and OASIS SARIF v2.1.0 auditing with zero outbound network calls and zero dynamic code execution.
- Cryptographic audit receipts sealed with RFC 8785 JSON Canonicalization Scheme (JCS) and RFC 6962 domain-separated SHA-256 Merkle roots.
