"""LeakSpec (KK-CAUSALSPEC v2.0) - Kramers-Kronig Black-Box Lookahead Auditor."""

from .core import (
    audit_causality,
    chi2_survival,
    discrete_kk_hilbert_freq,
    fft_radix2,
    kk_spectral_defect_from_impulse,
    mv_kk_joint_audit,
)
from .scanner import (
    audit_csv_file,
    audit_feature_store,
    build_sarif_report,
    generate_synthetic_d10_csv,
    load_feature_store_csv,
)

__version__ = "2.0.0"
__all__ = [
    "__version__",
    "audit_causality",
    "audit_csv_file",
    "audit_feature_store",
    "build_sarif_report",
    "chi2_survival",
    "discrete_kk_hilbert_freq",
    "fft_radix2",
    "generate_synthetic_d10_csv",
    "kk_spectral_defect_from_impulse",
    "load_feature_store_csv",
    "mv_kk_joint_audit",
]
