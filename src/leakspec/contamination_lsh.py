# SPDX-License-Identifier: MIT
"""
Layer 3 (v2.7.0): Train-Test Benchmark Contamination Engine
Exact Sublinear Pigeonhole Multi-Block SimHash64 Indexing (Charikar 2002) +
MinHash-LSH Banding (Broder 1997) + Decimal-Before-Punctuation & Full Unicode
isalnum() Canonical n-gram Fingerprinting.

Zero external dependencies (Pure Python 3.10+ standard library).
"""

from __future__ import annotations

import hashlib
import re
import struct
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

_MERSENNE_P = (1 << 61) - 1
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
_NON_ALNUM_UNICODE_RE = re.compile(r"[^\w\s]|_")

# Precomputed 256-entry 128-bit SWAR lookup table expanding 8 bits into 8 x 16-bit lanes
_SWAR_BYTE_16: Tuple[int, ...] = tuple(
    sum(((byte_val >> bit) & 1) << (16 * bit) for bit in range(8))
    for byte_val in range(256)
)


def canonicalize_text(text: str, mask_numbers: bool = False) -> str:
    """
    Unicode-aware canonical NFKC + casefold + punctuation-stripped normalizer.
    When mask_numbers=True, replaces integer, thousands-grouped, and decimal
    numeric literals BEFORE punctuation removal so '123.45' maps cleanly to a
    single 'num' token. Preserves full Unicode isalnum() characters across
    multilingual, mathematical, and code corpora.
    """
    norm = unicodedata.normalize("NFKC", text).casefold()
    if mask_numbers:
        norm = _NUM_RE.sub(" num ", norm)
    cleaned = _NON_ALNUM_UNICODE_RE.sub(" ", norm)
    return _WS_RE.sub(" ", cleaned).strip()


def _ngrams_from_canonical(
    canon: str,
    word_n: int = 5,
    char_k: int = 5,
) -> Tuple[Set[str], Set[str]]:
    tokens = canon.split()
    word_ngrams: Set[str] = set()
    if len(tokens) >= word_n:
        for i in range(len(tokens) - word_n + 1):
            word_ngrams.add(" ".join(tokens[i : i + word_n]))
    elif tokens:
        word_ngrams.add(" ".join(tokens))

    compact = canon.replace(" ", "_")
    char_shingles: Set[str] = set()
    if len(compact) >= char_k:
        for i in range(len(compact) - char_k + 1):
            char_shingles.add(compact[i : i + char_k])
    elif compact:
        char_shingles.add(compact)

    return word_ngrams, char_shingles


def extract_canonical_ngrams(
    text: str,
    word_n: int = 5,
    char_k: int = 5,
    mask_numbers: bool = False,
) -> Tuple[Set[str], Set[str]]:
    canon = canonicalize_text(text, mask_numbers=mask_numbers)
    return _ngrams_from_canonical(canon, word_n=word_n, char_k=char_k)


def _blake64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]


def simhash64(features: Sequence[str] | Set[str]) -> int:
    """
    Charikar (2002) 64-bit SimHash over canonical features using BLAKE2b-64.
    Uses exact 1024-bit SWAR (64 x 16-bit lane) bit-parallel accumulation in C-int
    arithmetic, producing bit-for-bit identical hashes ~20x faster than scalar loops.
    """
    n_feats = len(features)
    if n_feats == 0:
        return 0
    if n_feats < 65535:
        swar = _SWAR_BYTE_16
        blake = hashlib.blake2b
        lanes = 0
        for feat in features:
            d = blake(feat.encode("utf-8"), digest_size=8).digest()
            lanes += (
                swar[d[0]]
                | (swar[d[1]] << 128)
                | (swar[d[2]] << 256)
                | (swar[d[3]] << 384)
                | (swar[d[4]] << 512)
                | (swar[d[5]] << 640)
                | (swar[d[6]] << 768)
                | (swar[d[7]] << 896)
            )
        out = 0
        for bit in range(64):
            ones = (lanes >> (16 * bit)) & 0xFFFF
            if (ones << 1) >= n_feats:
                out |= 1 << bit
        return out

    acc = [0] * 64
    for feat in features:
        h = _blake64(feat)
        for bit in range(64):
            if (h >> bit) & 1:
                acc[bit] += 1
            else:
                acc[bit] -= 1
    out = 0
    for bit in range(64):
        if acc[bit] >= 0:
            out |= 1 << bit
    return out


def hamming_distance64(a: int, b: int) -> int:
    return ((a ^ b) & 0xFFFFFFFFFFFFFFFF).bit_count()


def _simhash_pigeonhole_slices(max_hamming: int) -> List[Tuple[int, int]]:
    """
    Partitions 64 bits into (max_hamming + 1) disjoint bit slices (offset, mask).
    By the Pigeonhole Principle (machine-checked in LeakSpecCausalMask.lean,
    simhash_pigeonhole_zero_block), any two 64-bit hashes with Hamming distance <= k
    MUST match identically in at least one of the (k + 1) disjoint blocks, giving
    exact O(1) bucket lookup without disabling SimHash above 5,000 rows.
    """
    n_blocks = max(1, min(32, max_hamming + 1))
    base = 64 // n_blocks
    rem = 64 % n_blocks
    slices: List[Tuple[int, int]] = []
    offset = 0
    for b in range(n_blocks):
        width = base + (1 if b < rem else 0)
        mask = (1 << width) - 1
        slices.append((offset, mask))
        offset += width
    return slices


def _make_minhash_params(num_perm: int, seed: int = 20261126) -> List[Tuple[int, int]]:
    params: List[Tuple[int, int]] = []
    for i in range(num_perm):
        seed_bytes = f"leakspec-minhash-{seed}-{i}".encode("utf-8")
        digest = hashlib.blake2b(seed_bytes, digest_size=16).digest()
        raw_a, raw_b = struct.unpack("<QQ", digest)
        a = (raw_a % (_MERSENNE_P - 1)) + 1
        b = raw_b % _MERSENNE_P
        params.append((a, b))
    return params


def minhash_signature(
    shingles: Set[str],
    params: Sequence[Tuple[int, int]],
) -> Tuple[int, ...]:
    """Broder (1997) universal linear MinHash signature modulo 2^61 - 1."""
    p = _MERSENNE_P
    if not shingles:
        return tuple(p for _ in params)
    hashed = [_blake64(s) % p for s in shingles]
    sig: List[int] = []
    for a, b in params:
        mval = p
        for x in hashed:
            v = a * x + b
            v = (v & p) + (v >> 61)
            if v >= p:
                v -= p
            if v < mval:
                mval = v
        sig.append(mval)
    return tuple(sig)


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / float(union) if union > 0 else 0.0


def containment_ratio(query: Set[str], corpus: Set[str]) -> float:
    if not query:
        return 0.0
    return len(query & corpus) / float(len(query))


@dataclass(frozen=True)
class RecordFingerprint:
    record_id: str
    canonical_sha256: str
    simhash_64: int
    minhash_sig: Tuple[int, ...]
    word_ngrams: Set[str]
    char_shingles: Set[str]


@dataclass
class ContaminationIndex:
    """Reusable sublinear Pigeonhole SimHash64 + MinHash-LSH + Inverted n-gram index."""

    word_n: int
    char_k: int
    num_perm: int
    lsh_bands: int
    rows_per_band: int
    simhash_max_hamming: int
    mask_numbers: bool
    use_pigeonhole: bool
    mh_params: List[Tuple[int, int]]
    sh_slices: List[Tuple[int, int]]
    train_fps: List[RecordFingerprint]
    exact_index: Dict[str, List[int]]
    lsh_buckets: List[Dict[Tuple[int, ...], List[int]]]
    simhash_tables: List[Dict[int, List[int]]]
    ngram_inverted: Dict[str, List[int]]
    swar_cache: Dict[str, int]
    h61_cache: Dict[str, int]


def _fingerprint_record(
    rec_id: str,
    raw_text: str,
    word_n: int,
    char_k: int,
    mask_numbers: bool,
    mh_params: Sequence[Tuple[int, int]],
    swar_cache: Dict[str, int],
    h61_cache: Dict[str, int],
) -> RecordFingerprint:
    p_mersenne = _MERSENNE_P
    swar_tbl = _SWAR_BYTE_16
    canon = canonicalize_text(raw_text, mask_numbers=mask_numbers)
    sha = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    w_ng, c_sh = _ngrams_from_canonical(canon, word_n=word_n, char_k=char_k)

    union_feats = w_ng | c_sh
    n_feats = len(union_feats)
    if n_feats == 0:
        sh64 = 0
    elif n_feats < 65535:
        lanes = 0
        for ft in union_feats:
            val = swar_cache.get(ft)
            if val is None:
                d = hashlib.blake2b(ft.encode("utf-8"), digest_size=8).digest()
                val = (
                    swar_tbl[d[0]]
                    | (swar_tbl[d[1]] << 128)
                    | (swar_tbl[d[2]] << 256)
                    | (swar_tbl[d[3]] << 384)
                    | (swar_tbl[d[4]] << 512)
                    | (swar_tbl[d[5]] << 640)
                    | (swar_tbl[d[6]] << 768)
                    | (swar_tbl[d[7]] << 896)
                )
                if len(swar_cache) < 300000:
                    swar_cache[ft] = val
            lanes += val
        sh64 = 0
        for bit in range(64):
            if (((lanes >> (16 * bit)) & 0xFFFF) << 1) >= n_feats:
                sh64 |= 1 << bit
    else:
        sh64 = simhash64(union_feats)

    if not c_sh:
        mhsig = tuple(p_mersenne for _ in mh_params)
    else:
        hashed: List[int] = []
        for s in c_sh:
            hv = h61_cache.get(s)
            if hv is None:
                hv = _blake64(s) % p_mersenne
                if len(h61_cache) < 300000:
                    h61_cache[s] = hv
            hashed.append(hv)
        sig_list: List[int] = []
        for a, b in mh_params:
            mval = p_mersenne
            for x in hashed:
                v = a * x + b
                v = (v & p_mersenne) + (v >> 61)
                if v >= p_mersenne:
                    v -= p_mersenne
                if v < mval:
                    mval = v
            sig_list.append(mval)
        mhsig = tuple(sig_list)

    return RecordFingerprint(
        record_id=rec_id,
        canonical_sha256=sha,
        simhash_64=sh64,
        minhash_sig=mhsig,
        word_ngrams=w_ng,
        char_shingles=c_sh,
    )


def build_contamination_index(
    train_records: Sequence[Tuple[str, str]],
    word_n: int = 5,
    char_k: int = 5,
    num_perm: int = 64,
    lsh_bands: int = 16,
    simhash_max_hamming: int = 6,
    mask_numbers: bool = False,
) -> ContaminationIndex:
    """Builds a reusable sublinear Pigeonhole SimHash64 + MinHash-LSH contamination index."""
    if isinstance(word_n, bool) or not isinstance(word_n, int) or word_n < 1:
        raise ValueError(f"word_n must be an integer >= 1, got {word_n!r}")
    if isinstance(char_k, bool) or not isinstance(char_k, int) or char_k < 1:
        raise ValueError(f"char_k must be an integer >= 1, got {char_k!r}")
    if (
        isinstance(num_perm, bool)
        or isinstance(lsh_bands, bool)
        or not isinstance(num_perm, int)
        or not isinstance(lsh_bands, int)
        or num_perm <= 0
        or lsh_bands <= 0
        or (num_perm % lsh_bands != 0)
    ):
        raise ValueError(f"num_perm ({num_perm!r}) must be a positive int divisible by lsh_bands ({lsh_bands!r})")
    if isinstance(simhash_max_hamming, bool) or not isinstance(simhash_max_hamming, int) or simhash_max_hamming < 0 or simhash_max_hamming > 64:
        raise ValueError(f"simhash_max_hamming must be an int in [0, 64], got {simhash_max_hamming!r}")

    rows_per_band = num_perm // lsh_bands
    mh_params = _make_minhash_params(num_perm)
    use_pigeonhole = simhash_max_hamming <= 31
    sh_slices = _simhash_pigeonhole_slices(simhash_max_hamming) if use_pigeonhole else []
    swar_cache: Dict[str, int] = {}
    h61_cache: Dict[str, int] = {}

    train_fps: List[RecordFingerprint] = [
        _fingerprint_record(
            rid,
            txt,
            word_n=word_n,
            char_k=char_k,
            mask_numbers=mask_numbers,
            mh_params=mh_params,
            swar_cache=swar_cache,
            h61_cache=h61_cache,
        )
        for rid, txt in train_records
    ]

    exact_index: Dict[str, List[int]] = {}
    lsh_buckets: List[Dict[Tuple[int, ...], List[int]]] = [{} for _ in range(lsh_bands)]
    simhash_tables: List[Dict[int, List[int]]] = [{} for _ in sh_slices]
    ngram_inverted: Dict[str, List[int]] = {}

    for idx, fp in enumerate(train_fps):
        exact_index.setdefault(fp.canonical_sha256, []).append(idx)
        for b in range(lsh_bands):
            band_key = fp.minhash_sig[b * rows_per_band : (b + 1) * rows_per_band]
            lsh_buckets[b].setdefault(band_key, []).append(idx)
        if use_pigeonhole:
            for s_idx, (offset, mask) in enumerate(sh_slices):
                block_key = (fp.simhash_64 >> offset) & mask
                simhash_tables[s_idx].setdefault(block_key, []).append(idx)
        for ng in fp.word_ngrams:
            ngram_inverted.setdefault(ng, []).append(idx)

    return ContaminationIndex(
        word_n=word_n,
        char_k=char_k,
        num_perm=num_perm,
        lsh_bands=lsh_bands,
        rows_per_band=rows_per_band,
        simhash_max_hamming=simhash_max_hamming,
        mask_numbers=mask_numbers,
        use_pigeonhole=use_pigeonhole,
        mh_params=mh_params,
        sh_slices=sh_slices,
        train_fps=train_fps,
        exact_index=exact_index,
        lsh_buckets=lsh_buckets,
        simhash_tables=simhash_tables,
        ngram_inverted=ngram_inverted,
        swar_cache=swar_cache,
        h61_cache=h61_cache,
    )


def audit_against_contamination_index(
    index: ContaminationIndex,
    test_records: Sequence[Tuple[str, str]],
    jaccard_threshold: float = 0.50,
    ngram_containment_threshold: float = 0.60,
) -> Dict[str, object]:
    """Queries a pre-built ContaminationIndex in sublinear O(1) bucket time per test item."""
    if (
        isinstance(jaccard_threshold, bool)
        or not isinstance(jaccard_threshold, (int, float))
        or not (0.0 <= float(jaccard_threshold) <= 1.0)
    ):
        raise ValueError(f"jaccard_threshold must be a finite float in [0.0, 1.0], got {jaccard_threshold!r}")
    if (
        isinstance(ngram_containment_threshold, bool)
        or not isinstance(ngram_containment_threshold, (int, float))
        or not (0.0 <= float(ngram_containment_threshold) <= 1.0)
    ):
        raise ValueError(
            f"ngram_containment_threshold must be a finite float in [0.0, 1.0], got {ngram_containment_threshold!r}"
        )
    findings: List[Dict[str, object]] = []
    contaminated_test_ids: Set[str] = set()
    exact_matches = 0
    ngram_matches = 0
    near_dup_matches = 0

    train_fps = index.train_fps
    rows_per_band = index.rows_per_band
    lsh_bands = index.lsh_bands
    sh_slices = index.sh_slices
    simhash_max_hamming = index.simhash_max_hamming

    for te_id, te_txt in test_records:
        te_fp = _fingerprint_record(
            te_id,
            te_txt,
            word_n=index.word_n,
            char_k=index.char_k,
            mask_numbers=index.mask_numbers,
            mh_params=index.mh_params,
            swar_cache=index.swar_cache,
            h61_cache=index.h61_cache,
        )

        if te_fp.canonical_sha256 in index.exact_index:
            matched_ids = [train_fps[i].record_id for i in index.exact_index[te_fp.canonical_sha256][:5]]
            exact_matches += 1
            contaminated_test_ids.add(te_id)
            findings.append({
                "rule_id": "LS-TIME-006",
                "severity": "error",
                "match_type": "EXACT_CANONICAL_SHA256",
                "test_id": te_id,
                "train_id": matched_ids[0],
                "matched_train_ids": matched_ids,
                "canonical_sha256": te_fp.canonical_sha256,
                "containment": 1.0,
                "jaccard": 1.0,
                "simhash_hamming": 0,
                "detail": (
                    f"Exact canonical train-test duplicate: test='{te_id}' matches "
                    f"train='{matched_ids[0]}' (sha256={te_fp.canonical_sha256[:12]})."
                ),
            })
            continue

        lsh_hit_set: Set[int] = set()
        ngram_hit_set: Set[int] = set()
        simhash_cand_set: Set[int] = set()

        for b in range(lsh_bands):
            band_key = te_fp.minhash_sig[b * rows_per_band : (b + 1) * rows_per_band]
            for tr_idx in index.lsh_buckets[b].get(band_key, ()):
                lsh_hit_set.add(tr_idx)

        for ng in te_fp.word_ngrams:
            for tr_idx in index.ngram_inverted.get(ng, ()):
                ngram_hit_set.add(tr_idx)

        if index.use_pigeonhole:
            for s_idx, (offset, mask) in enumerate(sh_slices):
                block_key = (te_fp.simhash_64 >> offset) & mask
                for tr_idx in index.simhash_tables[s_idx].get(block_key, ()):
                    simhash_cand_set.add(tr_idx)
        else:
            for tr_idx, tr_fp in enumerate(train_fps):
                if hamming_distance64(te_fp.simhash_64, tr_fp.simhash_64) <= simhash_max_hamming:
                    simhash_cand_set.add(tr_idx)

        best_containment = 0.0
        best_containment_tr = ""
        for tr_idx in sorted(ngram_hit_set):
            tr_fp = train_fps[tr_idx]
            cont = containment_ratio(te_fp.word_ngrams, tr_fp.word_ngrams)
            if cont > best_containment:
                best_containment = cont
                best_containment_tr = tr_fp.record_id

        best_near_score = -1.0
        best_near_info: Tuple[str, float, int, bool] | None = None
        near_candidates = lsh_hit_set | simhash_cand_set
        for tr_idx in sorted(near_candidates):
            tr_fp = train_fps[tr_idx]
            hdist = hamming_distance64(te_fp.simhash_64, tr_fp.simhash_64)
            is_lsh = tr_idx in lsh_hit_set
            if hdist > simhash_max_hamming and not is_lsh:
                continue
            jac = jaccard_similarity(te_fp.char_shingles, tr_fp.char_shingles)
            if (hdist <= simhash_max_hamming) or (is_lsh and jac >= jaccard_threshold):
                score = jac + (1.0 - hdist / 64.0)
                if score > best_near_score:
                    best_near_score = score
                    best_near_info = (tr_fp.record_id, jac, hdist, is_lsh)

        if best_containment >= ngram_containment_threshold:
            ngram_matches += 1
            contaminated_test_ids.add(te_id)
            findings.append({
                "rule_id": "LS-TIME-006",
                "severity": "error",
                "match_type": "CANONICAL_NGRAM_CONTAINMENT",
                "test_id": te_id,
                "train_id": best_containment_tr,
                "canonical_sha256": te_fp.canonical_sha256,
                "containment": round(best_containment, 4),
                "detail": (
                    f"Canonical {index.word_n}-gram benchmark contamination: test='{te_id}' "
                    f"overlaps train='{best_containment_tr}' (containment={best_containment:.3f})."
                ),
            })
        elif best_near_info is not None:
            tr_id, jac, hdist, is_lsh = best_near_info
            near_dup_matches += 1
            contaminated_test_ids.add(te_id)
            findings.append({
                "rule_id": "LS-TIME-007",
                "severity": "warning",
                "match_type": "SIMHASH_MINHASH_LSH_NEARDUP",
                "test_id": te_id,
                "train_id": tr_id,
                "canonical_sha256": te_fp.canonical_sha256,
                "jaccard": round(jac, 4),
                "simhash_hamming": hdist,
                "lsh_band_hit": is_lsh,
                "detail": (
                    f"Near-duplicate contamination: test='{te_id}' ~ train='{tr_id}' "
                    f"(SimHash64_hamming={hdist}, char_jaccard={jac:.3f}, lsh_hit={is_lsh})."
                ),
            })

    n_test = len(test_records)
    return {
        "layer": "L3_CONTAMINATION_LSH",
        "n_train": len(train_fps),
        "n_test": n_test,
        "pigeonhole_blocks": len(sh_slices) if index.use_pigeonhole else 0,
        "exact_matches": exact_matches,
        "ngram_matches": ngram_matches,
        "near_dup_matches": near_dup_matches,
        "contaminated_test_count": len(contaminated_test_ids),
        "contamination_rate": round(len(contaminated_test_ids) / float(max(n_test, 1)), 6),
        "verdict": "CONTAMINATION_DETECTED" if findings else "BENCHMARK_CLEAN",
        "findings": findings,
    }


def audit_train_test_contamination(
    train_records: Sequence[Tuple[str, str]],
    test_records: Sequence[Tuple[str, str]],
    word_n: int = 5,
    char_k: int = 5,
    num_perm: int = 64,
    lsh_bands: int = 16,
    simhash_max_hamming: int = 6,
    jaccard_threshold: float = 0.50,
    ngram_containment_threshold: float = 0.60,
    mask_numbers: bool = False,
) -> Dict[str, object]:
    """
    Audits train vs. test benchmark records with 100% exact Pigeonhole Multi-Block
    SimHash64 recall at arbitrary corpus sizes (no 5,000-row brute-force cutoff),
    MinHash-LSH banding, and Unicode-aware canonical n-gram containment.
    """
    index = build_contamination_index(
        train_records=train_records,
        word_n=word_n,
        char_k=char_k,
        num_perm=num_perm,
        lsh_bands=lsh_bands,
        simhash_max_hamming=simhash_max_hamming,
        mask_numbers=mask_numbers,
    )
    return audit_against_contamination_index(
        index=index,
        test_records=test_records,
        jaccard_threshold=jaccard_threshold,
        ngram_containment_threshold=ngram_containment_threshold,
    )
