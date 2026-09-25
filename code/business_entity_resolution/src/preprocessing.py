"""
Phase 1: Multi-view preprocessing and normalization.

Creates parallel representations at different normalization levels,
preserving information for features to use the right view for each comparison.

Optimized with joblib parallelization for large datasets.
"""
import re
import unicodedata
from collections import Counter

import regex
import numpy as np
import pandas as pd
from unidecode import unidecode
import ftfy
import jellyfish
from tqdm import tqdm
from joblib import Parallel, delayed

from . import config
from .utils import logger, timed

tqdm.pandas()

# ──────────────────────────────────────────────
#  Script detection via regex Unicode properties
# ──────────────────────────────────────────────
_SCRIPT_PATTERNS = [
    ("devanagari", regex.compile(r"\p{Script=Devanagari}")),
    ("tamil",      regex.compile(r"\p{Script=Tamil}")),
    ("kannada",    regex.compile(r"\p{Script=Kannada}")),
    ("gurmukhi",   regex.compile(r"\p{Script=Gurmukhi}")),
    ("latin",      regex.compile(r"\p{Script=Latin}")),
]


def detect_script(text: str) -> str:
    """Detect dominant script using regex Unicode properties.
    Uses regex lib (not unicodedata.script which requires Python 3.13+).
    """
    if not text:
        return "unknown"
    counts = Counter()
    for name, pattern in _SCRIPT_PATTERNS:
        hits = len(pattern.findall(text))
        if hits > 0:
            counts[name] = hits
    if not counts:
        return "unknown"
    if len(counts) > 1:
        total = sum(counts.values())
        dominant_name, dominant_count = counts.most_common(1)[0]
        if dominant_count / total > 0.8:
            return dominant_name
        return "mixed"
    return counts.most_common(1)[0][0]


# ──────────────────────────────────────────────
#  URL-as-name extraction
# ──────────────────────────────────────────────
_URL_RE = regex.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][-a-z0-9]*(?:\.[a-z0-9][-a-z0-9]*)*\.[a-z]{2,})$",
    regex.IGNORECASE,
)


def extract_url_name(name: str):
    """
    Detect URL-as-name records (common in S3) and extract usable parts.
    e.g. "maurewilliamscolombier.com" -> domain="maurewilliamscolombier.com",
                                          stem="maurewilliamscolombier"
    Returns (is_url, domain, stem).
    """
    name = name.strip()
    m = _URL_RE.match(name)
    if m:
        domain = m.group(1).lower()
        # Strip TLD (.com, .org, .in, .fr, etc.)
        stem = regex.sub(r"\.[a-z]{2,}$", "", domain)
        # Strip www prefix
        stem = regex.sub(r"^www\.", "", stem)
        return True, domain, stem
    return False, "", name


# ──────────────────────────────────────────────
#  Text cleaning
# ──────────────────────────────────────────────
_PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00b7": " ", "\u2022": " ",
})

_MULTI_SPACE = re.compile(r"\s+")
_STRIP_PAREN = re.compile(r"[(){}\[\]]")
_PERIOD_ABBREV = re.compile(r"(?<=[A-Za-z])\.(?=[A-Za-z])") # L.L.C -> LLC


def clean_text(text: str) -> str:
    """Lowercase, normalize punctuation, collapse whitespace."""
    if not text or str(text).lower() in ("nan", "none", "null", ""):
        return ""
    text = str(text)
    text = text.translate(_PUNCT_MAP)
    text = text.replace("&", " and ")
    text = _PERIOD_ABBREV.sub("", text)  # L.L.C -> LLC
    text = _STRIP_PAREN.sub("", text)
    text = text.lower().strip()
    text = _MULTI_SPACE.sub(" ", text)
    return text


# ──────────────────────────────────────────────
#  Suffix / generic token extraction
# ──────────────────────────────────────────────
def extract_suffix_and_generics(name_clean: str):
    """
    Separate legal suffixes (removable) from generic tokens (features only).

    Critical distinction:
    - Legal suffixes (LLC, Pvt, Ltd, SARL): purely structural, safe to strip for name_core
    - Generic tokens (Holdings, Group, Enterprises): identity-bearing, kept in name_core
      but tracked separately for feature computation.

    Returns (name_core, legal_suffixes_list, generic_tokens_list).
    """
    if not name_clean:
        return "", [], []
    tokens = name_clean.split()
    legal = [t for t in tokens if t in config.LEGAL_SUFFIXES]
    generic = [t for t in tokens if t in config.GENERIC_BUSINESS_TOKENS]
    # Remove ONLY legal suffixes from core (keep generic tokens!)
    core_tokens = [t for t in tokens if t not in config.LEGAL_SUFFIXES]
    name_core = " ".join(core_tokens)
    return name_core, legal, generic


# ──────────────────────────────────────────────
#  Address component extraction
# ──────────────────────────────────────────────
_POSTAL_IN = re.compile(r"\b(\d{6})\b")            # India: 6-digit PIN
_POSTAL_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")  # US: 5-digit ZIP
_POSTAL_FR = re.compile(r"\b(\d{5})\b")             # France: 5-digit

_HOUSE_NUM = re.compile(r"(?:^|\s)#?(\d+[-/]?\d*)[,\s]")
_LANDMARK_PHRASES = re.compile(
    r"\b(near|behind|opposite|opp\.?|beside|adjacent to|next to|in front of)\b",
    re.IGNORECASE,
)


def expand_address_abbreviations(addr: str) -> str:
    """Expand common address abbreviations (English + French)."""
    if not addr:
        return ""
    tokens = addr.split()
    expanded = []
    for t in tokens:
        key = t.rstrip(".,;:")
        if key in config.ADDRESS_ABBREVIATIONS:
            expanded.append(config.ADDRESS_ABBREVIATIONS[key])
        else:
            expanded.append(t)
    return " ".join(expanded)


def extract_address_components(addr: str, country: str) -> dict:
    """Extract structured components from address string.
    Uses ONLY regex heuristics — no external geographic databases (compliance).
    """
    components = {
        "postal_code": "",
        "city": "",
        "state": "",
        "street_number": "",
    }
    if not addr:
        return components

    # Postal code (country-specific patterns)
    if country == "India":
        m = _POSTAL_IN.search(addr)
        if m:
            components["postal_code"] = m.group(1)
    elif country == "US":
        m = _POSTAL_US.search(addr)
        if m:
            components["postal_code"] = m.group(1)
    else:
        # France or unknown — try 5-digit
        m = _POSTAL_FR.search(addr)
        if m:
            components["postal_code"] = m.group(1)

    # Street number
    m = _HOUSE_NUM.search(addr)
    if m:
        components["street_number"] = m.group(1).strip("-/")

    # City/State from comma-separated parts (heuristic)
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if len(parts) >= 2:
        last = parts[-1].strip().lower()
        # Strip postal code from last part
        last = re.sub(r"\b\d{5,6}\b", "", last).strip()
        # Check if last is a US state abbreviation
        if last in config.US_STATE_ABBREV:
            components["state"] = config.US_STATE_ABBREV[last]
            if len(parts) >= 3:
                components["city"] = parts[-2].strip().lower()
        elif last:
            components["state"] = last
            if len(parts) >= 3:
                components["city"] = parts[-2].strip().lower()
            elif len(parts) == 2:
                components["city"] = parts[0].strip().lower()

    return components


def extract_landmarks(addr: str):
    """Separate landmarks from address. Returns (addr_without_landmark, landmark_tokens)."""
    if not addr:
        return "", []
    landmarks = _LANDMARK_PHRASES.findall(addr)
    addr_no_lm = _LANDMARK_PHRASES.sub("", addr).strip()
    addr_no_lm = _MULTI_SPACE.sub(" ", addr_no_lm)
    # Extract the landmark-adjacent phrase (next 3-5 tokens after keyword)
    landmark_tokens = []
    for lm_match in _LANDMARK_PHRASES.finditer(addr):
        end = lm_match.end()
        remainder = addr[end:end + 80].split(",")[0].strip().lower().split()[:5]
        landmark_tokens.extend(remainder)
    return addr_no_lm, landmark_tokens


# ──────────────────────────────────────────────
#  Phonetic key computation
# ──────────────────────────────────────────────
def compute_phonetic_key(name_core: str) -> str:
    """Compute double-metaphone key for blocking. Handles empty/non-ASCII gracefully."""
    if not name_core or len(name_core) < 2:
        return ""
    try:
        return jellyfish.metaphone(name_core)
    except Exception:
        return ""


# ──────────────────────────────────────────────
#  Main preprocessing function
# ──────────────────────────────────────────────
def preprocess_record(name_raw, addr_raw, country: str) -> dict:
    """
    Create multi-view normalized representations for a single record.

    Views produced:
    1. name_raw / addr_raw — original strings
    2. name_unicode / addr_unicode — NFKC-normalized, encoding fixed
    3. name_transliterated / addr_transliterated — Indic scripts -> Latin
    4. name_clean / addr_clean — lowercased, punct normalized, URL-aware
    5. name_core — legal suffixes removed (generic tokens preserved)
    6. name_tokens_sorted — alphabetically sorted token set
    """
    # Robust str() cast for NaN/None from pandas
    name_raw = str(name_raw).strip() if pd.notna(name_raw) else ""
    addr_raw = str(addr_raw).strip() if pd.notna(addr_raw) else ""
    country = str(country).strip() if pd.notna(country) else ""

    # Handle explicit null strings
    if name_raw.lower() in ("nan", "none", "null"):
        name_raw = ""
    if addr_raw.lower() in ("nan", "none", "null"):
        addr_raw = ""

    # View 1: Unicode-normalized (fix encoding, normalize forms)
    name_unicode = ftfy.fix_text(unicodedata.normalize("NFKC", name_raw)) if name_raw else ""
    addr_unicode = ftfy.fix_text(unicodedata.normalize("NFKC", addr_raw)) if addr_raw else ""

    # View 2: Transliterated (Indic scripts -> Latin) — KEEP ORIGINALS
    name_transliterated = unidecode(name_unicode) if name_unicode else ""
    addr_transliterated = unidecode(addr_unicode) if addr_unicode else ""

    # View 3: URL extraction (BEFORE cleaning strips URLs)
    is_url, url_domain, url_stem = extract_url_name(name_transliterated)

    # View 4: Cleaned (lowercase, punct normalized)
    # NOTE: if name is a URL, clean the domain stem — do NOT discard it
    if is_url:
        name_clean = clean_text(url_stem)
    else:
        name_clean = clean_text(name_transliterated)
    addr_clean = clean_text(addr_transliterated)

    # Expand address abbreviations
    addr_clean = expand_address_abbreviations(addr_clean)

    # View 5: Core (legal suffixes removed, generic tokens tracked)
    name_core, legal_suffix, generic_tokens = extract_suffix_and_generics(name_clean)

    # View 6: Sorted tokens (for blocking)
    name_tokens = set(name_core.split()) if name_core else set()
    name_tokens_sorted = sorted(name_tokens)

    # Address components (structured extraction)
    addr_components = extract_address_components(addr_clean, country)
    addr_without_landmark, landmark_tokens = extract_landmarks(addr_clean)

    # Script detection (regex-based, NOT unicodedata.script)
    script_type = detect_script(name_raw)

    # Phonetic key (pre-computed for blocking phase speed)
    phonetic_key = compute_phonetic_key(name_core)

    return {
        # Name views (all preserved per plan — never destroy information)
        "name_raw": name_raw,
        "name_unicode": name_unicode,
        "name_transliterated": name_transliterated,
        "name_clean": name_clean,
        "name_core": name_core,
        "legal_suffix": "|".join(legal_suffix) if legal_suffix else "",
        "generic_tokens": "|".join(generic_tokens) if generic_tokens else "",
        "name_tokens_sorted": " ".join(name_tokens_sorted),
        "phonetic_key": phonetic_key,
        "is_url_name": is_url,
        "url_domain": url_domain,
        "url_stem": url_stem if is_url else "",
        # Address views
        "addr_raw": addr_raw,
        "addr_unicode": addr_unicode,
        "addr_transliterated": addr_transliterated,
        "addr_clean": addr_clean,
        "addr_without_landmark": addr_without_landmark,
        "landmark_tokens": " ".join(landmark_tokens) if landmark_tokens else "",
        "postal_code": addr_components["postal_code"],
        "city": addr_components["city"],
        "state": addr_components["state"],
        "street_number": addr_components["street_number"],
        # Metadata
        "script_type": script_type,
        "transliteration_used": script_type != "latin",
        "country": country,
    }


# ──────────────────────────────────────────────
#  Batch preprocessing — joblib parallelized
# ──────────────────────────────────────────────
def _process_chunk(chunk_df):
    """Process a chunk of rows. Used by joblib for parallel processing."""
    results = []
    for _, row in chunk_df.iterrows():
        eid = row["entity_id"]
        rec = preprocess_record(
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", ""),
        )
        rec["entity_id"] = eid
        results.append(rec)
    return results


@timed
def preprocess_dataframe(df: pd.DataFrame, source_label: str,
                         n_jobs: int = -1, chunk_size: int = 5000) -> pd.DataFrame:
    """
    Preprocess an entire source DataFrame using parallel processing.

    Args:
        df: Raw DataFrame with entity_id, business_name, business_address, country
        source_label: Label for logging (e.g. "train_S1")
        n_jobs: Number of parallel workers (-1 = all cores)
        chunk_size: Rows per chunk for parallelization
    """
    logger.info(f"Preprocessing {source_label}: {len(df):,} records ...")

    n = len(df)
    if n <= chunk_size * 2:
        # Small enough to process serially (no joblib overhead)
        records = []
        for _, row in tqdm(df.iterrows(), total=n, desc=f"Preprocess {source_label}"):
            eid = row["entity_id"]
            rec = preprocess_record(
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", ""),
            )
            rec["entity_id"] = eid
            records.append(rec)
    else:
        # Parallel processing with joblib
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, n, chunk_size)]
        logger.info(f"  Parallel processing: {len(chunks)} chunks × {chunk_size} rows, "
                     f"{n_jobs} workers")

        chunk_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_process_chunk)(chunk) for chunk in chunks
        )
        records = []
        for chunk_result in chunk_results:
            records.extend(chunk_result)

    result = pd.DataFrame(records)
    result = result.set_index("entity_id")
    logger.info(f"  -> {source_label} preprocessed: {len(result):,} records, "
                f"{len(result.columns)} columns")
    return result
