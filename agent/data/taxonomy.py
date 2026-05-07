"""Canonical label space derived from `operational/historical_records.json`.

`taxonomy.md` describes the categories in prose; the historical corpus
has the canonical *strings* the scorer matches against. This module is
the single source of truth for valid `(category, subcategory)` pairs —
the classifier validates against it, the design doc cross-references it,
and unit tests use it to assert label coverage.

Loaded once from the historicals on first call, cached for the process
lifetime. ~10 categories × ~36 subcategories — small enough that a dict
lookup per classification is free.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

_HISTORICAL_PATH: Path = (
    Path(__file__).resolve().parents[2] / "operational" / "historical_records.json"
)

_TAXONOMY_CACHE: Optional[Dict[str, FrozenSet[str]]] = None


def _build() -> Dict[str, FrozenSet[str]]:
    """Derive the label space from `final_category` / `final_subcategory`
    of every historical record. `final_*` (not `intake_*`) is canonical
    because it reflects the on-site technician's authoritative call."""
    raw = json.loads(_HISTORICAL_PATH.read_text())
    by_cat: Dict[str, set] = defaultdict(set)
    for r in raw:
        cat = r.get("final_category")
        sub = r.get("final_subcategory")
        if cat and sub:
            by_cat[cat].add(sub)
    return {k: frozenset(v) for k, v in by_cat.items()}


def _taxonomy() -> Dict[str, FrozenSet[str]]:
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        _TAXONOMY_CACHE = _build()
    return _TAXONOMY_CACHE


def categories() -> List[str]:
    """Sorted list of every valid top-level category string."""
    return sorted(_taxonomy().keys())


def subcategories_for(category: str) -> List[str]:
    """Sorted list of subcategory strings valid under `category`.

    Returns `[]` for an unknown category.
    """
    return sorted(_taxonomy().get(category, frozenset()))


def all_subcategories() -> List[str]:
    """Sorted flat list of every subcategory across every category."""
    return sorted({s for subs in _taxonomy().values() for s in subs})


def all_pairs() -> List[Tuple[str, str]]:
    """Sorted list of every valid `(category, subcategory)` pair."""
    return sorted(
        (cat, sub)
        for cat, subs in _taxonomy().items()
        for sub in subs
    )


def is_valid_category(category: str) -> bool:
    return category in _taxonomy()


def is_valid_pair(category: str, subcategory: str) -> bool:
    """True iff `subcategory` is a documented child of `category`."""
    return subcategory in _taxonomy().get(category, frozenset())


def category_for_subcategory(subcategory: str) -> Optional[str]:
    """Reverse lookup: which category owns `subcategory`?

    Returns None if the subcategory isn't in the taxonomy. Each
    subcategory string appears under exactly one category in the corpus,
    so the mapping is unambiguous.
    """
    for cat, subs in _taxonomy().items():
        if subcategory in subs:
            return cat
    return None


def _reset_cache_for_tests() -> None:
    global _TAXONOMY_CACHE
    _TAXONOMY_CACHE = None
