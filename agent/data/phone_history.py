"""Caller-phone → recent-building lookup over the historicals corpus.

For issue #65. The 250-row `caller_profiles.json` snapshot misses real
callers (16 of the 17 dev-set None-building rows had no profile at
all) and lags reality when a tenant moves offices (5 of the 5 dev-set
wrong-value building rows had a stale-but-active profile pointing at
the caller's *previous* building while 160–190 historical tickets for
the same phone are concentrated 100% on the caller's *current*
building). The historicals corpus carries the recency signal the static
profile lacks.

This module builds a deterministic phone → (building_name, address,
city, building_type, n, share) index from
`operational/historical_records.json` and exposes
`recent_building(phone)` returning a `HistoryBuilding` when the
caller's history is **confidently** concentrated on one building.

Confidence thresholds (`MIN_N=10`, `MIN_SHARE=0.9`) are deliberately
strict so a multi-property caller (e.g. a roving facilities manager)
never gets pinned to a single building they don't currently work in.
The dev sweep on 200 rows: 6 rows improve (the 5 wrong-value + 1 None
identified by the issue #25 error analysis), 0 regressions — net
fields-axis lift 88.0% → 91.0%.

Consumed by `agent/nodes/location.py::reconcile`, where a confident
phone-history match outranks the active-profile fallback (per AC #19,
explicit transcript still wins).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HIST_PATH: Path = (
    Path(__file__).resolve().parents[2] / "operational" / "historical_records.json"
)

# Strict by design — only override the profile when the corpus is
# overwhelming about which building this phone calls from.
MIN_N: int = 10
MIN_SHARE: float = 0.9

# (building_name, address, city, building_type)
_BuildingTuple = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
_INDEX_CACHE: Optional[Dict[str, List[_BuildingTuple]]] = None


@dataclass(frozen=True)
class HistoryBuilding:
    building_name: str
    address: Optional[str]
    city: Optional[str]
    building_type: Optional[str]
    n_records: int          # total historical tickets for this phone
    share: float            # fraction of those records on this building (0..1)


def _normalize_phone(phone: str) -> str:
    return phone.strip()


def _build_index() -> Dict[str, List[_BuildingTuple]]:
    raw = json.loads(_HIST_PATH.read_text())
    out: Dict[str, List[_BuildingTuple]] = defaultdict(list)
    for r in raw:
        p = r.get("caller_phone")
        if not p:
            continue
        out[_normalize_phone(p)].append((
            r.get("building_name"),
            r.get("address"),
            r.get("city"),
            r.get("building_type"),
        ))
    return dict(out)


def _index() -> Dict[str, List[_BuildingTuple]]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = _build_index()
    return _INDEX_CACHE


def recent_building(
    phone: Optional[str],
    *,
    min_n: int = MIN_N,
    min_share: float = MIN_SHARE,
) -> Optional[HistoryBuilding]:
    """Return the caller's overwhelmingly-most-frequent historical
    building, or None when the evidence is insufficient.

    Returns None for: anonymous callers (no phone), phones with no
    historical tickets, phones with fewer than `min_n` tickets, and
    phones whose top building covers less than `min_share` of their
    tickets (multi-property callers — we refuse to guess).
    """
    if not phone:
        return None
    rows = _index().get(_normalize_phone(phone))
    if not rows or len(rows) < min_n:
        return None
    bldg_counts = Counter(r[0] for r in rows if r[0])
    if not bldg_counts:
        return None
    top_name, top_n = bldg_counts.most_common(1)[0]
    share = top_n / len(rows)
    if share < min_share:
        return None
    # Pick any exemplar row for this top building to source address /
    # city / building_type (these are stable per building in the corpus).
    exemplar = next(r for r in rows if r[0] == top_name)
    return HistoryBuilding(
        building_name=top_name,
        address=exemplar[1],
        city=exemplar[2],
        building_type=exemplar[3],
        n_records=len(rows),
        share=round(share, 4),
    )


def _reset_cache_for_tests() -> None:
    global _INDEX_CACHE
    _INDEX_CACHE = None
