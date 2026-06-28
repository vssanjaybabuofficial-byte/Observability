from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Chronicle


def branch_at(chronicle: "Chronicle", eid: str) -> "Chronicle":
    """Return a new Chronicle containing `eid` and all its causal ancestors.

    Because events are content-addressed, the new Chronicle shares ids with the
    original — `Chronicle.diff(orig, branch)` is exact set algebra, not fuzzy
    matching. The caller can record() additional events on top of the branch.
    """
    from .core import Chronicle
    if eid not in chronicle._events:
        raise KeyError(eid)
    keep = chronicle.ancestors(eid) | {eid}
    new = Chronicle(embedder=chronicle._embedder)
    for i in chronicle._order:
        if i in keep:
            evt = chronicle._events[i]
            new._events[evt.id] = evt
            new._order.append(evt.id)
            for p in evt.parents:
                new._children[p].add(evt.id)
            new._index.add(evt)
    return new


def diff(a: "Chronicle", b: "Chronicle") -> dict:
    """Exact set difference over event ids."""
    aids = set(a._events.keys())
    bids = set(b._events.keys())
    return {
        "only_in_a": sorted(aids - bids),
        "only_in_b": sorted(bids - aids),
        "common": sorted(aids & bids),
    }
