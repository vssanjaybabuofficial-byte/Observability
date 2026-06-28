from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Chronicle


def merkle_root(ids: list[str]) -> str:
    """Compute a Merkle root over a list of hex event ids.

    Sorted to be order-independent — the root identifies the *set* of events,
    not the recording order. (Order is recoverable from the parent edges.)
    """
    if not ids:
        return hashlib.blake2b(b"", digest_size=16).hexdigest()
    layer = [bytes.fromhex(i) for i in sorted(set(ids))]
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer), 2):
            a = layer[i]
            b = layer[i + 1] if i + 1 < len(layer) else layer[i]
            nxt.append(hashlib.blake2b(a + b, digest_size=16).digest())
        layer = nxt
    return layer[0].hex()


def verify(chronicle: "Chronicle") -> tuple[bool, list[str]]:
    """Return (ok, bad_ids). bad_ids contains stored keys whose value's content
    no longer hashes to the key, or events whose parents are missing.
    """
    bad: list[str] = []
    for stored_id, evt in chronicle._events.items():
        if evt.id != stored_id:
            bad.append(stored_id)
            continue
        for p in evt.parents:
            if p not in chronicle._events:
                bad.append(stored_id)
                break
    return (not bad, bad)


def root(chronicle: "Chronicle") -> str:
    return merkle_root(list(chronicle._events.keys()))
