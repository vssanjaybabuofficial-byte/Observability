from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


def canonical_json(obj: Any) -> bytes:
    """Stable, deterministic JSON encoding for content addressing."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _normalize(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_normalize(v) for v in x]
    return x


@dataclass(frozen=True)
class Event:
    """A single trace event. Identity is the blake2b hash of its content + parents.

    `kind`, `actor`, `payload`, `parents`, `meta` are all hashed.
    Wall-clock time is intentionally *not* part of identity — store it in `meta`
    if you want it preserved, knowing that doing so makes cross-run identity
    sensitive to it.
    """
    kind: str
    actor: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    parents: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        body = {
            "kind": self.kind,
            "actor": self.actor,
            "payload": _normalize(self.payload),
            "parents": list(self.parents),
            "meta": _normalize(self.meta),
        }
        return hashlib.blake2b(canonical_json(body), digest_size=16).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "actor": self.actor,
            "payload": _normalize(self.payload),
            "parents": list(self.parents),
            "meta": _normalize(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            kind=d["kind"],
            actor=d["actor"],
            payload=d.get("payload", {}),
            parents=tuple(d.get("parents", [])),
            meta=d.get("meta", {}),
        )
