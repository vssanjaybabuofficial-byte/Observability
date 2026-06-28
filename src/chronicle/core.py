from __future__ import annotations
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .event import Event


class Chronicle:
    """A content-addressed causal DAG of agent events.

    Three views over one structure:
      * Causal — `ancestors`, `descendants`, `why`, `affects`
      * Integrity — `verify`, `root` (Merkle)
      * Semantic — `search` (cosine over an embedding overlay)

    Plus first-class counterfactual `branch_at` + `Chronicle.diff`.
    """

    def __init__(self, embedder: Optional[Any] = None):
        from .semantic import HashingEmbedder, SemanticIndex
        self._events: dict[str, Event] = {}
        self._children: dict[str, set[str]] = defaultdict(set)
        self._order: list[str] = []
        self._embedder = embedder if embedder is not None else HashingEmbedder()
        self._index = SemanticIndex(self._embedder)

    # ---- recording ----

    def record(
        self,
        kind: str,
        *,
        actor: str,
        payload: Optional[Mapping[str, Any]] = None,
        parents: Iterable[str] = (),
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Event:
        payload = dict(payload) if payload is not None else {}
        meta = dict(meta) if meta is not None else {}
        parents = tuple(parents)
        for p in parents:
            if p not in self._events:
                raise ValueError(f"parent event {p!r} not recorded")
        evt = Event(kind=kind, actor=actor, payload=payload, parents=parents, meta=meta)
        if evt.id not in self._events:
            self._events[evt.id] = evt
            self._order.append(evt.id)
            for p in parents:
                self._children[p].add(evt.id)
            self._index.add(evt)
        return evt

    # ---- accessors ----

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events[i] for i in self._order)

    def __contains__(self, eid: str) -> bool:
        return eid in self._events

    @property
    def events(self) -> dict[str, Event]:
        return dict(self._events)

    @property
    def order(self) -> list[str]:
        return list(self._order)

    # ---- causal queries ----

    def ancestors(self, eid: str) -> set[str]:
        seen: set[str] = set()
        q = deque([eid])
        while q:
            x = q.popleft()
            for p in self._events[x].parents:
                if p not in seen:
                    seen.add(p)
                    q.append(p)
        return seen

    def descendants(self, eid: str) -> set[str]:
        seen: set[str] = set()
        q = deque([eid])
        while q:
            x = q.popleft()
            for c in self._children.get(x, ()):
                if c not in seen:
                    seen.add(c)
                    q.append(c)
        return seen

    def why(self, eid: str) -> list[Event]:
        anc = self.ancestors(eid) | {eid}
        return [self._events[i] for i in self._order if i in anc]

    def affects(self, eid: str) -> list[Event]:
        desc = self.descendants(eid) | {eid}
        return [self._events[i] for i in self._order if i in desc]

    def lineage(self, eid: str) -> list[Event]:
        """A linear chain via first-parent. Useful for printing."""
        out: list[Event] = []
        cur = self._events[eid]
        while True:
            out.append(cur)
            if not cur.parents:
                break
            cur = self._events[cur.parents[0]]
        return list(reversed(out))

    # ---- semantic ----

    def search(self, query: str, k: int = 5) -> list[tuple[Event, float]]:
        hits = self._index.search(query, k)
        return [(self._events[eid], score) for eid, score in hits]

    # ---- integrity ----

    def verify(self) -> tuple[bool, list[str]]:
        from .integrity import verify as _v
        return _v(self)

    def root(self) -> str:
        from .integrity import root as _r
        return _r(self)

    # ---- branching ----

    def branch_at(self, eid: str) -> "Chronicle":
        from .branch import branch_at as _b
        return _b(self, eid)

    @staticmethod
    def diff(a: "Chronicle", b: "Chronicle") -> dict:
        from .branch import diff as _d
        return _d(a, b)

    # ---- persistence ----

    def dump(self, path) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            for eid in self._order:
                f.write(json.dumps(self._events[eid].to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path, embedder=None) -> "Chronicle":
        c = cls(embedder=embedder)
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                evt = Event.from_dict(d)
                if evt.id != d["id"]:
                    raise ValueError(
                        f"corrupt trace at line: stored id={d['id']!r} but content hashes to {evt.id!r}"
                    )
                c._events[evt.id] = evt
                c._order.append(evt.id)
                for p in evt.parents:
                    c._children[p].add(evt.id)
                c._index.add(evt)
        return c
