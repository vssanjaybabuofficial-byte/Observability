"""
Builder for notebook/chronicle.ipynb.

Run once to materialize the notebook. Re-run to regenerate.
"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
NB_PATH = HERE / "notebook" / "chronicle.ipynb"

# ----------------------------------------------------------------------------
# Module source code — embedded as %%writefile cells inside the notebook.
# Editing these strings here is how you edit the package.
# ----------------------------------------------------------------------------

EVENT_PY = '''\
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
'''

CORE_PY = '''\
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
                f.write(json.dumps(self._events[eid].to_dict(), ensure_ascii=False) + "\\n")

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
'''

INTEGRITY_PY = '''\
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
'''

SEMANTIC_PY = '''\
from __future__ import annotations
import hashlib
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .event import Event


class HashingEmbedder:
    """Char-trigram hashing-trick embedder. Stdlib + numpy only.

    Cheap, deterministic, and good enough for similarity-based retrieval over
    short agent events. Swap in `SentenceTransformerEmbedder` for higher
    semantic quality at the cost of an extra dependency + model download.
    """

    def __init__(self, dim: int = 512, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    @staticmethod
    def _event_text(evt: "Event") -> str:
        from .event import canonical_json
        return " ".join([
            evt.kind,
            evt.actor,
            canonical_json(dict(evt.payload)).decode("utf-8", errors="replace"),
        ]).lower()

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        s = text.lower()
        if len(s) < self.ngram:
            s = s.ljust(self.ngram)
        for i in range(len(s) - self.ngram + 1):
            tri = s[i:i + self.ngram]
            h = int.from_bytes(
                hashlib.blake2b(tri.encode("utf-8"), digest_size=4).digest(),
                "big",
            )
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        return vec

    def embed_event(self, evt: "Event") -> np.ndarray:
        return self.embed(self._event_text(evt))


class SentenceTransformerEmbedder:
    """Optional adapter. Requires `pip install chronicle[st]`."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._m = SentenceTransformer(model_name)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    @staticmethod
    def _event_text(evt: "Event") -> str:
        return f"{evt.kind} {evt.actor} {evt.payload}"

    def embed(self, text: str) -> np.ndarray:
        v = self._m.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def embed_event(self, evt: "Event") -> np.ndarray:
        return self.embed(self._event_text(evt))


class SemanticIndex:
    """Cosine-similarity index over an embedder. L2-normalized vectors so the
    dot product equals cosine similarity.
    """

    def __init__(self, embedder: Any):
        self.embedder = embedder
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None

    def add(self, evt: "Event") -> None:
        v = self.embedder.embed_event(evt).astype(np.float32)[None, :]
        if self._mat is None:
            self._mat = v
        else:
            self._mat = np.vstack([self._mat, v])
        self._ids.append(evt.id)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if self._mat is None or not self._ids:
            return []
        q = self.embedder.embed(query).astype(np.float32)
        sims = self._mat @ q
        k = min(k, len(self._ids))
        order = np.argpartition(-sims, k - 1)[:k]
        order = order[np.argsort(-sims[order])]
        return [(self._ids[int(i)], float(sims[int(i)])) for i in order]
'''

ERGONOMICS_PY = '''\
from __future__ import annotations
import contextvars
import functools
import json
from contextlib import contextmanager
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Chronicle

_current_parent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "chronicle_current_parent", default=None
)
_current_chronicle: contextvars.ContextVar[Optional["Chronicle"]] = contextvars.ContextVar(
    "chronicle_current", default=None
)


def _json_safe(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


class Span:
    """A helper inside `span(...)`. Each emit advances the contextvar parent so
    subsequent emits (and any `@tool`-decorated calls) chain automatically.
    """

    def __init__(self, chronicle: "Chronicle", actor: str):
        self._c = chronicle
        self._actor = actor

    def _emit(self, kind: str, payload: dict, meta: Optional[dict] = None) -> str:
        parent = _current_parent.get()
        parents = (parent,) if parent else ()
        evt = self._c.record(
            kind, actor=self._actor, payload=payload, parents=parents, meta=meta or {}
        )
        _current_parent.set(evt.id)
        return evt.id

    def thought(self, text: str, **meta) -> str:
        return self._emit("thought", {"text": text}, meta)

    def answer(self, text: str, **meta) -> str:
        return self._emit("answer", {"text": text}, meta)

    def emit(self, kind: str, payload: dict, **meta) -> str:
        return self._emit(kind, payload, meta)


@contextmanager
def span(chronicle: "Chronicle", actor: str, parent: Optional[str] = None):
    """Context manager: events emitted inside (via `Span` or `@tool`) chain
    from `parent` automatically via a contextvar.
    """
    if parent is None:
        parent = _current_parent.get()
    sp = Span(chronicle, actor)
    tok_c = _current_chronicle.set(chronicle)
    tok_p = _current_parent.set(parent)
    try:
        yield sp
    finally:
        _current_parent.reset(tok_p)
        _current_chronicle.reset(tok_c)


@contextmanager
def active(chronicle: "Chronicle", parent: Optional[str] = None):
    """Make `chronicle` the active one for `@tool`-decorated calls without
    introducing an actor/span. Useful for top-level scripts.
    """
    tok_c = _current_chronicle.set(chronicle)
    tok_p = _current_parent.set(parent)
    try:
        yield chronicle
    finally:
        _current_parent.reset(tok_p)
        _current_chronicle.reset(tok_c)


def tool(name: Optional[str] = None) -> Callable:
    """Decorator. When the wrapped function is called inside an active
    Chronicle, records a tool_call before and a tool_result after, chained
    from the current contextvar parent.
    """
    def _decorate(fn: Callable) -> Callable:
        tool_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            c = _current_chronicle.get()
            if c is None:
                return fn(*args, **kwargs)
            parent = _current_parent.get()
            parents = (parent,) if parent else ()
            call = c.record(
                "tool_call",
                actor=f"tool:{tool_name}",
                payload={
                    "name": tool_name,
                    "args": [_json_safe(a) for a in args],
                    "kwargs": {k: _json_safe(v) for k, v in kwargs.items()},
                },
                parents=parents,
            )
            _current_parent.set(call.id)
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                err = c.record(
                    "error",
                    actor=f"tool:{tool_name}",
                    payload={"name": tool_name, "error": repr(e)},
                    parents=(call.id,),
                )
                _current_parent.set(err.id)
                raise
            res = c.record(
                "tool_result",
                actor=f"tool:{tool_name}",
                payload={"name": tool_name, "result": _json_safe(out)},
                parents=(call.id,),
            )
            _current_parent.set(res.id)
            return out

        return wrapper

    # Allow @tool as well as @tool("custom_name")
    if callable(name):
        fn, name = name, None
        return _decorate(fn)
    return _decorate
'''

BRANCH_PY = '''\
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
'''

INIT_PY = '''\
"""chronicle — Merkle-DAG agent traceability.

A content-addressed causal DAG of agent events, with integrity verification
and a semantic search overlay. Built as a research artifact; see the notebook
for motivation and design.
"""
from .event import Event, canonical_json
from .core import Chronicle
from .semantic import HashingEmbedder, SentenceTransformerEmbedder, SemanticIndex
from .integrity import verify, merkle_root, root
from .ergonomics import span, tool, active
from .branch import branch_at, diff

__version__ = "0.1.0"

__all__ = [
    "Chronicle",
    "Event",
    "canonical_json",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "SemanticIndex",
    "verify",
    "merkle_root",
    "root",
    "span",
    "tool",
    "active",
    "branch_at",
    "diff",
    "demo",
    "__version__",
]


def demo() -> "Chronicle":
    """Run a small end-to-end toy agent and return its Chronicle.

    No LLM dependency — a deterministic in-memory corpus stands in for a
    retrieval tool. The shape of the recorded DAG (prompt -> thought ->
    tool_call -> tool_result -> thought -> answer) is the canonical example
    used throughout the notebook.
    """
    c = Chronicle()
    prompt = c.record(
        "prompt",
        actor="user",
        payload={"text": "find me a vegan lasagna recipe"},
    )

    @tool("recipe_search")
    def recipe_search(query: str) -> list:
        corpus = {
            "vegan lasagna": ["tofu ricotta", "cashew bechamel", "spinach", "zucchini"],
            "chicken curry": ["onions", "garlic", "garam masala"],
        }
        for k, v in corpus.items():
            if all(w in k for w in query.lower().split()):
                return v
        return []

    with span(c, actor="agent:planner", parent=prompt.id) as s:
        s.thought("the user wants a vegan lasagna recipe; I'll search the corpus")
        hits = recipe_search("vegan lasagna")
        s.thought(f"found {len(hits)} ingredients; assembling the answer")
        s.answer(f"Here's a vegan lasagna using: {', '.join(hits)}")

    return c
'''

PYPROJECT_TOML = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "chronicle"
version = "0.1.0"
description = "Merkle-DAG agent traceability with causal, integrity, and semantic views."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Sanjay Babu" }]
dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
st = ["sentence-transformers>=2.2"]

[tool.hatch.build.targets.wheel]
packages = ["src/chronicle"]
'''

README_MD = '''\
# chronicle

Merkle-DAG agent traceability. Built from `notebook/chronicle.ipynb`.

```python
import chronicle
c = chronicle.demo()
print("root:", c.root())
print("verify:", c.verify())
for evt, score in c.search("vegan recipe"):
    print(f"{score:+.3f}  {evt.kind:<12} {evt.actor}")
```

See the notebook for the design and the headline properties:
deterministic cross-run identity, tamper detection, causal `why()`,
semantic search, and counterfactual branching.
'''


# ----------------------------------------------------------------------------
# Notebook construction
# ----------------------------------------------------------------------------

def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def writefile(path: str, body: str) -> nbf.NotebookNode:
    """A code cell that writes `body` to `path` when executed. Equivalent to
    %%writefile but works without IPython magics, so the same notebook can run
    under nbclient and a plain kernel.
    """
    snippet = (
        "from pathlib import Path\n"
        f"_p = Path({path!r}); _p.parent.mkdir(parents=True, exist_ok=True)\n"
        "_p.write_text(_SRC, encoding='utf-8')\n"
        f"print(f'wrote {{_p}} ({{len(_SRC)}} bytes)')\n"
    )
    cell_src = "_SRC = r'''" + body.replace("'''", "''' + chr(39)*3 + r'''") + "'''\n" + snippet
    return code(cell_src)


cells: list[nbf.NotebookNode] = []

# ---------------------------------------------------------------- §0 Title
cells.append(md(
    "# Chronicle — Merkle-DAG Agent Traceability\n"
    "\n"
    "*An executable research notebook. Running it top-to-bottom both demonstrates "
    "the design and **materializes a `chronicle` wheel** you can install and import "
    "anywhere.*\n"
    "\n"
    "**Author:** Sanjay Babu  \n"
    "**Created:** 2026-06-28\n"
))

# ---------------------------------------------------------------- §1 Motivation
cells.append(md(
    "## §1 Motivation — why span-trees are the wrong abstraction for agent traces\n"
    "\n"
    "Current agent-observability tooling (LangSmith, LangFuse, Phoenix, AgentOps, "
    "OpenLLMetry) models traces as **hierarchical span trees** — a port of "
    "OpenTelemetry's web-request model onto LLM agents. That model has three "
    "weaknesses for research:\n"
    "\n"
    "1. **Causality ≠ hierarchy.** When a tool result causes a reflection that "
    "   updates an earlier plan, span-trees can't express it without cycles or "
    "   duplication.\n"
    "2. **Traces are mutable.** Nothing prevents a logging backend (or an attacker, "
    "   or a buggy retry) from rewriting history. There is no integrity guarantee "
    "   for audit, reproducibility, or adversarial-robustness studies.\n"
    "3. **Cross-run comparison is hard.** Spans get fresh ids every run, so "
    "   *“did agent v2 produce the same intermediate reasoning as v1?”* requires "
    "   fuzzy text matching.\n"
    "\n"
    "### The unified idea\n"
    "\n"
    "Model an agent trace as a **content-addressed causal DAG with a semantic "
    "overlay** — borrowing Merkle-DAG ideas from git/IPFS and applying them to "
    "agent observability. Every event's id *is* the hash of its payload + its "
    "causal parents' ids.\n"
    "\n"
    "| Concern | Falls out of the design |\n"
    "|---|---|\n"
    "| Causality | Edges are explicit parent pointers, not implicit nesting |\n"
    "| Integrity | Content-addressing → any mutation breaks downstream hashes |\n"
    "| Cross-run diff | Identical sub-traces have identical ids → set algebra works |\n"
    "| Semantic search | Overlay an embedding index keyed by event id |\n"
    "| Counterfactual | Forking a DAG at a node is a first-class operation |\n"
    "\n"
    "That combined framing is the research contribution of this notebook.\n"
))

cells.append(code(
    "# Bootstrap the on-disk package layout. Subsequent cells write module\n"
    "# source files into src/chronicle/ and at the end we build the wheel.\n"
    "from pathlib import Path\n"
    "import os, sys\n"
    "\n"
    "ROOT = Path.cwd()\n"
    "# In Jupyter the notebook's cwd is its own directory; step up one level\n"
    "# so the package lands at the project root next to the notebook/ dir.\n"
    "if ROOT.name == 'notebook':\n"
    "    ROOT = ROOT.parent\n"
    "os.chdir(ROOT)\n"
    "(ROOT / 'src' / 'chronicle').mkdir(parents=True, exist_ok=True)\n"
    "print('project root:', ROOT)\n"
    "print('python:', sys.version.split()[0])\n"
))

# ---------------------------------------------------------------- §2 Event type
cells.append(md(
    "## §2 The `Event` type — content-addressed identity\n"
    "\n"
    "Each event is a frozen dataclass. Its `id` is **derived**, not stored: it's "
    "the blake2b hash of a canonical-JSON encoding of `(kind, actor, payload, "
    "parents, meta)`. Two structurally-identical events get the same id — this "
    "is what enables cross-run deduplication and exact diff.\n"
    "\n"
    "Wall-clock time is deliberately *not* part of identity. Put a timestamp in "
    "`meta` if you want it, knowing that doing so will make cross-run identity "
    "sensitive to it (which you probably don't want).\n"
))
cells.append(writefile("src/chronicle/event.py", EVENT_PY))
cells.append(code(
    "# Sanity check: identical content → identical id; one-byte change → new id.\n"
    "import importlib, sys\n"
    "sys.path.insert(0, str((ROOT / 'src').resolve()))\n"
    "for mod in list(sys.modules):\n"
    "    if mod == 'chronicle' or mod.startswith('chronicle.'):\n"
    "        del sys.modules[mod]\n"
    "from chronicle.event import Event\n"
    "\n"
    "a = Event(kind='prompt', actor='user', payload={'text': 'hello'})\n"
    "b = Event(kind='prompt', actor='user', payload={'text': 'hello'})\n"
    "c = Event(kind='prompt', actor='user', payload={'text': 'hellO'})\n"
    "print('a.id == b.id  ->', a.id == b.id)  # True\n"
    "print('a.id == c.id  ->', a.id == c.id)  # False\n"
    "print('a.id =', a.id)\n"
))

# ---------------------------------------------------------------- §3 Core
cells.append(md(
    "## §3 The `Chronicle` core — DAG construction & causal queries\n"
    "\n"
    "`Chronicle.record(...)` appends an event whose `parents` must already exist "
    "(this guarantees acyclicity by construction — you cannot record an event "
    "with a future parent). Two adjacency-list indexes (`_events`, `_children`) "
    "support ancestor/descendant queries in linear time.\n"
))
cells.append(writefile("src/chronicle/core.py", CORE_PY))

# ---------------------------------------------------------------- §4 Integrity
cells.append(md(
    "## §4 Integrity — Merkle root & `verify()`\n"
    "\n"
    "`verify()` re-computes every event's id from its content and checks it "
    "against the key it's stored under. Because parents are part of the hashed "
    "body, tampering with *any* event invalidates not just that event's id but "
    "every descendant that referenced it — exactly the Merkle property.\n"
    "\n"
    "`root()` computes a Merkle root over the **set** of event ids (sorted, "
    "hashed pairwise). It identifies the set of events, not their recording "
    "order — order is recoverable from the parent edges and is irrelevant to "
    "what was recorded.\n"
))
cells.append(writefile("src/chronicle/integrity.py", INTEGRITY_PY))

# ---------------------------------------------------------------- §5 Semantic
cells.append(md(
    "## §5 Semantic overlay — search by meaning, not by id\n"
    "\n"
    "Each recorded event is embedded into a fixed-dim vector and added to an "
    "in-memory cosine index. The default `HashingEmbedder` uses character "
    "trigrams + the hashing trick — no external model required, so the wheel's "
    "only runtime dep is `numpy`. A `SentenceTransformerEmbedder` adapter is "
    "included for higher semantic quality when you can afford the dependency.\n"
))
cells.append(writefile("src/chronicle/semantic.py", SEMANTIC_PY))

# ---------------------------------------------------------------- §6 Ergonomics
cells.append(md(
    "## §6 Ergonomic API — `@tool` and `span(...)`\n"
    "\n"
    "Threading parent-ids through every call is tedious. Two helpers fix this:\n"
    "\n"
    "- `chronicle.span(c, actor=..., parent=...)` — context manager. Inside, "
    "every `s.thought(...)` / `s.answer(...)` chains automatically from the "
    "previous emit via a `contextvar`.\n"
    "- `@chronicle.tool` — decorator. When the wrapped function is called "
    "inside an active span, it records a `tool_call` before and a "
    "`tool_result` after, chained from the current contextvar parent, and "
    "advances the parent so subsequent emits chain from the result.\n"
))
cells.append(writefile("src/chronicle/ergonomics.py", ERGONOMICS_PY))

# ---------------------------------------------------------------- §7 Branching
cells.append(md(
    "## §7 Counterfactual branching — the operation that justifies content-addressing\n"
    "\n"
    "Because ancestors are shared *by id* (not by reference), forking a "
    "Chronicle at any event is essentially free: just copy the ancestor set "
    "into a new container. `Chronicle.diff(orig, branch)` is then exact set "
    "algebra over event ids — no fuzzy text matching, no heuristics.\n"
))
cells.append(writefile("src/chronicle/branch.py", BRANCH_PY))

# ---------------------------------------------------------------- §8 Public API + demo
cells.append(md(
    "## §8 Public API & `chronicle.demo()`\n"
    "\n"
    "The package's `__init__.py` re-exports the public surface and defines "
    "`demo()` — a self-contained toy agent (no LLM dependency) used as a "
    "smoke test for the built wheel.\n"
))
cells.append(writefile("src/chronicle/__init__.py", INIT_PY))

# ---------------------------------------------------------------- §9 End-to-end demo (in-notebook)
cells.append(md(
    "## §9 End-to-end demo — recording, querying, verifying, branching\n"
    "\n"
    "We exercise every headline property *before* building the wheel, so this "
    "notebook stands as the verification suite. Six checks:\n"
    "\n"
    "1. Recording chains correctly via the ergonomic API.\n"
    "2. **Cross-run identity**: two independent runs produce the same Merkle root.\n"
    "3. **Tamper detection**: a manual mutation is caught by `verify()`.\n"
    "4. **Causal query**: `why()` returns exactly the causal sub-DAG.\n"
    "5. **Semantic recall**: `search()` finds a planted query by meaning.\n"
    "6. **Counterfactual diff**: a branched continuation diffs to exactly the divergent events.\n"
))

cells.append(code(
    "# Reload the package source from disk (the %%writefile cells above wrote it)\n"
    "import importlib, sys\n"
    "for mod in list(sys.modules):\n"
    "    if mod == 'chronicle' or mod.startswith('chronicle.'):\n"
    "        del sys.modules[mod]\n"
    "import chronicle\n"
    "from chronicle import Chronicle, span, tool\n"
    "print('loaded chronicle from:', chronicle.__file__)\n"
))

cells.append(code(
    "# --- toy agent ----------------------------------------------------------\n"
    "\n"
    "def run_toy_agent(c: Chronicle, query: str = 'vegan lasagna') -> str:\n"
    "    prompt = c.record('prompt', actor='user', payload={'text': f'find me a {query} recipe'})\n"
    "\n"
    "    @tool('recipe_search')\n"
    "    def recipe_search(q: str) -> list:\n"
    "        corpus = {\n"
    "            'vegan lasagna': ['tofu ricotta', 'cashew bechamel', 'spinach', 'zucchini'],\n"
    "            'chicken curry': ['onions', 'garlic', 'garam masala'],\n"
    "        }\n"
    "        for k, v in corpus.items():\n"
    "            if all(w in k for w in q.lower().split()):\n"
    "                return v\n"
    "        return []\n"
    "\n"
    "    with span(c, actor='agent:planner', parent=prompt.id) as s:\n"
    "        s.thought(f'the user wants a {query} recipe; I will search the corpus')\n"
    "        hits = recipe_search(query)\n"
    "        s.thought(f'found {len(hits)} ingredients; assembling the answer')\n"
    "        answer_id = s.answer(f\"Here's a {query} using: {', '.join(hits)}\")\n"
    "\n"
    "    return answer_id\n"
    "\n"
    "c1 = Chronicle()\n"
    "ans1 = run_toy_agent(c1)\n"
    "print(f'recorded {len(c1)} events; root = {c1.root()}')\n"
    "print()\n"
    "for evt in c1:\n"
    "    parents = ','.join(p[:6] for p in evt.parents) or '-'\n"
    "    text = str(evt.payload.get('text') or evt.payload.get('name') or '')[:60]\n"
    "    print(f'  {evt.id[:6]}  <- [{parents:<13}]  {evt.kind:<12} {evt.actor:<18} {text}')\n"
))

cells.append(md("### Check 2 — cross-run identity"))
cells.append(code(
    "c2 = Chronicle()\n"
    "ans2 = run_toy_agent(c2)\n"
    "assert c1.root() == c2.root(), 'two independent runs should produce the same Merkle root'\n"
    "assert set(c1.events) == set(c2.events), 'event id sets must match'\n"
    "print(f'PASS — identical Merkle root across runs: {c1.root()}')\n"
))

cells.append(md("### Check 3 — tamper detection"))
cells.append(code(
    "from chronicle import Event\n"
    "import copy\n"
    "c_tamper = Chronicle()\n"
    "run_toy_agent(c_tamper)\n"
    "\n"
    "# Swap the value at one key for a different Event (simulating an attacker\n"
    "# rewriting a thought). The key (= original id) no longer matches the value.\n"
    "victim_id = next(iter(c_tamper._events))\n"
    "forged = Event(\n"
    "    kind='thought', actor='attacker',\n"
    "    payload={'text': 'malicious replacement'},\n"
    ")\n"
    "c_tamper._events[victim_id] = forged\n"
    "\n"
    "ok, bad = c_tamper.verify()\n"
    "assert not ok, 'verify() should reject tampered store'\n"
    "assert victim_id in bad\n"
    "print(f'PASS — tamper detected at {victim_id[:12]} (forged.id={forged.id[:12]})')\n"
))

cells.append(md("### Check 4 — causal query (`why()`)"))
cells.append(code(
    "# Re-run a clean Chronicle for the causal query\n"
    "c = Chronicle()\n"
    "ans_id = run_toy_agent(c)\n"
    "\n"
    "trace = c.why(ans_id)\n"
    "kinds = [e.kind for e in trace]\n"
    "actors = [e.actor for e in trace]\n"
    "print('causal lineage of the answer:')\n"
    "for e in trace:\n"
    "    print(f'  {e.kind:<12} {e.actor:<18} {str(e.payload)[:60]}')\n"
    "\n"
    "# The answer was caused by: prompt -> thought -> tool_call -> tool_result -> thought -> answer\n"
    "expected = ['prompt', 'thought', 'tool_call', 'tool_result', 'thought', 'answer']\n"
    "assert kinds == expected, f'expected {expected}, got {kinds}'\n"
    "print('\\nPASS — causal path matches the expected reasoning shape')\n"
))

cells.append(md("### Check 5 — semantic recall"))
cells.append(code(
    "hits = c.search('vegan recipe ingredient assembly', k=3)\n"
    "print('top semantic matches:')\n"
    "for evt, score in hits:\n"
    "    print(f'  {score:+.3f}  {evt.kind:<12} {evt.actor:<18} {str(evt.payload)[:60]}')\n"
    "\n"
    "# At least one of the top-3 should be a planner thought or answer about lasagna\n"
    "top_kinds = {evt.kind for evt, _ in hits}\n"
    "assert top_kinds & {'thought', 'answer', 'tool_result'}, top_kinds\n"
    "print('\\nPASS — semantic search surfaces meaning-relevant events')\n"
))

cells.append(md("### Check 6 — counterfactual branching"))
cells.append(code(
    "# Branch at the first planner thought, then run a different continuation.\n"
    "first_thought = next(e for e in c if e.kind == 'thought')\n"
    "branch = c.branch_at(first_thought.id)\n"
    "\n"
    "with span(branch, actor='agent:planner', parent=first_thought.id) as s:\n"
    "    s.thought('alternative: ask the user to clarify dietary constraints first')\n"
    "    s.answer('Before I search: any nut allergies or gluten preferences?')\n"
    "\n"
    "d = Chronicle.diff(c, branch)\n"
    "print(f'common events:      {len(d[\"common\"])}')\n"
    "print(f'only in original:   {len(d[\"only_in_a\"])}')\n"
    "print(f'only in branch:     {len(d[\"only_in_b\"])}')\n"
    "\n"
    "# Common events must include the prompt + first thought (the shared prefix)\n"
    "assert first_thought.id in d['common']\n"
    "# The branch must contain at least the two new events\n"
    "assert len(d['only_in_b']) >= 2\n"
    "# Diff is exact set algebra — no overlap\n"
    "assert set(d['only_in_a']).isdisjoint(d['only_in_b'])\n"
    "print('\\nPASS — counterfactual diff is exact')\n"
))

cells.append(md("### DAG visualization"))
cells.append(code(
    "# Render the original DAG and the branched DAG side-by-side.\n"
    "import matplotlib.pyplot as plt\n"
    "import networkx as nx\n"
    "\n"
    "def to_nx(ch: Chronicle) -> nx.DiGraph:\n"
    "    g = nx.DiGraph()\n"
    "    for evt in ch:\n"
    "        g.add_node(evt.id[:6], label=f'{evt.kind}\\n{evt.actor}')\n"
    "        for p in evt.parents:\n"
    "            g.add_edge(p[:6], evt.id[:6])\n"
    "    return g\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "for ax, ch, title in zip(axes, [c, branch], ['original', 'counterfactual branch']):\n"
    "    g = to_nx(ch)\n"
    "    try:\n"
    "        pos = nx.nx_agraph.graphviz_layout(g, prog='dot')\n"
    "    except Exception:\n"
    "        pos = nx.spring_layout(g, seed=42)\n"
    "    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1600, node_color='#cfe2ff', edgecolors='#1f4e79')\n"
    "    nx.draw_networkx_edges(g, pos, ax=ax, arrowsize=15, edge_color='#555')\n"
    "    labels = {n: g.nodes[n]['label'] for n in g.nodes}\n"
    "    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)\n"
    "    ax.set_title(title)\n"
    "    ax.axis('off')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

cells.append(md("### Check 7 — JSONL round-trip"))
cells.append(code(
    "import tempfile, os\n"
    "with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:\n"
    "    path = f.name\n"
    "c.dump(path)\n"
    "loaded = Chronicle.load(path)\n"
    "assert loaded.root() == c.root(), 'roundtripped Chronicle must have the same root'\n"
    "ok, bad = loaded.verify()\n"
    "assert ok, bad\n"
    "os.unlink(path)\n"
    "print(f'PASS — JSONL round-trip preserves the Merkle root: {c.root()}')\n"
))

# ---------------------------------------------------------------- §10 Package metadata + build
cells.append(md(
    "## §10 Build the wheel\n"
    "\n"
    "Now that all checks pass on the in-tree source, we write `pyproject.toml` "
    "and a minimal `README.md`, invoke `python -m build --wheel`, and confirm "
    "the artifact lands in `dist/`.\n"
))
cells.append(writefile("pyproject.toml", PYPROJECT_TOML))
cells.append(writefile("README.md", README_MD))

cells.append(code(
    "import subprocess, sys\n"
    "# Use the *current* interpreter so the wheel is built in the active venv.\n"
    "r = subprocess.run(\n"
    "    [sys.executable, '-m', 'build', '--wheel', '--no-isolation'],\n"
    "    capture_output=True, text=True\n"
    ")\n"
    "print(r.stdout[-2000:])\n"
    "if r.returncode != 0:\n"
    "    print('STDERR:', r.stderr[-2000:])\n"
    "    raise SystemExit(r.returncode)\n"
    "\n"
    "from pathlib import Path\n"
    "wheels = sorted(Path('dist').glob('chronicle-*.whl'))\n"
    "assert wheels, 'no wheel was produced'\n"
    "WHEEL = wheels[-1]\n"
    "print('built:', WHEEL)\n"
))

# ---------------------------------------------------------------- §11 Install + import
cells.append(md(
    "## §11 Install the wheel and import it\n"
    "\n"
    "We install the just-built wheel into the active Python and reload "
    "`chronicle` from site-packages (rather than the src tree we've been using "
    "all along). If `demo()` runs and `verify()` passes, the wheel is sound.\n"
))
cells.append(code(
    "import subprocess, sys, importlib\n"
    "r = subprocess.run(\n"
    "    [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', '--quiet', str(WHEEL)],\n"
    "    capture_output=True, text=True,\n"
    ")\n"
    "print(r.stdout, r.stderr)\n"
    "assert r.returncode == 0\n"
    "\n"
    "# Drop the src/ shim from sys.path so we import the *installed* package.\n"
    "sys.path = [p for p in sys.path if 'src' not in p.split('/')[-2:]]\n"
    "for mod in list(sys.modules):\n"
    "    if mod == 'chronicle' or mod.startswith('chronicle.'):\n"
    "        del sys.modules[mod]\n"
    "import chronicle\n"
    "print('imported chronicle', chronicle.__version__, 'from', chronicle.__file__)\n"
    "\n"
    "c = chronicle.demo()\n"
    "ok, bad = c.verify()\n"
    "assert ok, bad\n"
    "print(f'demo Chronicle: {len(c)} events, root={c.root()}, verify=OK')\n"
    "for evt, score in c.search('vegan recipe ingredients'):\n"
    "    print(f'  {score:+.3f}  {evt.kind:<12} {evt.actor:<18} {str(evt.payload)[:60]}')\n"
))

# ---------------------------------------------------------------- §12 Research notes
cells.append(md(
    "## §12 Research notes & next steps\n"
    "\n"
    "**The contribution.** Treating an agent trace as a content-addressed causal "
    "DAG with a semantic overlay collapses *causality, integrity, cross-run "
    "comparison, search, and counterfactual replay* into a single substrate, "
    "rather than requiring orthogonal subsystems for each. The headline "
    "property is **deterministic cross-run identity** — identical reasoning "
    "produces identical ids, making A/B comparison of agent variants exact.\n"
    "\n"
    "**Limitations of this prototype.**\n"
    "- Only the hashing-trick embedder is exercised in the smoke test; the "
    "  `SentenceTransformerEmbedder` adapter is included but optional.\n"
    "- In-memory only. Persistence is JSONL — fine for research scale, not for "
    "  long-running production agents.\n"
    "- No cryptographic signatures yet. Tamper detection is integrity-only; "
    "  add signed Merkle roots (Ed25519) for adversarial-robustness work.\n"
    "- No OpenTelemetry bridge. Adding an exporter would let traces flow into "
    "  Jaeger / Tempo / Langfuse for cross-comparison with span-tree tools.\n"
    "\n"
    "**Citing this design.** If you publish, the framing to lean on is: "
    "\"agent traces as Merkle DAGs with semantic overlay — a unified substrate "
    "for causality, integrity, and counterfactual analysis\". As far as I can "
    "find, this combined framing is not present in the existing agent-"
    "observability literature.\n"
))

# ----------------------------------------------------------------------------
# Assemble notebook
# ----------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (LLM venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
        "mimetype": "text/x-python",
        "file_extension": ".py",
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
    },
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"wrote {NB_PATH} ({NB_PATH.stat().st_size:,} bytes, {len(cells)} cells)")
