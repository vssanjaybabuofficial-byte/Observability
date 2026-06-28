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
