"""Bi-encoder embedding interface (spec section 4) and a deterministic fake.

Contract file: owned by the orchestrator.
"""

import hashlib
import re
import threading
from typing import Protocol

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """L2-normalised float32 array of shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """L2-normalised float32 vector of shape (dim,), with the model's query prefix."""
        ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int = 384, revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.dim = dim
        self._model = SentenceTransformer(model_name, device="cpu", revision=revision)
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        # One lock for every caller (retrieval and verifier share this object). HF fast
        # tokenizers are not re-entrant across threads ("Already borrowed").
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self._model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32
            ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            return self._model.encode(
                self._query_prefix + text, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)

    def count_tokens(self, text: str) -> int:
        with self._lock:
            return len(self._model.tokenizer.encode(text, add_special_tokens=False))


_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag-of-words. Texts sharing words get higher cosine. No model download."""

    def __init__(self, dim: int = 384, model_name: str = "fake-embedder") -> None:
        self.model_name = model_name
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.sha1(w.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())
