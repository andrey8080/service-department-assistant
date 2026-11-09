"""Векторный индекс для плоского RAG: косинусное сходство на numpy."""

import numpy as np


class VectorIndex:
    def __init__(self, ids: list[str], vectors: list[list[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError(f"идентификаторов {len(ids)}, векторов {len(vectors)}")
        self.ids = list(ids)
        matrix = np.asarray(vectors, dtype=np.float32)
        self._matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)

    def search(self, query: list[float], top_k: int = 4) -> list[str]:
        """Идентификаторы top_k ближайших векторов, от самого близкого."""
        vector = np.asarray(query, dtype=np.float32)
        scores = self._matrix @ (vector / max(float(np.linalg.norm(vector)), 1e-9))
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [self.ids[i] for i in order]
