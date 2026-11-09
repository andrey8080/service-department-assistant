import pytest

from assistant.infra.vector_index import VectorIndex


def test_search_returns_nearest_by_cosine():
    index = VectorIndex(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    assert index.search([2.0, 0.1], top_k=2) == ["a", "c"]


def test_vector_length_does_not_matter():
    index = VectorIndex(["short", "long"], [[0.1, 0.0], [0.0, 50.0]])
    assert index.search([1.0, 0.2], top_k=1) == ["short"]


def test_top_k_larger_than_index_returns_everything():
    assert len(VectorIndex(["a"], [[1.0, 0.0]]).search([1.0, 0.0], top_k=5)) == 1


def test_mismatched_sizes_are_rejected():
    with pytest.raises(ValueError):
        VectorIndex(["a", "b"], [[1.0, 0.0]])
