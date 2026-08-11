import numpy as np

from scripts.direct_spline_split_specificity import descriptor_from_statistics, robust_distances


def test_transductive_descriptor_uses_query_features_but_not_query_labels():
    context_x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
    context_y = np.array([0, 0, 1, 1], dtype=np.int64)
    query_a = np.array([[4.0, 5.0], [5.0, 6.0]], dtype=np.float32)
    query_b = query_a + 100.0
    context_a, transductive_a = descriptor_from_statistics(context_x, context_y, query_a)
    context_b, transductive_b = descriptor_from_statistics(context_x, context_y, query_b)
    assert np.allclose(context_a, context_b)
    assert not np.allclose(transductive_a, transductive_b)


def test_robust_distances_are_symmetric_with_zero_diagonal():
    distances = robust_distances([np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0])])
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.all(distances[np.triu_indices(3, 1)] > 0)
