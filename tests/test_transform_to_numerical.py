import numpy as np
import pandas as pd
import pytest

from tabicl._sklearn.preprocessing import TransformToNumerical


@pytest.mark.parametrize("dataframe", [False, True])
@pytest.mark.parametrize("empty_position", [None, 0, 1, 2])
def test_typed_parts_follow_context_imputer_column_selection(dataframe, empty_position):
    context = np.array([[1., np.nan], [3., 4.], [np.nan, 6.]])
    query = np.array([[np.nan, 8.], [7., np.nan]])
    if empty_position is not None:
        context = np.insert(context, empty_position, np.nan, axis=1)
        # Query observations must not restore a feature absent from context.
        query = np.insert(query, empty_position, 999., axis=1)
    if dataframe:
        columns = [f"x{i}" for i in range(context.shape[1])]
        context = pd.DataFrame(context, columns=columns)
        query = pd.DataFrame(query, columns=columns)
        context.insert(1, "category", ["a", "b", "a"])
        query.insert(1, "category", ["new", "a"])

    encoder = TransformToNumerical().fit(context)
    context_parts = encoder.transform_parts(context)
    query_parts = encoder.transform_parts(query)
    np.testing.assert_array_equal(context_parts.numerical, [[1., 5.], [3., 4.], [2., 6.]])
    np.testing.assert_array_equal(query_parts.numerical, [[2., 8.], [7., 5.]])
    np.testing.assert_array_equal(context_parts.numerical_missing,
                                  [[False, True], [False, False], [True, False]])
    np.testing.assert_array_equal(query_parts.numerical_missing, [[True, False], [False, True]])
    for raw, parts in ((context, context_parts), (query, query_parts)):
        np.testing.assert_array_equal(np.concatenate((parts.categorical, parts.numerical), axis=1),
                                      encoder.transform(raw))
    if dataframe:
        np.testing.assert_array_equal(query_parts.categorical, [[-1], [0]])
    else:
        assert query_parts.categorical.shape == (2, 0)


@pytest.mark.parametrize("dataframe", [False, True])
def test_all_numerical_features_missing_in_context(dataframe):
    context = np.full((3, 2), np.nan)
    query = np.array([[10., 20.]])
    if dataframe:
        context, query = pd.DataFrame(context, columns=["x0", "x1"]), pd.DataFrame(query, columns=["x0", "x1"])
        context["category"], query["category"] = ["a", "b", "a"], ["a"]
    encoder = TransformToNumerical().fit(context)
    for raw in (context, query):
        parts = encoder.transform_parts(raw)
        assert parts.numerical.shape == parts.numerical_missing.shape == (len(raw), 0)
        np.testing.assert_array_equal(parts.categorical, encoder.transform(raw))


def test_categorical_only_frame_does_not_inspect_unfitted_numeric_imputer():
    frame = pd.DataFrame({"category": ["a", "b", "a"]})
    parts = TransformToNumerical().fit_transform_parts(frame)
    assert parts.numerical.shape == parts.numerical_missing.shape == (3, 0)
    np.testing.assert_array_equal(parts.categorical, [[0], [1], [0]])


def test_refitting_restores_previously_empty_feature():
    encoder = TransformToNumerical().fit(np.array([[np.nan, 1.], [np.nan, 3.]]))
    encoder.fit(np.array([[2., 1.], [4., 3.]]))
    parts = encoder.transform_parts(np.array([[np.nan, 5.]]))
    np.testing.assert_array_equal(parts.numerical, [[3., 5.]])
    np.testing.assert_array_equal(parts.numerical_missing, [[True, False]])
