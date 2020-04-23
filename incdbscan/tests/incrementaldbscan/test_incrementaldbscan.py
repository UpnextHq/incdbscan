import numpy as np

from incdbscan import IncrementalDBSCANWarning
from utils import (
    CLUSTER_LABEL_NOISE,
    CLUSTER_LABEL_UNKNOWN_OBJECT,
    delete_object_and_assert_error,
    delete_object_and_assert_no_warning,
    delete_object_and_assert_warning,
    insert_object_and_assert_error,
    get_label_and_assert_error,
    get_label_and_assert_no_warning,
    get_label_and_assert_warning
)


def test_error_when_input_is_non_numeric(incdbscan3):
    inputs_not_welcomed = np.array([
        [1, 2, 'x'],
        [1, 2, None],
        [1, 2, np.nan],
        [1, 2, np.inf],
    ])

    for i in range(len(inputs_not_welcomed)):
        input_ = inputs_not_welcomed[[i]]

        insert_object_and_assert_error(incdbscan3, input_, ValueError)
        delete_object_and_assert_error(incdbscan3, input_, ValueError)
        get_label_and_assert_error(incdbscan3, input_, ValueError)


def test_handling_of_same_object_with_different_dtype(incdbscan3):
    object_as_int = np.array([[1, 2, 3]])
    object_as_float = np.array([[1., 2., 3.]])

    incdbscan3.insert(object_as_int)

    assert incdbscan3.get_cluster_labels(object_as_int) == \
        incdbscan3.get_cluster_labels(object_as_float)

    delete_object_and_assert_no_warning(incdbscan3, object_as_float)


def test_no_warning_when_a_known_object_is_deleted(
        incdbscan3,
        point_at_origin):

    incdbscan3.insert(point_at_origin)
    delete_object_and_assert_no_warning(incdbscan3, point_at_origin)

    incdbscan3.insert(point_at_origin)
    incdbscan3.insert(point_at_origin)
    delete_object_and_assert_no_warning(incdbscan3, point_at_origin)
    delete_object_and_assert_no_warning(incdbscan3, point_at_origin)


def test_warning_when_unknown_object_is_deleted(
        incdbscan3,
        point_at_origin,
        object_far_away):

    delete_object_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)

    incdbscan3.insert(point_at_origin)

    incdbscan3.delete(point_at_origin)

    delete_object_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)


def test_no_warning_when_cluster_label_is_gotten_for_known_object(
        incdbscan3,
        point_at_origin):

    expected_label = np.array([CLUSTER_LABEL_NOISE])

    incdbscan3.insert(point_at_origin)
    label = get_label_and_assert_no_warning(incdbscan3, point_at_origin)
    assert label == expected_label

    incdbscan3.insert(point_at_origin)
    incdbscan3.delete(point_at_origin)
    label = get_label_and_assert_no_warning(incdbscan3, point_at_origin)
    assert label == expected_label


def test_warning_when_cluster_label_is_gotten_for_unknown_object(
        incdbscan3,
        point_at_origin,
        object_far_away):

    expected_label_for_unknown = np.array([CLUSTER_LABEL_UNKNOWN_OBJECT])

    label = get_label_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)
    assert label == expected_label_for_unknown

    incdbscan3.insert(point_at_origin)
    incdbscan3.delete(point_at_origin)

    label = get_label_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)
    assert label == expected_label_for_unknown
