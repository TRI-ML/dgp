# Copyright 2019-2020 Toyota Research Institute. All rights reserved.
import unittest

from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)

__all__ = [
    "assert_between",
    "assert_equal",
    "assert_not_equal",
    "assert_raises",
    "assert_true",
    "assert_false",
    "assert_allclose",
    "assert_almost_equal",
    "assert_array_equal",
    "assert_array_almost_equal",
    "assert_array_less",
    "assert_less",
    "assert_less_equal",
    "assert_greater",
    "assert_greater_equal",
    "assert_approx_equal",
    "SkipTest",
]

_dummy = unittest.TestCase('__init__')
assert_equal = _dummy.assertEqual
assert_not_equal = _dummy.assertNotEqual
assert_true = _dummy.assertTrue
assert_false = _dummy.assertFalse
assert_raises = _dummy.assertRaises
SkipTest = unittest.case.SkipTest
assert_dict_equal = _dummy.assertDictEqual
assert_in = _dummy.assertIn
assert_not_in = _dummy.assertNotIn
assert_less = _dummy.assertLess
assert_greater = _dummy.assertGreater
assert_less_equal = _dummy.assertLessEqual
assert_greater_equal = _dummy.assertGreaterEqual


def assert_between(value, low, high, low_inclusive=True, high_inclusive=True):
    """
    Assert ``value`` is inside the specified range.
    Parameters
    ----------
    value : comparable
        Value to be asserted.

    low : comparable
        Range lower bound.

    high : comparable
        Range upper bound.

    low_inclusive : bool, optional
        Allow case when value == low. Default: True.

    high_inclusive : bool, optional
        Allow case when value == high. Default: True.
    """
    if low_inclusive:
        assert_greater_equal(value, low)
    else:
        assert_greater(value, low)

    if high_inclusive:
        assert_less_equal(value, high)
    else:
        assert_less(value, high)
