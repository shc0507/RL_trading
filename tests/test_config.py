from __future__ import annotations

import unittest
import warnings
from datetime import date

from rl_trading.config import (
    TEST_END,
    TRAIN_START,
    walk_forward_splits,
)


def _d(s: str) -> date:
    return date.fromisoformat(s)


class WalkForwardSplitsTest(unittest.TestCase):
    def test_produces_five_folds_with_default_range(self) -> None:
        folds = walk_forward_splits()
        self.assertEqual(len(folds), 5)

    def test_test_window_shifts_by_test_years(self) -> None:
        folds = walk_forward_splits()
        for prev, curr in zip(folds[:-1], folds[1:]):
            prev_start = _d(prev["test"][0])
            curr_start = _d(curr["test"][0])
            self.assertEqual(curr_start.year - prev_start.year, 3)

    def test_ordering_within_each_fold(self) -> None:
        for fold in walk_forward_splits():
            train_s, train_e = map(_d, fold["train"])
            val_s, val_e = map(_d, fold["val"])
            test_s, test_e = map(_d, fold["test"])
            self.assertLess(train_s, train_e)
            self.assertLess(train_e, val_s)
            self.assertLessEqual(val_s, val_e)
            self.assertLess(val_e, test_s)
            self.assertLessEqual(test_s, test_e)

    def test_train_starts_at_data_start(self) -> None:
        for fold in walk_forward_splits():
            self.assertEqual(fold["train"][0], TRAIN_START)

    def test_val_is_approximately_target_fraction(self) -> None:
        """Val months should be ~10% of the combined train+val months."""
        folds = walk_forward_splits(val_frac=0.10)
        for fold in folds:
            train_s = _d(fold["train"][0])
            val_s = _d(fold["val"][0])
            val_e = _d(fold["val"][1])

            def months(a: date, b: date) -> int:
                return (b.year - a.year) * 12 + (b.month - a.month) + 1

            combined_months = months(train_s, val_e)
            val_months = months(val_s, val_e)
            frac = val_months / combined_months
            # Allow generous rounding tolerance: val is at least 1 month,
            # and month-boundary rounding for small windows can drift a bit.
            self.assertGreaterEqual(frac, 0.08)
            self.assertLessEqual(frac, 0.13)

    def test_custom_val_frac(self) -> None:
        folds_10 = walk_forward_splits(val_frac=0.10)
        folds_20 = walk_forward_splits(val_frac=0.20)
        for f10, f20 in zip(folds_10, folds_20):
            val10 = (_d(f10["val"][1]) - _d(f10["val"][0])).days
            val20 = (_d(f20["val"][1]) - _d(f20["val"][0])).days
            self.assertGreater(val20, val10)

    def test_last_test_end_within_data_range(self) -> None:
        folds = walk_forward_splits(data_end=TEST_END)
        self.assertLessEqual(_d(folds[-1]["test"][1]), _d(TEST_END))

    def test_train_val_gap_is_clean(self) -> None:
        """val_start should be the very next day / month after train_end."""
        for fold in walk_forward_splits():
            train_e = _d(fold["train"][1])
            val_s = _d(fold["val"][0])
            # train ends on last day of a month, val starts on first day of next.
            self.assertEqual(val_s.day, 1)
            # And the day after train_end equals val_start.
            from datetime import timedelta
            self.assertEqual(train_e + timedelta(days=1), val_s)

    def test_val_years_parameter_deprecated_but_accepted(self) -> None:
        # Default val_years=3 must NOT warn.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            walk_forward_splits()
            self.assertFalse(
                any(issubclass(wi.category, DeprecationWarning) for wi in w)
            )
        # A non-default val_years should warn but not raise.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            walk_forward_splits(val_years=5)
            self.assertTrue(
                any(issubclass(wi.category, DeprecationWarning) for wi in w)
            )

    def test_val_frac_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            walk_forward_splits(val_frac=0.0)
        with self.assertRaises(ValueError):
            walk_forward_splits(val_frac=1.0)


if __name__ == "__main__":
    unittest.main()
