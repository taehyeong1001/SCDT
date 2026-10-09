from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import runtime as rt
import numpy as np


class CriteriaTests(unittest.TestCase):
    def test_crossings(self):
        self.assertAlmostEqual(rt.crossing([0, 1], [1, 0]), .5)
        self.assertAlmostEqual(rt.crossing([0, 1], [0, 1], increasing=True), .5)
        self.assertEqual(rt.crossing([0, 1, 2], [1, .5, 0]), 1)
        with self.assertRaises(ValueError):
            rt.crossing([0, 1], [1, 1])

    def test_food_tail_only_and_finite(self):
        x = np.r_[np.zeros(500), np.full(500, .1)]
        self.assertTrue(rt.food_status(x)['survived'])
        x[0] = np.nan
        self.assertFalse(rt.food_status(x)['survived'])

    def test_kuramoto_dwell_not_mean(self):
        x = np.zeros((2, 10000))
        x[0, -2400:] = .5
        x[1, -2399:] = .9
        _, flags = rt.kuramoto_status(x)
        np.testing.assert_array_equal(flags, [True, False])

    def test_power_any_threshold_event_excludes_later_recovery(self):
        module = SimpleNamespace(calculate_autocorrelation=lambda _: .9)
        x = np.full(2000, .8)
        self.assertTrue(rt.power_status(module, x)['survived'])
        x[20] = .30
        self.assertFalse(rt.power_status(module, x)['survived'])
        x[20] = np.nan
        self.assertFalse(rt.power_status(module, x)['survived'])

    def test_window_protocol(self):
        sequence = np.arange(2301)[:, None]
        windows = rt.kuramoto_windows(sequence)
        self.assertEqual(windows.shape, (20, 60, 1))
        self.assertEqual(windows[0, 0, 0], round(2241 / 21))
        self.assertEqual(windows[-1, 0, 0], round(2241 * 20 / 21))

    def test_target_uses_inverse_of_same_stat_fit(self):
        loaded = rt.kuramoto_load()
        _, _, training, _ = loaded
        sigma = .231408064581645
        _, source, out, _ = rt.kuramoto_target(loaded, sigma)
        self.assertEqual(source, 4)
        self.assertAlmostEqual(np.polyval(np.polyfit(training.stat_1, training.stat_2, 1), out), sigma)


if __name__ == '__main__':
    unittest.main()
