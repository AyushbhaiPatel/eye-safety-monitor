"""Run: .venv/bin/python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alerts import AlertManager  # noqa: E402
from metrics import (BlinkCounter, BreakTimer, distance_cm, eye_aspect_ratio,  # noqa: E402
                     focal_from_calibration, measure_posture)


class DistanceTest(unittest.TestCase):
    def test_calibration_round_trip(self):
        focal = focal_from_calibration(iris_px=30, known_distance_cm=50)
        self.assertAlmostEqual(distance_cm(30, focal), 50)

    def test_smaller_iris_means_farther(self):
        focal = focal_from_calibration(30, 50)
        self.assertAlmostEqual(distance_cm(15, focal), 100)
        self.assertIsNone(distance_cm(0, focal))


class EarTest(unittest.TestCase):
    def test_open_vs_closed(self):
        open_eye = [(0, 0), (10, -5), (20, -5), (30, 0), (20, 5), (10, 5)]
        closed_eye = [(0, 0), (10, -1), (20, -1), (30, 0), (20, 1), (10, 1)]
        self.assertAlmostEqual(eye_aspect_ratio(open_eye), 10 / 30)
        self.assertLess(eye_aspect_ratio(closed_eye), 0.1)


class BlinkCounterTest(unittest.TestCase):
    def feed(self, counter, samples, fps=30, t0=0.0):
        t = t0
        for ear in samples:
            counter.update(ear, t)
            t += 1 / fps
        return t

    def test_counts_short_closures(self):
        counter = BlinkCounter(threshold=0.2, window=10)
        self.feed(counter, ([0.3] * 30 + [0.1] * 4) * 5 + [0.3] * 10)
        self.assertEqual(counter.total, 5)

    def test_ignores_long_closure(self):
        counter = BlinkCounter(threshold=0.2, window=10)
        self.feed(counter, [0.3] * 10 + [0.1] * 60 + [0.3] * 10)  # eyes shut for 2 s
        self.assertEqual(counter.total, 0)

    def test_ignores_one_frame_dip(self):
        counter = BlinkCounter(threshold=0.2, window=10)
        self.feed(counter, [0.3] * 30 + [0.1] + [0.3] * 30)
        self.assertEqual(counter.total, 0)

    def test_ignores_shallow_dip(self):
        counter = BlinkCounter(threshold=0.2, window=10)
        self.feed(counter, [0.3] * 60 + [0.21] * 4 + [0.3] * 30)  # glance down, eye 70% open
        self.assertEqual(counter.total, 0)

    def test_rate_needs_full_window(self):
        counter = BlinkCounter(threshold=0.2, window=10)
        end = self.feed(counter, ([0.3] * 56 + [0.1] * 4) * 3)  # 3 blinks in 6 s
        self.assertIsNone(counter.rate(end))
        end = self.feed(counter, ([0.3] * 56 + [0.1] * 4 + [0.3]) * 3, t0=end)
        self.assertAlmostEqual(counter.rate(end), 5 * 6, delta=6)

    def test_running_rate_is_available_before_the_window_fills(self):
        counter = BlinkCounter(threshold=0.2, window=30)
        self.assertIsNone(counter.running_rate(0.0))
        counter.window_start = 0.0
        counter.times.extend([1.0, 2.0])
        self.assertIsNone(counter.rate(5.0))
        self.assertEqual(counter.running_rate(5.0), 12)    # 10 s floor applies
        self.assertEqual(counter.running_rate(20.0), 6)    # 2 blinks in 20 s
        self.assertEqual(counter.running_rate(45.0), 0)    # both blinks left the window


class BreakTimerTest(unittest.TestCase):
    def test_break_due_then_taken(self):
        timer = BreakTimer(work_seconds=5, break_seconds=2)
        t = 0.0
        while t <= 6:
            timer.update(True, t)
            t += 0.5
        self.assertTrue(timer.break_due)
        while t <= 9:
            timer.update(False, t)
            t += 0.5
        self.assertFalse(timer.break_due)
        self.assertEqual(timer.breaks_taken, 1)


class PostureTest(unittest.TestCase):
    @staticmethod
    def pose(nose_y, visibility=0.9):
        lm = [SimpleNamespace(x=0.5, y=0.5, visibility=visibility) for _ in range(33)]
        lm[0] = SimpleNamespace(x=0.5, y=nose_y, visibility=0.9)
        lm[11] = SimpleNamespace(x=0.7, y=0.8, visibility=visibility)
        lm[12] = SimpleNamespace(x=0.3, y=0.8, visibility=visibility)
        return lm

    def test_slouch_lowers_ratio(self):
        upright = measure_posture(self.pose(0.4), 100, 100)
        slouched = measure_posture(self.pose(0.6), 100, 100)
        self.assertAlmostEqual(upright.neck_ratio, 1.0)
        self.assertLess(slouched.neck_ratio, upright.neck_ratio * 0.8)
        self.assertAlmostEqual(upright.shoulder_tilt_deg, 0)

    def test_hidden_shoulders(self):
        self.assertIsNone(measure_posture(self.pose(0.4, visibility=0.1), 100, 100))


class AlertManagerTest(unittest.TestCase):
    def test_hold_then_fire_once_per_episode(self):
        alerts = AlertManager({"x": 2.0}, cooldown=100, sound=False)
        self.assertFalse(alerts.update("x", True, 0.0, "msg"))
        self.assertFalse(alerts.update("x", True, 1.0, "msg"))
        self.assertTrue(alerts.update("x", True, 2.5, "msg"))
        self.assertTrue(alerts.update("x", True, 3.0, "msg"))
        self.assertEqual(alerts.counts["x"], 1)
        alerts.update("x", False, 4.0, "msg")
        self.assertEqual(alerts.active_messages, [])


if __name__ == "__main__":
    unittest.main()
