"""Computer-vision measurements: eye aspect ratio, iris size, screen distance, posture."""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

import config

# MediaPipe face landmark indices (478-point model, includes the iris)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # p1..p6 for the EAR formula
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS = [469, 470, 471, 472]           # right, top, left, bottom edge points
LEFT_IRIS = [474, 475, 476, 477]
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12  # pose landmark indices
IRIS_DIAMETER_MM = 11.7  # almost the same for every adult, so it works as a ruler


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _pixels(landmarks, indices, width, height):
    return np.array([(landmarks[i].x * width, landmarks[i].y * height) for i in indices])


def eye_aspect_ratio(eye):
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). Falls towards 0 when the eye closes."""
    p1, p2, p3, p4, p5, p6 = eye
    horizontal = _dist(p1, p4)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal) if horizontal else 0.0


def default_focal_length(frame_width, fov_deg=config.DEFAULT_CAMERA_FOV_DEG):
    return (frame_width / 2) / math.tan(math.radians(fov_deg) / 2)


def focal_from_calibration(iris_px, known_distance_cm):
    # Pinhole camera model: iris_px = focal_px * iris_mm / distance_mm
    return iris_px * known_distance_cm * 10 / IRIS_DIAMETER_MM


def distance_cm(iris_px, focal_px):
    return focal_px * IRIS_DIAMETER_MM / iris_px / 10 if iris_px > 0 else None


@dataclass
class FaceMeasurement:
    ear: float
    iris_px: float
    right_eye: np.ndarray
    left_eye: np.ndarray
    right_iris: np.ndarray
    left_iris: np.ndarray


def measure_face(landmarks, width, height):
    right_eye, left_eye, right_iris, left_iris = (
        _pixels(landmarks, idx, width, height) for idx in (RIGHT_EYE, LEFT_EYE, RIGHT_IRIS, LEFT_IRIS))
    ear = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2
    # Horizontal iris width, because the eyelids cover the top and bottom
    iris_px = (_dist(right_iris[0], right_iris[2]) + _dist(left_iris[0], left_iris[2])) / 2
    return FaceMeasurement(ear, iris_px, right_eye, left_eye, right_iris, left_iris)


@dataclass
class PostureMeasurement:
    neck_ratio: float          # vertical nose-to-shoulder gap divided by shoulder width
    shoulder_tilt_deg: float
    nose: tuple
    left_shoulder: tuple
    right_shoulder: tuple


def measure_posture(landmarks, width, height, min_visibility=0.5):
    """Returns None when the shoulders are not clearly visible."""
    nose, left, right = (landmarks[i] for i in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER))
    if min(left.visibility or 0, right.visibility or 0) < min_visibility:
        return None
    nose_px, left_px, right_px = ((p.x * width, p.y * height) for p in (nose, left, right))
    shoulder_width = _dist(left_px, right_px)
    if shoulder_width < 1:
        return None
    # Slouching drops the head towards the shoulders, so this ratio shrinks.
    # Dividing by shoulder width keeps it the same whether you sit near or far.
    neck_ratio = ((left_px[1] + right_px[1]) / 2 - nose_px[1]) / shoulder_width
    tilt = math.degrees(math.atan2(abs(left_px[1] - right_px[1]), abs(left_px[0] - right_px[0])))
    return PostureMeasurement(neck_ratio, tilt, nose_px, left_px, right_px)


class Smoother:
    """Exponential moving average to stop numbers jittering frame to frame."""

    def __init__(self, alpha=0.3):
        self.alpha, self.value = alpha, None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class BlinkCounter:
    """Counts blinks from EAR samples and reports blinks per minute.

    Eye shapes differ between people, so the threshold adapts: a fraction of this person's
    open-eye EAR over the last few seconds. `threshold` is only used until enough samples exist.
    """
    CLOSE_FRACTION = 0.60     # eye counts as closed below 60% of the open-eye EAR
    OPEN_FRACTION = 0.75      # and open again above 75% (the gap stops flickering)
    HISTORY_SECONDS = 4.0
    MIN_BLINK_SECONDS = 0.04  # a real blink spans at least 2 frames; 1-frame dips are noise
    RUNNING_FLOOR_SECONDS = 10.0  # stops one early blink reading as a huge running rate

    def __init__(self, threshold=config.DEFAULT_EAR_THRESHOLD, window=config.BLINK_WINDOW_SECONDS):
        self.threshold, self.window, self.total = threshold, window, 0
        self.times, self._history = deque(), deque()
        self.closed_since = self.window_start = None

    def thresholds(self, now):
        """(close below, open above). Uses the 80th percentile of recent EAR as the open-eye value."""
        while self._history and now - self._history[0][0] > self.HISTORY_SECONDS:
            self._history.popleft()
        if len(self._history) < 15:
            return self.threshold, self.threshold
        open_ear = float(np.percentile([e for _, e in self._history], 80))
        return open_ear * self.CLOSE_FRACTION, open_ear * self.OPEN_FRACTION

    def update(self, ear, now):
        """Feed one EAR sample. Returns True when a blink has just finished."""
        self.window_start = now if self.window_start is None else self.window_start
        close_below, open_above = self.thresholds(now)
        self._history.append((now, ear))
        if self.closed_since is None:
            self.closed_since = now if ear < close_below else None
            return False
        if ear < open_above:
            return False
        closed_for, self.closed_since = now - self.closed_since, None
        if not self.MIN_BLINK_SECONDS <= closed_for <= config.MAX_BLINK_SECONDS:
            return False
        self.total += 1
        self.times.append(now)
        return True

    def restart_window(self, now):
        """Start measuring the rate again, e.g. after the user was away from the screen."""
        self.times.clear()
        self.closed_since, self.window_start = None, now

    def rate(self, now):
        """Blinks per minute over the last `window` seconds, or None until there is enough data."""
        if self.window_start is None or now - self.window_start < self.window:
            return None
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()
        return len(self.times) * 60 / self.window

    def running_rate(self, now):
        """Estimate available from the first second, for the report chart.

        Uses the time measured so far until a full window exists. Alerts and the score keep
        using rate(), because this estimate is unreliable early on.
        """
        if self.window_start is None:
            return None
        seconds = min(max(now - self.window_start, self.RUNNING_FLOOR_SECONDS), self.window)
        return sum(1 for t in self.times if now - t <= self.window) * 60 / seconds


class BreakTimer:
    """20-20-20 rule: after enough screen time, ask the user to look away for a while."""

    def __init__(self, work_seconds=config.WORK_SECONDS, break_seconds=config.BREAK_SECONDS):
        self.work_seconds, self.break_seconds = work_seconds, break_seconds
        self.screen_time = self.away_time = 0.0
        self.breaks_taken, self._last = 0, None

    def update(self, face_present, now):
        dt = 0.0 if self._last is None else min(now - self._last, 1.0)
        self._last = now
        if face_present:
            self.away_time = 0.0
            self.screen_time += dt
        else:
            self.away_time += dt
            if self.away_time >= self.break_seconds and self.screen_time > 0:
                self.breaks_taken += self.break_due
                self.screen_time = 0.0

    @property
    def break_due(self):
        return self.screen_time >= self.work_seconds

    @property
    def time_to_break(self):
        return max(0.0, self.work_seconds - self.screen_time)
