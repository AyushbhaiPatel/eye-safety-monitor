"""Thresholds and settings for the Eye Safety Monitor."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CALIBRATION_FILE = BASE_DIR / "calibration.json"

FACE_MODEL = MODELS_DIR / "face_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

# Screen distance (recommended viewing distance is 50-70 cm)
MIN_DISTANCE_CM = 45            # closer than this raises an alert
WARN_MARGIN_CM = 5              # shown in orange just above the limit
CALIBRATION_DISTANCE_CM = 50    # how far you sit while calibrating
DEFAULT_CAMERA_FOV_DEG = 65     # rough guess used until you calibrate

# Blinks (healthy rate is 15-20 per minute, it drops while staring at screens)
DEFAULT_EAR_THRESHOLD = 0.21
MAX_BLINK_SECONDS = 0.6         # longer closures are not counted as blinks
LOW_BLINK_RATE = 10             # blinks per minute
BLINK_WINDOW_SECONDS = 60

# Posture
SLOUCH_RATIO = 0.80             # neck ratio under 80% of your upright value = slouching
SHOULDER_TILT_DEG = 10

# 20-20-20 rule: every 20 minutes, look 20 feet away for 20 seconds
WORK_SECONDS = 20 * 60
BREAK_SECONDS = 20

# Demo mode shortens the timers so everything can be shown in a few minutes
DEMO_WORK_SECONDS = 60
DEMO_BLINK_WINDOW_SECONDS = 20

# Alerts: a condition must last this long before the alert fires
HOLD_SECONDS = {"too_close": 2.0, "low_blink": 2.0, "slouch": 3.0, "tilt": 3.0, "break": 0.0}
ALERT_COOLDOWN_SECONDS = 8      # minimum gap between repeated beeps
