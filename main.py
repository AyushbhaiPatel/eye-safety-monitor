"""Laptop Screen Distance & Eye Safety Monitor.

Run:  .venv/bin/python main.py          (add --demo for short timers during a presentation)
Keys: C calibrate | L landmarks on/off | R reset session | Q quit and show report
"""
import argparse
import json
import statistics
import sys
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

import config
import hud
from alerts import AlertManager
from detectors import Landmarker
from metrics import (BlinkCounter, BreakTimer, Smoother, default_focal_length, distance_cm,
                     focal_from_calibration, measure_face, measure_posture)
from report import SessionLogger

WINDOW = "Eye Safety Monitor"
FACE_AWAY_RESET_SECONDS = 3
CAMERA_TIMEOUT_SECONDS = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--demo", action="store_true", help="short timers for a live presentation")
    p.add_argument("--min-distance", type=float, default=config.MIN_DISTANCE_CM,
                   help="alert when closer than this, in cm")
    p.add_argument("--calib-distance", type=float, default=config.CALIBRATION_DISTANCE_CM,
                   help="how far from the camera you sit while calibrating, in cm")
    return p.parse_args()


def load_calibration():
    try:
        return json.loads(config.CALIBRATION_FILE.read_text())
    except (OSError, ValueError):
        return {"focal_px": None, "ear_threshold": config.DEFAULT_EAR_THRESHOLD, "neck_ratio": None}


class Calibrator:
    """Counts down 3 s, then collects 2 s of measurements while the user sits upright."""
    COUNTDOWN, COLLECT = 3.0, 2.0

    def __init__(self, now):
        self.start = now
        self.iris, self.ear, self.neck = [], [], []

    def collecting(self, now):
        return now - self.start >= self.COUNTDOWN

    def finished(self, now):
        return now - self.start >= self.COUNTDOWN + self.COLLECT

    def add(self, now, face, posture):
        if self.collecting(now):
            if face:
                self.iris.append(face.iris_px)
                self.ear.append(face.ear)
            if posture:
                self.neck.append(posture.neck_ratio)

    def overlay(self, frame, now, distance):
        if self.collecting(now):
            hud.draw_center_message(frame, ["Hold still... measuring"])
        else:
            left = int(self.COUNTDOWN - (now - self.start)) + 1
            hud.draw_center_message(frame, [f"Sit upright, about {distance:.0f} cm from the camera",
                                            f"Look straight at the camera   {left}"])

    def result(self, distance, previous):
        """Medians ignore the odd blink or bad frame. None if the face was not seen enough."""
        if len(self.iris) < 5:
            return None
        return {
            "focal_px": focal_from_calibration(statistics.median(self.iris), distance),
            "ear_threshold": round(min(max(statistics.median(self.ear) * 0.7, 0.15), 0.30), 3),
            "neck_ratio": statistics.median(self.neck) if len(self.neck) >= 5 else previous["neck_ratio"],
        }


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def main():
    args = parse_args()
    work_seconds = config.DEMO_WORK_SECONDS if args.demo else config.WORK_SECONDS
    blink_window = config.DEMO_BLINK_WINDOW_SECONDS if args.demo else config.BLINK_WINDOW_SECONDS

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Could not open the camera. Allow camera access for your terminal app in "
              "System Settings > Privacy & Security > Camera, and open the MacBook lid.")
        return 1

    calib = load_calibration()
    face_tracker, pose_tracker = Landmarker("face"), Landmarker("pose")
    distance_smoother, neck_smoother = Smoother(0.25), Smoother(0.3)

    def new_session():
        return (BlinkCounter(calib["ear_threshold"], blink_window),
                BreakTimer(work_seconds, config.BREAK_SECONDS),
                AlertManager(config.HOLD_SECONDS, config.ALERT_COOLDOWN_SECONDS),
                SessionLogger())

    blinks, breaks, alerts, logger = new_session()
    calibrator, show_landmarks, window_seen = None, True, False
    face_lost_since, failing_since = None, None
    fps, last_frame_time, last_ts = 0.0, None, -1
    start = time.monotonic()
    toast, toast_until = ("Press C to calibrate for accurate distance", start + 6) if not calib["focal_px"] else ("", 0)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:  # webcams send a few empty frames while starting up
            failing_since = failing_since or time.monotonic()
            if time.monotonic() - failing_since > CAMERA_TIMEOUT_SECONDS:
                print("The camera sent no images. Open the MacBook lid and make sure no other app is using it.")
                break
            time.sleep(0.05)
            continue
        failing_since = None
        frame = cv2.flip(frame, 1)  # mirror view feels natural on a webcam
        now = time.monotonic()
        height, width = frame.shape[:2]
        if last_frame_time is not None:
            fps = 0.9 * fps + 0.1 / max(now - last_frame_time, 1e-6)
        last_frame_time = now

        # --- detection -------------------------------------------------------
        last_ts = max(int((now - start) * 1000), last_ts + 1)  # must keep increasing
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_landmarks = face_tracker.detect(image, last_ts)
        pose_landmarks = pose_tracker.detect(image, last_ts)
        face = measure_face(face_landmarks, width, height) if face_landmarks else None
        posture = measure_posture(pose_landmarks, width, height) if pose_landmarks else None

        # --- measurements ----------------------------------------------------
        focal = calib["focal_px"] or default_focal_length(width)
        if face:
            distance = distance_smoother.update(distance_cm(face.iris_px, focal))
            blinks.update(face.ear, now)
            face_lost_since = None
        else:
            distance = None
            distance_smoother.reset()
            face_lost_since = face_lost_since or now
            if now - face_lost_since > FACE_AWAY_RESET_SECONDS:
                blinks.restart_window(now)
        neck = neck_smoother.update(posture.neck_ratio) if posture else neck_smoother.reset()
        blink_rate = blinks.rate(now)
        breaks.update(face is not None, now)

        too_close = distance is not None and distance < args.min_distance
        low_blink = face is not None and blink_rate is not None and blink_rate < config.LOW_BLINK_RATE
        slouching = (calib["neck_ratio"] is not None and neck is not None
                     and neck < calib["neck_ratio"] * config.SLOUCH_RATIO)
        tilted = posture is not None and posture.shoulder_tilt_deg > config.SHOULDER_TILT_DEG

        # --- alerts (paused while calibrating) --------------------------------
        live = calibrator is None
        alerts.update("too_close", live and too_close, now, f"Too close! Move back to at least {args.min_distance:.0f} cm")
        alerts.update("low_blink", live and low_blink, now, "Blink more - your eyes are drying out")
        alerts.update("slouch", live and slouching, now, "Sit up straight - you are slouching")
        alerts.update("tilt", live and tilted and not slouching, now, "Level your shoulders")
        alerts.update("break", live and breaks.break_due, now, "20-20-20: look 20 ft away for 20 seconds")
        logger.record(now, face_present=face is not None, distance_cm=distance, blink_rate=blink_rate,
                      running_blink_rate=blinks.running_rate(now),
                      neck_ratio=neck, slouching=slouching, too_close=too_close, low_blink=low_blink)

        # --- calibration -----------------------------------------------------
        if calibrator:
            calibrator.add(now, face, posture)
            if calibrator.finished(now):
                result = calibrator.result(args.calib_distance, calib)
                if result:
                    calib = result
                    config.CALIBRATION_FILE.write_text(json.dumps(calib, indent=2))
                    blinks.threshold = calib["ear_threshold"]
                toast = "Calibrated!" if result else "Calibration failed - keep your face in view and try again"
                toast_until, calibrator = now + 3, None

        # --- drawing ---------------------------------------------------------
        if show_landmarks and face:
            hud.draw_face(frame, face)
        if show_landmarks and posture:
            hud.draw_posture(frame, posture, hud.RED if slouching else hud.GREEN)
        if distance is None:
            dist_text, dist_color = "--", hud.GRAY
        else:
            dist_text = f"{distance:.0f} cm" + ("" if calib["focal_px"] else " (est.)")
            dist_color = (hud.RED if too_close else hud.ORANGE
                          if distance < args.min_distance + config.WARN_MARGIN_CM else hud.GREEN)
        posture_text, posture_color = (
            ("not visible", hud.GRAY) if posture is None else
            ("calibrate (C)", hud.ORANGE) if calib["neck_ratio"] is None else
            ("SLOUCHING", hud.RED) if slouching else
            ("tilted", hud.ORANGE) if tilted else ("good", hud.GREEN))
        rate_text, rate_color = (("measuring...", hud.GRAY) if blink_rate is None else
                                 (f"{blink_rate:.0f} / min", hud.RED if low_blink else hud.GREEN))
        panel_bottom = hud.draw_panel(frame, [
            ("Distance", dist_text, dist_color),
            ("Blinks", str(blinks.total), hud.WHITE),
            ("Blink rate", rate_text, rate_color),
            ("Posture", posture_text, posture_color),
            ("Screen time", mmss(breaks.screen_time), hud.WHITE),
            ("Break in", "NOW" if breaks.break_due else mmss(breaks.time_to_break),
             hud.RED if breaks.break_due else hud.WHITE),
            ("FPS", f"{fps:.0f}", hud.GRAY),
        ])
        hud.draw_distance_gauge(frame, distance, args.min_distance, y=panel_bottom + 6)
        if face is None and calibrator is None:
            hud.put(frame, "No face detected", (width - 230, 36), 0.7, hud.ORANGE, 2)
        hud.draw_alerts(frame, alerts.active_messages)
        if calibrator:
            calibrator.overlay(frame, now, args.calib_distance)
        elif now < toast_until:
            hud.draw_center_message(frame, [toast])
        hud.draw_footer(frame, "C calibrate   L landmarks   R reset   Q quit & report"
                        + ("  |  DEMO MODE" if args.demo else ""))
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            calibrator = Calibrator(now)
        elif key == ord("l"):
            show_landmarks = not show_landmarks
        elif key == ord("r"):
            blinks, breaks, alerts, logger = new_session()
            toast, toast_until = "Session reset", now + 2
        # Stop when the window is closed with the mouse. Only trust the flag once the
        # window has been reported visible, since some macOS builds never set it.
        visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        if window_seen and not visible:
            break
        window_seen = window_seen or visible

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    face_tracker.close()
    pose_tracker.close()

    if logger.rows:
        summary = logger.summary(blinks.total, alerts.counts, breaks.breaks_taken)
        slouch_limit = calib["neck_ratio"] * config.SLOUCH_RATIO if calib["neck_ratio"] else None
        path = logger.save(config.REPORTS_DIR, summary, args.min_distance, slouch_limit)
        print(json.dumps(summary, indent=2))
        print(f"Report saved to {path}")
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
