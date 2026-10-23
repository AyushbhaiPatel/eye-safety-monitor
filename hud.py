"""Everything drawn on top of the camera image."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GRAY, GREEN = (255, 255, 255), (190, 190, 190), (90, 210, 90)
ORANGE, RED, CYAN = (0, 165, 255), (70, 70, 235), (230, 200, 60)


def put(frame, text, org, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def shade(frame, x, y, w, h, alpha=0.6):
    """Semi-transparent dark rectangle so text stays readable on any background."""
    roi = frame[max(y, 0):y + h, max(x, 0):x + w]
    roi[:] = (roi * (1 - alpha) + np.array((25, 25, 25)) * alpha).astype(np.uint8)


def draw_panel(frame, rows, x=12, y=12, width=290, line_h=30):
    """rows: list of (label, value, color). Returns the y just below the panel."""
    shade(frame, x, y, width, 14 + line_h * len(rows))
    for i, (label, value, color) in enumerate(rows):
        put(frame, label, (x + 12, y + 28 + i * line_h), 0.52, GRAY)
        put(frame, value, (x + 132, y + 28 + i * line_h), 0.62, color, 2)
    return y + 14 + line_h * len(rows)


def draw_distance_gauge(frame, distance, min_distance, x=12, y=0, width=290, lo=20, hi=90):
    """Bar from `lo` to `hi` cm, red below the safe distance, with a marker at the current value."""
    shade(frame, x, y, width, 46)
    bar_x, bar_y, bar_w = x + 12, y + 22, width - 24
    to_x = lambda cm: int(bar_x + (min(max(cm, lo), hi) - lo) / (hi - lo) * bar_w)  # noqa: E731
    cv2.rectangle(frame, (bar_x, bar_y), (to_x(min_distance), bar_y + 10), RED, -1)
    cv2.rectangle(frame, (to_x(min_distance), bar_y), (bar_x + bar_w, bar_y + 10), GREEN, -1)
    put(frame, f"{lo}", (bar_x, bar_y - 6), 0.4, GRAY)
    put(frame, f"{hi} cm", (bar_x + bar_w - 40, bar_y - 6), 0.4, GRAY)
    if distance is not None:
        cv2.rectangle(frame, (to_x(distance) - 3, bar_y - 5), (to_x(distance) + 3, bar_y + 15), WHITE, -1)


def draw_face(frame, face):
    for eye in (face.right_eye, face.left_eye):
        cv2.polylines(frame, [eye.astype(np.int32)], True, CYAN, 1, cv2.LINE_AA)
    for iris in (face.right_iris, face.left_iris):
        radius = int(np.linalg.norm(iris[0] - iris[2]) / 2)
        cv2.circle(frame, tuple(iris.mean(axis=0).astype(int)), max(radius, 1), GREEN, 1, cv2.LINE_AA)


def draw_posture(frame, posture, color):
    left, right, nose = (tuple(int(v) for v in p) for p in
                         (posture.left_shoulder, posture.right_shoulder, posture.nose))
    mid = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
    cv2.line(frame, left, right, color, 2, cv2.LINE_AA)
    cv2.line(frame, nose, mid, color, 2, cv2.LINE_AA)
    for point in (left, right, nose):
        cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)


def draw_alerts(frame, messages):
    height, width = frame.shape[:2]
    y = height - 44
    for message in reversed(messages):
        (tw, th), _ = cv2.getTextSize(message, FONT, 0.75, 2)
        x = (width - tw) // 2
        cv2.rectangle(frame, (x - 14, y - th - 14), (x + tw + 14, y + 10), RED, -1)
        put(frame, message, (x, y), 0.75, WHITE, 2)
        y -= th + 34


def draw_center_message(frame, lines):
    height, width = frame.shape[:2]
    box_h = 40 + 36 * len(lines)
    top = height // 2 - box_h // 2
    shade(frame, 0, top, width, box_h, alpha=0.7)
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, FONT, 0.8, 2)
        put(frame, line, ((width - tw) // 2, top + 44 + i * 36), 0.8, WHITE, 2)


def draw_footer(frame, text):
    height, width = frame.shape[:2]
    shade(frame, 0, height - 28, width, 28)
    put(frame, text, (12, height - 9), 0.5, GRAY)
