# Laptop Screen Distance & Eye Safety Monitor

A webcam app that watches how you use your laptop and warns you when you:

- **sit too close** to the screen (distance measured from your iris size)
- **blink too little** (blink counting with the Eye Aspect Ratio)
- **slouch** or tilt your shoulders (body pose landmarks)
- **skip breaks** (20-20-20 rule: every 20 min, look 20 ft away for 20 s)

When you quit, it saves a session report with charts and an eye-care score.

Everything runs on the laptop CPU, with no GPU, dataset or training.

## Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The two MediaPipe model files are in `models/`. If they are missing, the app downloads them on first run.

## Run

```bash
.venv/bin/python main.py
```

For a presentation, use demo mode. The break reminder comes after 1 minute instead of 20, and the blink rate is measured over 20 seconds instead of 60:

```bash
.venv/bin/python main.py --demo
```

The first time, macOS asks to let your terminal use the camera. Click **Allow** and run it again.

| Key | Action |
|---|---|
| `C` | Calibrate: sit upright about 50 cm from the camera, look at it, hold still |
| `L` | Show / hide the landmarks |
| `R` | Reset the session |
| `Q` / `Esc` | Quit and show the report |

Calibrate once. The values are saved to `calibration.json` and reused next time.

Other options: `--min-distance 50`, `--calib-distance 60`, `--camera 1`.

## Demo script (all done sitting down)

1. Start with `--demo` and press **C** to calibrate.
2. **Lean towards the screen.** The distance turns red and the "Too close!" alert beeps.
3. **Stare without blinking** for about 20 s. The low-blink alert appears.
4. **Slouch down in the chair.** Posture shows SLOUCHING and the alert fires.
5. Wait for the 1-minute break reminder, then **look away** for 20 s. The timer resets.
6. Press **Q**. The report window opens with charts and the score.

## How it works

| Feature | Technique |
|---|---|
| Face, eye and iris landmarks | MediaPipe Face Landmarker (478 points) |
| Distance | Pinhole camera model: the human iris is about 11.7 mm wide, so `distance = focal_length × 11.7 mm / iris_width_px`. Calibration at a known distance finds the focal length. |
| Blinks | Eye Aspect Ratio `(|p2−p6| + |p3−p5|) / (2·|p1−p4|)` drops when the eye closes. The threshold adapts to each person (60% of their recent open-eye EAR); closures shorter than 2 frames or longer than 0.6 s are ignored. |
| Posture | MediaPipe Pose Landmarker. The ratio `(shoulder height − nose height) / shoulder width` shrinks when you slouch, and doesn't change with distance. Slouching = below 80% of your calibrated upright value. |
| Smoothing / alerts | Exponential moving average. An alert only fires after its condition lasts 2–3 s, and the beep has a cooldown. |

## Project structure

```
main.py        app loop, calibration, key handling
metrics.py     EAR, iris distance, posture, blink counter, break timer
detectors.py   MediaPipe face/pose wrapper
alerts.py      alert timing and sounds
hud.py         on-screen drawing
report.py      per-second log, summary, chart
config.py      all thresholds in one place
tests/         unit tests: .venv/bin/python -m unittest discover tests
reports/       one report PNG per session
```

## Limitations

- On a MacBook the camera is inside the lid, so the lid must be open. With the lid closed on an external monitor the app starts but gets no frames.
- Distance is an estimate, accurate to a few cm after calibration. It is less accurate when your head is turned to the side.
- Posture needs both shoulders visible. Tilt the laptop screen back slightly if they are cut off.
- Glasses with strong reflections or very dim light can make blink detection less accurate.
