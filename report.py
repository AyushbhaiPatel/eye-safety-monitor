"""Logs the session once per second, then builds the summary and the report chart."""
from datetime import datetime

import matplotlib.pyplot as plt

import config


class SessionLogger:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.rows = []
        self.started_at = datetime.now()
        self._start = self._last = None

    def record(self, now, **values):
        if self._start is None:
            self._start = now
        if self._last is None or now - self._last >= self.interval:
            self._last = now
            self.rows.append({"time_s": round(now - self._start, 1), **values})

    def summary(self, total_blinks, alert_counts, breaks_taken):
        present = [r for r in self.rows if r["face_present"]]
        distances = [r["distance_cm"] for r in present if r["distance_cm"] is not None]
        screen_minutes = len(present) * self.interval / 60

        def percent(key, rows):
            return round(100 * sum(1 for r in rows if r[key]) / len(rows), 1) if rows else 0.0

        too_close = percent("too_close", present)
        slouching = percent("slouching", [r for r in present if r["neck_ratio"] is not None])
        low_blink = percent("low_blink", [r for r in present if r["blink_rate"] is not None])
        return {
            "started_at": self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "session_minutes": round(self.rows[-1]["time_s"] / 60, 1),
            "screen_minutes": round(screen_minutes, 1),
            "avg_distance_cm": round(sum(distances) / len(distances), 1) if distances else None,
            "min_distance_cm": round(min(distances), 1) if distances else None,
            "too_close_percent": too_close,
            "total_blinks": total_blinks,
            "avg_blinks_per_min": round(total_blinks / screen_minutes, 1) if screen_minutes else None,
            "slouching_percent": slouching,
            "breaks_taken": breaks_taken,
            "alerts": dict(alert_counts),
            # 0-100: lose points for the share of time spent in each bad habit
            "eye_care_score": round(max(0.0, 100 - 0.4 * too_close - 0.3 * slouching - 0.3 * low_blink)),
        }

    def save(self, reports_dir, summary, min_distance, slouch_limit):
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"report_{self.started_at:%Y%m%d_%H%M%S}.png"
        build_figure(self.rows, summary, min_distance, slouch_limit).savefig(path, dpi=120)
        return path


def _plot(ax, rows, key, color, title, ylabel, limit, limit_label, shade_below=False):
    points = [(r["time_s"], r[key]) for r in rows if r[key] is not None]
    t, v = [p[0] for p in points], [p[1] for p in points]
    ax.plot(t, v, color=color, label=key.replace("_", " "))
    if limit is not None:
        ax.axhline(limit, color="tab:red", linestyle="--", label=limit_label)
        if shade_below:
            ax.fill_between(t, v, limit, where=[x < limit for x in v], color="tab:red", alpha=0.25)
    if not v:
        ax.text(0.5, 0.5, "not enough data", ha="center", va="center", color="dimgray",
                transform=ax.transAxes)
    ax.set(title=title, xlabel="time (s)", ylabel=ylabel)
    ax.legend(loc="upper right")


def build_figure(rows, summary, min_distance, slouch_limit):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle(f"Eye Safety Report  -  {summary['started_at']}", fontsize=14, fontweight="bold")
    _plot(ax[0][0], rows, "distance_cm", "tab:blue", "Distance from screen", "cm",
          min_distance, "minimum safe", shade_below=True)
    _plot(ax[0][1], rows, "blink_rate", "tab:green", "Blink rate", "blinks / min",
          config.LOW_BLINK_RATE, "too low")
    _plot(ax[1][0], rows, "neck_ratio", "tab:purple", "Posture (head height above shoulders)",
          "ratio", slouch_limit, "slouching below")

    ax[1][1].axis("off")
    lines = [
        ("Eye-care score", f"{summary['eye_care_score']} / 100"),
        ("Session length", f"{summary['session_minutes']} min"),
        ("Time facing screen", f"{summary['screen_minutes']} min"),
        ("Average distance", f"{summary['avg_distance_cm']} cm"),
        ("Closest distance", f"{summary['min_distance_cm']} cm"),
        ("Time too close", f"{summary['too_close_percent']} %"),
        ("Total blinks", str(summary["total_blinks"])),
        ("Average blink rate", f"{summary['avg_blinks_per_min']} /min"),
        ("Time slouching", f"{summary['slouching_percent']} %"),
        ("Breaks taken", str(summary["breaks_taken"])),
        ("Alerts", ", ".join(f"{k} x{v}" for k, v in summary["alerts"].items()) or "none"),
    ]
    for i, (label, value) in enumerate(lines):
        ax[1][1].text(0.02, 0.95 - i * 0.085, label, fontsize=11, color="dimgray", transform=ax[1][1].transAxes)
        ax[1][1].text(0.48, 0.95 - i * 0.085, value, fontsize=11, fontweight="bold", transform=ax[1][1].transAxes)
    fig.tight_layout()
    return fig
