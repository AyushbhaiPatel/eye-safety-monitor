"""Turns raw conditions (too close, slouching, ...) into on-screen alerts and beeps."""
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MAC_SOUND = Path("/System/Library/Sounds/Ping.aiff")


class AlertManager:
    """Fires an alert only when a condition lasts a while, and limits how often it beeps."""

    def __init__(self, hold_seconds, cooldown, sound=True):
        self.hold_seconds = hold_seconds
        self.cooldown = cooldown
        self.sound = sound
        self.counts = defaultdict(int)   # how many separate times each alert fired
        self._since = {}
        self._last_beep = {}
        self._active = {}

    def update(self, key, condition, now, message):
        if not condition:
            self._since.pop(key, None)
            self._active.pop(key, None)
            return False
        start = self._since.setdefault(key, now)
        if now - start < self.hold_seconds.get(key, 0):
            return False
        if key not in self._active:
            self.counts[key] += 1
        self._active[key] = message
        if now - self._last_beep.get(key, -math.inf) >= self.cooldown:
            self._last_beep[key] = now
            self._beep()
        return True

    def clear(self):
        self._since.clear()
        self._active.clear()

    @property
    def active_messages(self):
        return list(self._active.values())

    def _beep(self):
        if not self.sound:
            return
        if sys.platform == "darwin" and MAC_SOUND.exists() and shutil.which("afplay"):
            subprocess.Popen(["afplay", str(MAC_SOUND)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)
