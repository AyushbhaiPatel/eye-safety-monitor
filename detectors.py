"""Thin wrapper around the MediaPipe face and pose landmark models."""
import urllib.request

from mediapipe.tasks.python import BaseOptions, vision

import config

MODELS = {
    "face": (config.FACE_MODEL, config.FACE_MODEL_URL, vision.FaceLandmarker,
             vision.FaceLandmarkerOptions, "num_faces"),
    "pose": (config.POSE_MODEL, config.POSE_MODEL_URL, vision.PoseLandmarker,
             vision.PoseLandmarkerOptions, "num_poses"),
}


class Landmarker:
    def __init__(self, kind):
        path, url, model_cls, options_cls, count_key = MODELS[kind]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)
        self.kind = kind
        self._model = model_cls.create_from_options(options_cls(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, **{count_key: 1}))

    def detect(self, image, timestamp_ms):
        """Landmarks of the first face/person in the frame, or None."""
        result = self._model.detect_for_video(image, timestamp_ms)
        found = result.face_landmarks if self.kind == "face" else result.pose_landmarks
        return found[0] if found else None

    def close(self):
        self._model.close()
