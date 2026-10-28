"""MediaPipe gesture recognition, model setup, and gesture-to-intent voting."""

from collections import Counter, deque
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

GESTURE_TO_INTENT = {
    "Thumb_Up": "ENGAGE",
    "Pointing_Up": "CHECK_IN",
    "Open_Palm": "CAUTION",
    "Thumb_Down": "DE_ESCALATE",
}

# MediaPipe hand landmark indices: wrist, four thumb points, then four points
# for each remaining finger. These edges draw the familiar hand skeleton.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def ensure_model(model_path):
    """Download the MediaPipe task model with curl when missing or empty."""
    if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
        return
    folder = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(folder, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".gesture-model-", dir=folder)
    os.close(fd)
    try:
        print(f"[MODEL] Downloading Gesture Recognizer model to {model_path}")
        subprocess.run(
            ["curl", "-fL", "--retry", "2", "--connect-timeout", "15",
             "--max-time", "300", "-o", temp_path, MODEL_URL],
            check=True,
        )
        if os.path.getsize(temp_path) == 0:
            raise RuntimeError("download returned an empty file")
        os.replace(temp_path, model_path)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Could not download the gesture model: {exc}. Download it manually "
            f"to {model_path} or check curl and internet access."
        ) from exc


class GestureRecognizer:
    """Recognize gestures from original camera frames, without tracking input."""

    def __init__(self, model_path):
        ensure_model(model_path)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(f"MediaPipe Gesture Recognizer unavailable: {exc}") from exc

        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self._mp = mp
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._started_at = time.monotonic()
        self._frame_number = 0
        self._last_timestamp_ms = 0

    def process_frame(self, original_frame):
        """Return the best gesture and landmarks from the original full frame."""
        rgb = np.ascontiguousarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._frame_number += 1
        timestamp_ms = int((time.monotonic() - self._started_at) * 1000)
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        hands = []
        tops = []
        categories_per_hand = result.gestures or []
        handedness_per_hand = result.handedness or []
        for index, landmarks in enumerate(result.hand_landmarks or []):
            categories = categories_per_hand[index] if index < len(categories_per_hand) else []
            top = categories[0] if categories else None
            if top is not None:
                tops.append(top)
            handedness = "Hand"
            if index < len(handedness_per_hand) and handedness_per_hand[index]:
                handedness = handedness_per_hand[index][0].category_name
            hands.append({
                "landmarks": landmarks,
                "label": top.category_name if top else "Hand detected",
                "score": float(top.score) if top else None,
                "handedness": handedness,
            })
        if not tops:
            return None, None, hands
        best = max(tops, key=lambda category: category.score)
        return best.category_name, float(best.score), hands

    @staticmethod
    def draw_hand_skeletons(frame, hands):
        """Draw detected hand bones and landmark dots onto the preview frame."""
        h, w = frame.shape[:2]
        bone = (86, 235, 165)
        joint = (70, 240, 255)
        for hand in hands:
            landmarks = hand["landmarks"]
            points = [
                (max(0, min(w - 1, int(round(point.x * w)))),
                 max(0, min(h - 1, int(round(point.y * h)))))
                for point in landmarks
            ]
            if not points:
                continue
            for start, end in HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(frame, points[start], points[end], (14, 42, 34), 5,
                             cv2.LINE_AA)
                    cv2.line(frame, points[start], points[end], bone, 2,
                             cv2.LINE_AA)
            for index, point in enumerate(points):
                cv2.circle(frame, point, 5, (14, 42, 34), -1, cv2.LINE_AA)
                cv2.circle(frame, point, 3, joint if index in (4, 8, 12, 16, 20) else bone,
                           -1, cv2.LINE_AA)

            x = min(point[0] for point in points)
            y = max(18, min(point[1] for point in points) - 9)
            score = hand["score"]
            suffix = f" {score:.2f}" if score is not None else ""
            text = f"{hand['handedness']} · {hand['label']}{suffix}"
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (14, 42, 34), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (238, 255, 247), 1, cv2.LINE_AA)
        return frame

    def close(self):
        self._recognizer.close()


class GestureIntentVoter:
    """Map stable recognized gestures to intents using the runner's hold check."""

    def __init__(self, min_score=0.6, window_s=1.0, min_votes=4, min_share=0.75):
        self.min_score = min_score
        self.window_s = window_s
        self.min_votes = min_votes
        self.min_share = min_share
        self.votes = deque()

    def update(self, timestamp, label, score):
        intent = None
        if label in GESTURE_TO_INTENT and score is not None and score >= self.min_score:
            intent = GESTURE_TO_INTENT[label]
            self.votes.append((timestamp, intent))

        while self.votes and timestamp - self.votes[0][0] > self.window_s:
            self.votes.popleft()
        if not self.votes:
            return None
        counts = Counter(vote for _, vote in self.votes)
        winner, count = counts.most_common(1)[0]
        if count >= self.min_votes and count / len(self.votes) >= self.min_share:
            return winner
        return None
