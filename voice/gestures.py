"""
Hand gesture recognition via MediaPipe.

Recognizes three gestures:
- THUMBS_UP   → positive / advance to next passage
- THUMBS_DOWN → negative / re-read passage
- OPEN_PALM   → "I want to speak" (activates microphone)

The hold_frames parameter in GestureWatcher requires a gesture to be stable
for N consecutive frames before confirming — important for accessibility when
the user has tremor (e.g. EDS / connective tissue disorder).

MediaPipe landmark coordinate system: y=0 is the top of the frame, so a
smaller y value means a higher position in the image.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class Gesture(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OPEN_PALM = "open_palm"


# MediaPipe landmark indices used for gesture classification.
# Finger layout: (tip_idx, pip_idx)
_FINGER_PAIRS = [
    (8, 6),   # index:  tip, PIP
    (12, 10), # middle: tip, PIP
    (16, 14), # ring:   tip, PIP
    (20, 18), # pinky:  tip, PIP
]

_THUMB_TIP = 4
_THUMB_MCP = 2

_CURL_THRESHOLD = 3  # at least this many of the 4 fingers must be curled/extended


def classify_hand(landmarks) -> Optional[Gesture]:
    """Classify a single hand into a Gesture, or return None.

    Parameters
    ----------
    landmarks:
        A list of 21 MediaPipe NormalizedLandmark-like objects (with .x, .y, .z),
        or None if no hand was detected.

    Returns
    -------
    Gesture or None
    """
    if landmarks is None:
        return None

    # Count curled and extended fingers across the four non-thumb fingers.
    curled = 0
    extended = 0
    for tip_idx, pip_idx in _FINGER_PAIRS:
        tip_y = landmarks[tip_idx].y
        pip_y = landmarks[pip_idx].y
        if tip_y >= pip_y:   # tip is lower than (or level with) PIP → curled
            curled += 1
        else:                # tip is higher than PIP → extended
            extended += 1

    fingers_curled = curled >= _CURL_THRESHOLD
    fingers_extended = extended >= _CURL_THRESHOLD

    if fingers_curled:
        thumb_tip_y = landmarks[_THUMB_TIP].y
        thumb_mcp_y = landmarks[_THUMB_MCP].y
        if thumb_tip_y < thumb_mcp_y:
            return Gesture.THUMBS_UP
        elif thumb_tip_y > thumb_mcp_y:
            return Gesture.THUMBS_DOWN
        # thumb level with MCP — ambiguous, fall through to None
        return None

    if fingers_extended:
        return Gesture.OPEN_PALM

    return None


class GestureWatcher:
    """Reads frames from a camera and emits confirmed gestures.

    A gesture must appear in `hold_frames` consecutive frames before it is
    returned by poll(). After confirmation the counter resets, so the same
    gesture won't re-fire until it disappears and re-stabilizes.

    Parameters
    ----------
    hold_frames:
        Number of consecutive stable frames required for confirmation.
        Higher values reduce false positives from tremor.
    """

    def __init__(self, hold_frames: int = 10) -> None:
        self._hold_frames = hold_frames
        self._cap = None
        self._hands = None
        self._mp_hands = None
        self._mp_drawing = None
        self._current_gesture: Optional[Gesture] = None
        self._streak: int = 0
        self._confirmed: bool = False  # True once we've fired this gesture

    def start(self) -> None:
        """Open the default camera (index 0) and initialise MediaPipe Hands."""
        import cv2
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._cap = cv2.VideoCapture(0)
        self._current_gesture = None
        self._streak = 0
        self._confirmed = False

    def poll(self) -> Optional[Gesture]:
        """Read one frame and return a confirmed gesture or None.

        Returns the gesture exactly once when `hold_frames` consecutive frames
        show the same gesture. After that, the counter resets and the gesture
        won't re-fire until it changes and restabilises.
        """
        if self._cap is None or self._hands is None:
            raise RuntimeError("GestureWatcher.start() must be called before poll()")

        import cv2

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Convert BGR → RGB for MediaPipe.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        gesture: Optional[Gesture] = None
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark
            gesture = classify_hand(landmarks)

        if gesture == self._current_gesture and gesture is not None:
            if not self._confirmed:
                self._streak += 1
                if self._streak >= self._hold_frames:
                    self._confirmed = True
                    return gesture
        else:
            # Gesture changed (or became None) — reset tracking.
            self._current_gesture = gesture
            self._streak = 1 if gesture is not None else 0
            self._confirmed = False

        return None

    def stop(self) -> None:
        """Release camera and MediaPipe resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        self._mp_hands = None
