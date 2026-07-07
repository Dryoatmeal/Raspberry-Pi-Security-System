import time

import cv2


def draw_detections(frame, boxes):
    """Draw a green box around each detected person."""
    for (x1, y1, x2, y2) in boxes:
        x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)


def draw_status_overlay(frame, detection_enabled, human_count, countdown):
    """Draw the enabled/disabled state, human count, and countdown/paused text."""
    status_text = "DETECTION ENABLED" if detection_enabled else "DETECTION DISABLED"
    status_color = (0, 255, 0) if detection_enabled else (0, 0, 255)

    cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, f"Humans: {human_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if detection_enabled:
        cv2.putText(frame, f"Countdown: {countdown:05.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
    else:
        cv2.putText(frame, "Detection paused", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class FlashOverlay:
    """Alternates a translucent red overlay on and off to make an active alert flash."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.active = False
        self.last_toggle = time.time()

    def reset(self):
        self.active = False

    def apply(self, frame):
        if time.time() - self.last_toggle >= self.interval:
            self.active = not self.active
            self.last_toggle = time.time()

        if not self.active:
            return frame

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
