import time
import threading
from collections import deque
from pathlib import Path
import json
import os
from typing import Deque, List, Tuple

import requests
import cv2
from ultralytics import YOLO

# Configuration loading
def load_config(config_path: str | Path) -> dict:
    """
    Load configuration from a JSON file.
    Expected keys:
      - webhook_url: Discord webhook URL for alerts.
      - log_file: Absolute or relative path to a log file.
      - password: String used to toggle detection on/off.
      - camera_index: (optional) index of the camera device (default 0).
      - target_fps: (optional) target frames per second for main loop.
      - conf_threshold: (optional) YOLO confidence threshold.
      - imgsz: (optional) YOLO image size.
      - model_name: (optional) YOLO model file, e.g. 'yolov8n.pt'.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise RuntimeError(
            f"Config file not found at {config_path}. "
            "Rename config.example.json to config.json and fill in your values."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Default config path: same directory as this script, file named config.json
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG = load_config(os.getenv("HD_CONFIG", DEFAULT_CONFIG_PATH))

WEBHOOK_URL: str = CONFIG.get("webhook_url", "")
LOG_FILE: str = CONFIG.get("log_file", "human_detection.log")
PASSWORD: str = CONFIG.get("password", "password")

CAMERA_INDEX: int = int(CONFIG.get("camera_index", 0))
TARGET_FPS: float = float(CONFIG.get("target_fps", 15))
CONF_THRESHOLD: float = float(CONFIG.get("conf_threshold", 0.65))
IMGSZ: int = int(CONFIG.get("imgsz", 224))
MODEL_NAME: str = CONFIG.get("model_name", "yolov8n.pt")


# Logging & notifications
def log_event(message: str) -> None:
    """
    Append a timestamped message to the log file.
    If the log file cannot be written, the error is printed to stdout.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Logging error: {e}")


def send_webhook_message(message: str) -> None:
    """
    Send a text message to the configured Discord webhook.
    This is used for alerts such as 'human detected' or 'area clear'.
    """
    if not WEBHOOK_URL:
        # Fail silently but log if no webhook is configured
        log_event(f"Webhook not configured. Message not sent: {message}")
        return

    try:
        data = {"content": message}
        requests.post(WEBHOOK_URL, json=data, timeout=3)
    except Exception as e:
        log_event(f"Webhook error: {e}")


# Camera wrapper (threaded)
class ThreadedCamera:
    """
    Simple threaded camera reader using OpenCV.
    A background thread continuously reads frames from the camera and stores
    them in a deque. The main loop calls `read()` to get the latest frame
    without blocking on I/O.
    This helps keep the detection loop responsive.
    """

    def __init__(self, src: int = 0, queue_size: int = 2) -> None:
        # On Raspberry Pi, it typically uses /dev/video0; src is kept for clarity.
        self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {src}")

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        # Print the codec being used (useful for debugging on Pi)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print("Camera codec in use:", codec)

        self.queue: Deque = deque(maxlen=queue_size)
        self.stopped: bool = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        """
        Background thread loop: continuously read frames into the queue.
        """
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # If we fail to read a frame, wait a bit and try again
                time.sleep(0.01)
                continue

            with self.lock:
                self.queue.append(frame)

    def read(self):
        """
        Return the latest frame available.
        If the queue is empty, this falls back to a direct read from the camera.
        """
        with self.lock:
            if self.queue:
                return self.queue[-1].copy()

        # Fallback: direct read if queue is empty
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return frame

    def release(self) -> None:
        """
        Stop the background thread and release the camera device.
        """
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# Drawing utilities
def draw_detections(frame, boxes: List[Tuple[float, float, float, float]]) -> None:
    """
    Draw bounding boxes for detected humans on the frame.
    """
    for (x1, y1, x2, y2) in boxes:
        x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)


# Main detection loop
def main(
    camera_index: int = CAMERA_INDEX,
    target_fps: float = TARGET_FPS,
    conf_threshold: float = CONF_THRESHOLD,
    imgsz: int = IMGSZ,
    model_name: str = MODEL_NAME,
) -> None:
    """
    Run the human detection loop.
    - Loads a YOLOv8 model.
    - Reads frames from a threaded camera.
    - Detects persons every N frames.
    - Triggers alerts if a human is present for 3 seconds.
    - Sends "area clear" once no human is detected for 3 seconds.
    """

    print("Loading model:", model_name)
    model = YOLO(model_name)
    person_class_id = 0  # COCO person class index

    cam = ThreadedCamera(src=camera_index, queue_size=2)
    log_event("Camera started successfully.")

    time.sleep(0.5)
    print("Press 'q' to quit.")

    frame_interval = 1.0 / float(target_fps)
    last_time = time.time()

    # Timers and state flags
    countdown = 3.00          # Seconds required to confirm detection
    no_human_timer = 0.0      # Seconds since last human when counting down to "clear"
    counting_active = False   # True when counting down to trigger
    triggered = False         # True once alert is active
    flash_state = False       # For flashing red overlay
    flash_last_time = time.time()

    frame_counter = 0
    detect_every = 3          # Run YOLO every N frames
    last_boxes: List = []

    # Notifications
    last_repeat_notification = 0.0  # Timestamp for repeat notifications

    # Detection enable/disable via password
    detection_enabled = True
    key_buffer: Deque[str] = deque(maxlen=32)  # stores last typed characters

    try:
        while True:
            loop_start = time.time()
            frame = cam.read()
            frame_counter += 1

            # Keyboard handling (OpenCV window must be focused)
            key = cv2.waitKey(1) & 0xFF

            if key != 255:  # any key pressed
                if 32 <= key <= 126:  # printable ASCII chars
                    key_buffer.append(chr(key))
                elif key in (10, 13):  # ENTER key
                    typed = "".join(key_buffer)
                    if typed == PASSWORD:
                        detection_enabled = not detection_enabled
                        state = "ENABLED" if detection_enabled else "DISABLED"
                        log_event(f"Detection toggled: {state}")
                        send_webhook_message(f"🔒 Detection {state}")
                    key_buffer.clear()

            if key == ord("q"):
                # Clean exit on 'q'
                break

            # Run YOLO inference periodically
            if frame_counter % detect_every == 0:
                results = model.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=conf_threshold,
                    classes=[person_class_id],
                    verbose=False,
                )

                last_boxes = []
                if len(results) > 0:
                    r = results[0]
                    if hasattr(r, "boxes"):
                        for box in r.boxes:
                            xyxy = box.xyxy.cpu().numpy().flatten()
                            last_boxes.append(xyxy)

            # Draw boxes on frame for visualization
            draw_detections(frame, last_boxes)

            now = time.time()
            delta_time = now - last_time
            last_time = now

            human_count = len(last_boxes)

            # Status text overlays
            status_text = "DETECTION ENABLED" if detection_enabled else "DETECTION DISABLED"
            status_color = (0, 255, 0) if detection_enabled else (0, 0, 255)

            cv2.putText(
                frame,
                status_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                f"Humans: {human_count}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # Skip detection logic entirely if disabled
            if not detection_enabled:
                cv2.putText(
                    frame,
                    "Detection paused",
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Raspberry Pi Human Detection", frame)

                elapsed = time.time() - loop_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                continue

            # Human presence logic (trigger & clear)
            if human_count > 0:
                # Reset "no human" timer whenever at least one human is detected
                no_human_timer = 0.0

                if not counting_active and not triggered:
                    # Start countdown when we first see a human
                    counting_active = True

                if counting_active and not triggered:
                    # Count down to trigger
                    countdown -= delta_time
                    if countdown <= 0:
                        countdown = 0.00
                        triggered = True
                        last_repeat_notification = time.time()
                        send_webhook_message("**Human detected for 3 seconds!**")
                        log_event("Human detected for 3 seconds.")

                # While triggered, keep sending periodic updates
                if triggered:
                    if time.time() - last_repeat_notification >= 10:
                        send_webhook_message(
                            "Human still detected (10-second update)."
                        )
                        log_event("Human still detected (10-second update).")
                        last_repeat_notification = time.time()

            else:
                # No humans detected in this frame
                if triggered or counting_active:
                    no_human_timer += delta_time
                    if no_human_timer >= 3.0:
                        # Reset everything after 3 seconds of no humans
                        countdown = 3.00
                        triggered = False
                        counting_active = False
                        flash_state = False
                        send_webhook_message(
                            "**No humans detected for 3 seconds. Area clear.**"
                        )
                        log_event("Area clear after 3 seconds of no detection.")
                        no_human_timer = 0.0

            # Countdown overlay
            cv2.putText(
                frame,
                f"Countdown: {countdown:05.2f}s",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 128, 255),
                2,
            )

            # Flash overlay when triggered
            if triggered:
                if time.time() - flash_last_time >= 0.5:
                    flash_state = not flash_state
                    flash_last_time = time.time()

                if flash_state:
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (0, 0),
                        (frame.shape[1], frame.shape[0]),
                        (0, 0, 255),
                        -1,
                    )
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            cv2.imshow("Raspberry Pi Human Detection", frame)

            # Maintain approximate target FPS
            elapsed = time.time() - loop_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        log_event("Interrupted by user.")
        print("Interrupted by user.")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        log_event("Camera and GUI shut down cleanly.")


if __name__ == "__main__":
    main()