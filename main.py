import time

import cv2

from config import load_config, get_config_path
from logger import EventLogger
from notifier import DiscordNotifier
from camera import ThreadedCamera
from detector import PersonDetector
from presence_tracker import PresenceTracker
from detection_toggle import DetectionToggle
from visualization import draw_detections, draw_status_overlay, FlashOverlay


def main():
    config = load_config(get_config_path())

    logger = EventLogger(config["log_file"])
    notifier = DiscordNotifier(config["webhook_url"], logger)
    detector = PersonDetector(config["model_name"], config["conf_threshold"], config["imgsz"])
    tracker = PresenceTracker(notifier, logger)
    toggle = DetectionToggle(config["password"], logger, notifier)
    flash = FlashOverlay(interval=0.5)

    cam = ThreadedCamera(src=config["camera_index"], queue_size=2)
    logger.log("Camera started successfully.")

    time.sleep(0.5)
    print("Press 'q' to quit.")

    frame_interval = 1.0 / config["target_fps"]
    last_time = time.time()

    frame_counter = 0
    detect_every = 3
    last_boxes = []

    try:
        while True:
            loop_start = time.time()
            frame = cam.read()
            frame_counter += 1

            key = cv2.waitKey(1) & 0xFF
            toggle.handle_key(key)

            if key == ord("q"):
                break

            if frame_counter % detect_every == 0:
                last_boxes = detector.detect(frame)

            draw_detections(frame, last_boxes)

            now = time.time()
            delta_time = now - last_time
            last_time = now

            human_count = len(last_boxes)

            if not toggle.enabled:
                draw_status_overlay(frame, toggle.enabled, human_count, tracker.countdown)
                cv2.imshow("Raspberry Pi Human Detection", frame)

                elapsed = time.time() - loop_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                continue

            was_triggered = tracker.triggered
            tracker.update(human_count, delta_time)
            if was_triggered and not tracker.triggered:
                flash.reset()

            draw_status_overlay(frame, toggle.enabled, human_count, tracker.countdown)

            if tracker.triggered:
                frame = flash.apply(frame)

            cv2.imshow("Raspberry Pi Human Detection", frame)

            elapsed = time.time() - loop_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        logger.log("Interrupted by user.")
        print("Interrupted by user.")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        logger.log("Camera and GUI shut down cleanly.")


if __name__ == "__main__":
    main()
