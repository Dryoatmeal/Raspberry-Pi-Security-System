import time
import threading
from collections import deque

import cv2
import platform

class ThreadedCamera:
    """
    Reads frames from the camera on a background thread and keeps the most
    recent ones in a small buffer, so the main loop can grab the latest
    frame without waiting on camera I/O.
    """

    def __init__(self, src=0, queue_size=2):
        # On Raspberry Pi this is normally /dev/video0.
        system = platform.system()

        self.cap = None

        if system == "Linux":

            # Try V4L2 first (best for Raspberry Pi)
            for device in [src, 0, 1, "/dev/video0", "/dev/video1"]:

                cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

                if cap.isOpened():
                    self.cap = cap
                    print(f"Using camera: {device}")
                    break

        elif system == "Darwin":

            # macOS
            for device in [src, 0, 1, 2]:

                cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)

                if cap.isOpened():
                    self.cap = cap
                    print(f"Using camera: {device}")
                    break

        else:

            # Windows
            for device in [src, 0, 1, 2]:

                cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)

                if cap.isOpened():
                    self.cap = cap
                    print(f"Using camera: {device}")
                    break

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("No camera could be opened.")

        # Raspberry Pi camera settings
        if system == "Linux":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)

            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
            print("Camera codec:", codec)

        self.queue = deque(maxlen=queue_size)
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.queue.append(frame)

    def read(self):
        with self.lock:
            if self.queue:
                frame = self.queue[-1].copy()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Camera read failed")

        # Un-mirror the image
        return cv2.flip(frame, 1)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()
