from collections import deque


class DetectionToggle:
    """Lets someone type a password on the OpenCV window to enable/disable detection."""

    def __init__(self, password, logger, notifier, enabled=True):
        self.password = password
        self.logger = logger
        self.notifier = notifier
        self.enabled = enabled
        self.key_buffer = deque(maxlen=32)

    def handle_key(self, key):
        if key == 255:
            return

        if 32 <= key <= 126:
            self.key_buffer.append(chr(key))
        elif key in (10, 13):
            typed = "".join(self.key_buffer)
            if typed == self.password:
                self.enabled = not self.enabled
                state = "ENABLED" if self.enabled else "DISABLED"
                self.logger.log(f"Detection toggled: {state}")
                self.notifier.send(f"🔒 Detection {state}")
            self.key_buffer.clear()
