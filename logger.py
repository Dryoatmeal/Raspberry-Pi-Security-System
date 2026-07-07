import time


class EventLogger:
    """Appends timestamped messages to a log file."""

    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Logging error: {e}")
