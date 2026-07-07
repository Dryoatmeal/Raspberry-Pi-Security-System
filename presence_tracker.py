import time


class PresenceTracker:
    """
    Watches human_count over time. Fires a 'detected' alert once someone has
    been in frame continuously for confirm_seconds, then fires an 'area
    clear' alert once they've been gone for clear_seconds.
    """

    def __init__(self, notifier, logger, confirm_seconds=3.0, clear_seconds=3.0, repeat_interval=10.0):
        self.notifier = notifier
        self.logger = logger
        self.confirm_seconds = confirm_seconds
        self.clear_seconds = clear_seconds
        self.repeat_interval = repeat_interval

        self.countdown = confirm_seconds
        self.no_human_timer = 0.0
        self.counting_active = False
        self.triggered = False
        self.last_repeat_notification = 0.0

    def update(self, human_count, delta_time):
        if human_count > 0:
            self.no_human_timer = 0.0

            if not self.counting_active and not self.triggered:
                self.counting_active = True

            if self.counting_active and not self.triggered:
                self.countdown -= delta_time
                if self.countdown <= 0:
                    self.countdown = 0.0
                    self.triggered = True
                    self.last_repeat_notification = time.time()
                    self.notifier.send("**Human detected for 3 seconds!**")
                    self.logger.log("Human detected for 3 seconds.")

            if self.triggered and time.time() - self.last_repeat_notification >= self.repeat_interval:
                self.notifier.send("Human still detected (10-second update).")
                self.logger.log("Human still detected (10-second update).")
                self.last_repeat_notification = time.time()

        else:
            if self.triggered or self.counting_active:
                self.no_human_timer += delta_time
                if self.no_human_timer >= self.clear_seconds:
                    self.countdown = self.confirm_seconds
                    self.triggered = False
                    self.counting_active = False
                    self.no_human_timer = 0.0
                    self.notifier.send("**No humans detected for 3 seconds. Area clear.**")
                    self.logger.log("Area clear after 3 seconds of no detection.")
