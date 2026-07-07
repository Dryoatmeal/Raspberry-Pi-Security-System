import requests


class DiscordNotifier:
    """Sends alert text to a Discord webhook. Logs instead if that fails or isn't set up."""

    def __init__(self, webhook_url, logger):
        self.webhook_url = webhook_url
        self.logger = logger

    def send(self, message):
        if not self.webhook_url:
            self.logger.log(f"Webhook not configured. Message not sent: {message}")
            return

        try:
            requests.post(self.webhook_url, json={"content": message}, timeout=3)
        except Exception as e:
            self.logger.log(f"Webhook error: {e}")
