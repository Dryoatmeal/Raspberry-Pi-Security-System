from pathlib import Path
import json
import os


def load_config(config_path):
    """Load settings from a JSON config file, filling in defaults for anything missing."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise RuntimeError(
            f"Config file not found at {config_path}. "
            "Rename config.example.json to config.json and fill in your values."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        "webhook_url": raw.get("webhook_url", ""),
        "log_file": raw.get("log_file", "human_detection.log"),
        "password": raw.get("password", "password"),
        "camera_index": int(raw.get("camera_index", 0)),
        "target_fps": float(raw.get("target_fps", 15)),
        "conf_threshold": float(raw.get("conf_threshold", 0.65)),
        "imgsz": int(raw.get("imgsz", 224)),
        "model_name": raw.get("model_name", "yolov8n.pt"),
    }


def get_config_path():
    """Path to config.json, or the HD_CONFIG env var if it's set."""
    default_path = Path(__file__).with_name("config.json")
    return os.getenv("HD_CONFIG", default_path)
