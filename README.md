# Raspberry Pi Human Detection (YOLOv8 + Discord Alerts)

This project runs a human-detection loop on a Raspberry Pi using a camera and a YOLOv8 model.  
If a person is detected continuously for 3 seconds, the system:

- Sends a Discord webhook alert (`🚨 Human detected for 3 seconds!`)
- Re-sends status updates every 10 seconds while the person remains
- Sends an “area clear” notification after 3 seconds with no detections  

Detection can be toggled on/off locally using a password typed into the OpenCV window.

---

## Features

- **YOLOv8 person detection** running directly on the Raspberry Pi  
- **Threaded camera reader** to keep frame capture responsive  
- **Configurable timers**:
  - 3-second confirmation before triggering
  - 3-second “no human” window before clearing
  - 10-second periodic “still detected” notifications
- **Discord integration** via a webhook URL
- **Password-protected enable/disable**:
  - Type the password and press Enter in the OpenCV window to toggle detection
- **Visual overlay**:
  - Bounding boxes around detected people
  - Status text (`DETECTION ENABLED` / `DETECTION DISABLED`)
  - Countdown timer
  - Flashing red overlay when triggered

---

## Hardware & Software Requirements

- Raspberry Pi 4 (or similar)  
- Camera accessible via V4L2 (e.g., `/dev/video0`)  
- Python 3.10+ (adjust as needed)  
- Internet access for sending Discord webhook requests  

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Dryoatmeal/Raspberry-Pi-Security-System.git
   cd Raspberry-Pi-Security-System