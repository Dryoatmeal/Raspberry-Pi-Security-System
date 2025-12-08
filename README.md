# Raspberry Pi Human Detection System (YOLOv8 + Discord Alerts)

A real-time human-detection security system designed for a Raspberry Pi.  
The application uses a camera feed, performs person detection with a YOLOv8 model from the Ultralytics framework, and sends alerts through a Discord Webhook.  
All major events are logged for later review.

---

## Features

### Real-Time Detection
- Runs YOLOv8 person detection at configurable frame intervals.
- Threaded frame capture for smooth performance.
- Bounding boxes, human count, countdown timer, and detection status overlays.

### Alerting System
- Sends a Discord alert when a person is continuously detected for 3 seconds.
- Sends an update every 10 seconds while a person remains present.
- Sends a “no human detected” message after the area has been clear for 3 seconds.

### Safety Controls
- Detection can be toggled on/off using a password typed into the OpenCV window.
- Flashing red overlay when the system is actively triggered.

### Logging
- All events (detections, toggles, status changes) are logged to a file.

### Configurable Behavior
- All parameters (webhook URL, thresholds, FPS, model name, etc.) live in `config.json`.
- `config.json` is excluded from version control for security.


---

## Quick Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Dryoatmeal/Raspberry-Pi-Security-System.git
   cd Raspberry-Pi-Security-System
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create your configuration**

   ```bash
   cp config.example.json config.json
   nano config.json
   ```

   Fill in:
   - your Discord webhook URL  
   - the log file path  
   - your password  
   - optional detection parameters  

4. **Run the system**

   ```bash
   python3 human_detection.py
   ```

Controls:
- Type your password + press **Enter** → toggle detection  
- Press **q** → quit  

---

## How It Works 

### 1. Configuration Loader
Reads `config.json` and initializes all runtime parameters.

### 2. Threaded Camera Capture
A dedicated thread continuously reads frames from the camera device (`/dev/video0` on Raspberry Pi), ensuring the detection loop never stalls.

### 3. YOLOv8 Inference
Uses Ultralytics YOLOv8 (`yolov8n.pt` by default) to detect the "person" class.  
Runs inference every few frames to balance performance and accuracy.

### 4. Detection Logic
Timers control when alerts are sent:
- 3 seconds of continuous detection → send “Human detected” alert  
- Every 10 seconds → send “still detected” updates  
- 3 seconds with no detection → send “Area clear”  

A flashing red overlay appears while triggered.

### 5. Visualization & Controls
- Human count  
- Bounding boxes  
- Countdown timer  
- Detection enabled/disabled state  
- Password-protected toggling  

### 6. Logging
All important events are written to the log file defined in `config.json`.
