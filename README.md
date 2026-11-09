# ObjectVision — Real-Time Object Detection & Hand Tracking

ObjectVision is a computer vision project that combines real-time object detection, object tracking, hand landmark detection, and an interactive finger-controlled mini-game in a single dashboard.

The project uses YOLOv8 for object detection and tracking, OpenCV for video processing, and MediaPipe Tasks for hand landmark detection.

## Features

- **Real-Time Object Detection:** Detects objects using the YOLOv8 model.
- **Object Tracking:** Tracks detected objects and displays tracking IDs using ByteTrack.
- **Hand Tracking:** Detects hand landmarks using MediaPipe Hand Landmarker.
- **Finger-Controlled Mini-Game:** Uses index-finger movement to interact with an on-screen target.
- **Live Dashboard:** Displays detection statistics, tracking information, FPS, hand status, and game score.
- **Webcam and Video Input:** Supports webcam-based detection and video input.
- **Screenshot Capture:** Save a screenshot by pressing `S`.
- **Keyboard Controls:** Press `Q` to quit the application.

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- MediaPipe Tasks
- ByteTrack
- NumPy

## Project Structure

```text
Object-Detection-Tracking/
├── main.py
├── yolov8n.pt
├── hand_landmarker.task
├── sample_video.mp4
├── README.md
├── .gitignore
└── .venv/
```

## Requirements

- Python
- A webcam for live hand tracking
- Required Python packages
- Model files used by the application

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CodeWithAnkit19/Object-Detection-and-Tracking.git
cd Object-Detection-and-Tracking
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages required by the project:

```bash
pip install -r requirements.txt
```

### 4. Prepare the model files

Ensure that the required files are available in the project folder:

- `yolov8n.pt` — YOLOv8 object detection model
- `hand_landmarker.task` — MediaPipe hand landmark model

Keep `sample_video.mp4` in the project folder if you want to use the sample video.

### 5. Run the application

```bash
python main.py
```

## Controls

| Key / Input | Action |
|---|---|
| `Q` | Quit the application |
| `S` | Capture a screenshot |
| Index finger | Control the interactive mini-game |

## How It Works

1. OpenCV captures frames from the webcam or video source.
2. YOLOv8 detects objects in each frame.
3. ByteTrack assigns tracking IDs to detected objects.
4. MediaPipe Tasks detects hand landmarks when webcam hand tracking is enabled.
5. The dashboard displays detection and tracking information.
6. Index-finger movement interacts with the mini-game target.

## Future Improvements

- Improve detection accuracy and tracking stability.
- Add more interactive computer vision features.
- Enhance dashboard design and performance.
- Add more configurable detection settings.

## Author

**Ankit Kumar**

GitHub: [CodeWithAnkit19](https://github.com/CodeWithAnkit19)

## License

This project is intended for educational and learning purposes. Add a specific open-source license if you decide to distribute the project under one.
