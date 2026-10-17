# ObjectVision – Object Detection and Tracking

## Project Overview
ObjectVision is a real-time computer vision application developed using Python, OpenCV, and the YOLO object detection model. It detects objects through a webcam and tracks them across video frames using ByteTrack.

## Features
- Real-time object detection
- Object tracking with tracking IDs
- Live FPS display
- Detected object count
- Object labels and bounding boxes
- Screenshot capture using the S key
- Webcam inputs
- Recorded video input support (requires a video file)

## Technologies Used
- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- NumPy

## Requirements
- Python installed on your computer
- Webcam for live detection
- Internet connection for initial package or model downloads

## Installation
Install the required packages inside your Python environment:

```bash
pip install opencv-python ultralytics
```

## How to Run
1. Open the project folder in VS Code.
2. Activate or use your Python virtual environment.
3. Run the application:

```bash
python main.py
```

4. Select the webcam option when prompted.
5. View detected objects and tracking information.

## Keyboard Controls
- **Q** – Quit the application
- **S** – Save a screenshot

## Project Objective
The objective of ObjectVision is to demonstrate real-time object detection and tracking using computer vision techniques.

## Future Enhancements
- Recorded video testing
- Detection history and analytics
- Saving processed video output
- Improved user interface

## Author
Developed as a computer vision project for internship learning and practical experience.
