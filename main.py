
import cv2
import time
from collections import Counter
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
MODEL_NAME = "yolov8n.pt"
CAMERA_INDEX = 0
VIDEO_PATH = "sample_video.mp4"
CONFIDENCE = 0.45
WINDOW_NAME = "ObjectVision - Detection & Tracking"

# ---------------- LOAD MODEL ----------------
print("Loading ObjectVision...")
model = YOLO(MODEL_NAME)

# Select input source
print("Choose input source:")
print("1. Webcam")
print("2. Recorded video")

choice = input("Enter your choice (1 or 2): ").strip()

if choice == "2":
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1100, 700)

print("ObjectVision started successfully!")
print("Press Q to quit | Press S to save screenshot")

previous_time = time.time()
fps = 0.0
total_frames = 0

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]
    output = result.plot()

    # Calculate FPS
    current_time = time.time()
    elapsed = current_time - previous_time

    if elapsed > 0:
        current_fps = 1.0 / elapsed
        fps = current_fps if fps == 0 else 0.85 * fps + 0.15 * current_fps

    previous_time = current_time
    total_frames += 1

    # Count objects and unique tracking IDs
    object_counts = Counter()
    tracked_ids = set()

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            object_counts[model.names[class_id]] += 1

            if box.id is not None:
                tracked_ids.add(int(box.id[0].item()))

    object_total = sum(object_counts.values())

    # Dashboard header
    cv2.rectangle(output, (0, 0), (output.shape[1], 82), (25, 25, 25), -1)

    cv2.putText(
        output, "OBJECTVISION",
        (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 220, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        output, "REAL-TIME DETECTION & TRACKING",
        (20, 62), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (235, 235, 235), 1, cv2.LINE_AA
    )

    stats = [
        f"FPS: {fps:.1f}",
        f"Objects: {object_total}",
        f"Tracking IDs: {len(tracked_ids)}"
    ]

    right_x = output.shape[1] - 20

    for text in reversed(stats):
        text_width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
        )[0][0]

        right_x -= text_width

        cv2.putText(
            output, text, (right_x, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (100, 255, 150), 2, cv2.LINE_AA
        )

        right_x -= 25

    # Bottom summary panel
    panel_y = output.shape[0] - 52

    cv2.rectangle(
        output, (0, panel_y),
        (output.shape[1], output.shape[0]),
        (25, 25, 25), -1
    )

    if object_counts:
        summary = " | ".join(
            f"{name}: {count}"
            for name, count in object_counts.most_common(5)
        )
    else:
        summary = "No objects detected"

    cv2.putText(
        output, summary, (18, panel_y + 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (255, 255, 255), 1, cv2.LINE_AA
    )

    cv2.putText(
        output, "Q: QUIT    S: SAVE SCREENSHOT",
        (18, panel_y + 43),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (0, 220, 255), 1, cv2.LINE_AA
    )

    cv2.imshow(WINDOW_NAME, output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        filename = f"capture_{int(time.time())}.jpg"

        if cv2.imwrite(filename, output):
            print(f"Screenshot saved: {filename}")
        else:
            print("Screenshot could not be saved.")

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()

print("ObjectVision stopped.")
print(f"Total frames processed: {total_frames}")
