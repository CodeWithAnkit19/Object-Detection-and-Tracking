
import cv2
import time
import random
from collections import Counter, deque

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ================= CONFIGURATION =================
MODEL_NAME = "yolov8n.pt"
HAND_MODEL = "hand_landmarker.task"
CAMERA_INDEX = 0
VIDEO_PATH = "sample_video.mp4"
CONFIDENCE = 0.25

WINDOW_NAME = "OBJECTVISION | AI Detection Studio"

DASH_WIDTH = 1280
DASH_HEIGHT = 800
HEADER_HEIGHT = 75
FOOTER_HEIGHT = 38
SIDEBAR_WIDTH = 310
PADDING = 20

BG = (20, 24, 32)
PANEL = (30, 36, 47)
PANEL_LIGHT = (39, 47, 60)
CYAN = (255, 210, 50)
GREEN = (90, 230, 140)
WHITE = (240, 244, 250)
MUTED = (155, 170, 190)
RED = (90, 90, 255)
ORANGE = (0, 165, 255)

# Hand landmark connections (MediaPipe landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ================= UI HELPERS =================
def draw_text(image, text, position, scale=0.55,
              color=WHITE, thickness=1):
    cv2.putText(
        image, str(text), position,
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA
    )


def draw_card(image, x, y, w, h, title, value, accent):
    cv2.rectangle(image, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(image, (x, y), (x + 4, y + h), accent, -1)

    draw_text(image, title.upper(), (x + 15, y + 24),
              0.40, MUTED, 1)
    draw_text(image, value, (x + 15, y + 57),
              0.85, WHITE, 2)


def fit_frame(frame, max_w, max_h):
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(
        frame, (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1
        else cv2.INTER_LINEAR
    )


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw the 21 hand landmarks and their connections."""
    h, w = frame.shape[:2]
    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], GREEN, 2,
                 cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 4, CYAN, -1, cv2.LINE_AA)

    # Index fingertip is landmark number 8.
    tip_x, tip_y = points[8]
    cv2.circle(frame, (tip_x, tip_y), 10, RED, 2, cv2.LINE_AA)
    draw_text(frame, "INDEX TIP", (tip_x + 12, tip_y - 10),
              0.45, WHITE, 1)

    return tip_x, tip_y


def move_target(frame):
    """Create a new game target within the current frame."""
    h, w = frame.shape[:2]
    margin = 45

    x = random.randint(margin, max(margin, w - margin))
    y = random.randint(margin, max(margin, h - margin))

    return x, y


def draw_game(frame, target, score, finger_tip, trail):
    """Draw the finger trail and target-touch mini-game."""
    for i in range(1, len(trail)):
        cv2.line(
            frame, trail[i - 1], trail[i],
            CYAN, max(1, int(5 * i / len(trail))),
            cv2.LINE_AA
        )

    tx, ty = target
    cv2.circle(frame, (tx, ty), 28, ORANGE, 3, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 7, WHITE, -1, cv2.LINE_AA)
    draw_text(frame, f"SCORE: {score}", (18, 35),
              0.85, GREEN, 2)
    draw_text(frame, "Touch the target with your index finger",
              (18, 65), 0.48, WHITE, 1)

    if finger_tip is not None:
        fx, fy = finger_tip
        distance = ((fx - tx) ** 2 + (fy - ty) ** 2) ** 0.5

        if distance < 32:
            return True

    return False


def build_dashboard(frame, fps, counts, tracked_ids,
                    source_name, frame_number, total_frames,
                    hand_status, game_score):
    canvas = np.full(
        (DASH_HEIGHT, DASH_WIDTH, 3), BG, dtype=np.uint8
    )

    # Header
    cv2.rectangle(
        canvas, (0, 0), (DASH_WIDTH, HEADER_HEIGHT), PANEL, -1
    )
    cv2.rectangle(
        canvas, (0, HEADER_HEIGHT - 2),
        (DASH_WIDTH, HEADER_HEIGHT), CYAN, -1
    )

    draw_text(canvas, "OBJECTVISION", (24, 34), 0.90, CYAN, 2)
    draw_text(canvas, "AI DETECTION & HAND TRACKING STUDIO",
              (25, 57), 0.42, MUTED, 1)

    cv2.circle(canvas, (DASH_WIDTH - 185, 31), 6, GREEN, -1)
    draw_text(canvas, "SYSTEM ACTIVE",
              (DASH_WIDTH - 170, 36), 0.48, WHITE, 1)

    # Preview panel
    preview_x = PADDING
    preview_y = HEADER_HEIGHT + PADDING
    preview_w = DASH_WIDTH - SIDEBAR_WIDTH - 3 * PADDING
    preview_h = DASH_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT - 2 * PADDING

    cv2.rectangle(
        canvas, (preview_x, preview_y),
        (preview_x + preview_w, preview_y + preview_h), PANEL, -1
    )

    draw_text(canvas, "LIVE DETECTION PREVIEW",
              (preview_x + 14, preview_y + 24), 0.48, WHITE, 1)
    draw_text(canvas, source_name,
              (preview_x + preview_w - 180, preview_y + 24),
              0.42, GREEN, 1)

    image_y = preview_y + 38
    image_h = preview_h - 50
    image_w = preview_w - 20

    fitted = fit_frame(frame, image_w, image_h)

    if fitted is not None:
        fh, fw = fitted.shape[:2]
        image_x = preview_x + 10 + (image_w - fw) // 2
        image_y_centered = image_y + (image_h - fh) // 2

        canvas[
            image_y_centered:image_y_centered + fh,
            image_x:image_x + fw
        ] = fitted

    # Sidebar
    side_x = DASH_WIDTH - SIDEBAR_WIDTH - PADDING
    side_y = preview_y
    side_w = SIDEBAR_WIDTH
    side_h = preview_h

    cv2.rectangle(
        canvas, (side_x, side_y),
        (side_x + side_w, side_y + side_h), PANEL, -1
    )

    draw_text(canvas, "LIVE STATISTICS",
              (side_x + 16, side_y + 27), 0.60, WHITE, 2)

    total_objects = sum(counts.values())
    card_y = side_y + 42
    card_gap = 9
    card_h = 68
    card_w = side_w - 32
    card_x = side_x + 16

    draw_card(canvas, card_x, card_y, card_w, card_h,
              "Detected objects", total_objects, CYAN)
    card_y += card_h + card_gap

    draw_card(canvas, card_x, card_y, card_w, card_h,
              "Tracking IDs", len(tracked_ids), GREEN)
    card_y += card_h + card_gap

    draw_card(canvas, card_x, card_y, card_w, card_h,
              "Processing speed", f"{fps:.1f} FPS", CYAN)
    card_y += card_h + card_gap

    draw_card(canvas, card_x, card_y, card_w, card_h,
              "Finger game score", game_score, ORANGE)
    card_y += card_h + 15

    draw_text(canvas, f"HAND: {hand_status}",
              (side_x + 16, card_y), 0.45,
              GREEN if hand_status == "DETECTED" else MUTED, 1)
    card_y += 25

    draw_text(canvas, "OBJECT BREAKDOWN",
              (side_x + 16, card_y), 0.52, WHITE, 2)
    card_y += 23

    if counts:
        for name, count in counts.most_common(8):
            if card_y > side_y + side_h - 65:
                break

            draw_text(canvas, name[:23],
                      (side_x + 17, card_y), 0.47, MUTED, 1)
            draw_text(canvas, count,
                      (side_x + side_w - 28, card_y),
                      0.48, GREEN, 2)

            bar_x = side_x + 17
            bar_y = card_y + 8
            bar_w = side_w - 45
            max_count = max(counts.values())
            filled = int(bar_w * count / max_count)

            cv2.rectangle(
                canvas, (bar_x, bar_y),
                (bar_x + bar_w, bar_y + 3), PANEL_LIGHT, -1
            )
            cv2.rectangle(
                canvas, (bar_x, bar_y),
                (bar_x + filled, bar_y + 3), CYAN, -1
            )

            card_y += 27
    else:
        draw_text(canvas, "No objects detected",
                  (side_x + 16, card_y + 8), 0.45, MUTED)

    # Footer
    footer_y = DASH_HEIGHT - FOOTER_HEIGHT
    cv2.rectangle(
        canvas, (0, footer_y),
        (DASH_WIDTH, DASH_HEIGHT), PANEL, -1
    )

    progress = ""
    if total_frames > 0:
        progress = f"  |  Frame: {frame_number}/{total_frames}"

    draw_text(
        canvas,
        f"Q QUIT  |  S SAVE SCREENSHOT  |  Source: {source_name}{progress}",
        (20, footer_y + 24), 0.42, WHITE, 1
    )

    return canvas


# ================= START APPLICATION =================
print("Loading ObjectVision AI Detection Studio...")

model = YOLO(MODEL_NAME)

# Load MediaPipe Tasks hand model only for webcam mode.
print("\nChoose input source:")
print("1. Webcam (object detection + hand tracking + mini-game)")
print("2. Recorded video (object detection + tracking)")

choice = input("Enter your choice (1 or 2): ").strip()

is_webcam = choice != "2"

if is_webcam:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    source_name = "WEBCAM"
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    source_name = "RECORDED VIDEO"

if not cap.isOpened():
    print(f"ERROR: Could not open {source_name.lower()}.")
    raise SystemExit

if is_webcam:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

hand_detector = None

if is_webcam:
    try:
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=HAND_MODEL
            ),
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        hand_detector = vision.HandLandmarker.create_from_options(
            hand_options
        )
        print("Hand tracking model loaded.")
    except Exception as error:
        print(f"Hand tracking could not start: {error}")
        print("Continuing with object detection only.")

total_frames = (
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
)

previous_time = time.time()
fps = 0.0
frame_number = 0
game_score = 0
target = None
trail = deque(maxlen=24)
last_score_time = 0.0

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DASH_WIDTH, DASH_HEIGHT)

print("\nObjectVision started!")
print("Q = quit | S = save dashboard screenshot")

# ================= MAIN LOOP =================
try:
    while True:
        success, frame = cap.read()

        if not success:
            if not is_webcam:
                print("Video finished.")
            else:
                print("Could not read webcam frame.")
            break

        frame_number += 1
        output = frame.copy()

        # YOLO object detection and ByteTrack tracking
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]
        output = result.plot()

        counts = Counter()
        tracked_ids = set()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                counts[model.names[class_id]] += 1

                if box.id is not None:
                    tracked_ids.add(int(box.id[0].item()))

        hand_status = "OFF"
        finger_tip = None

        # MediaPipe Tasks hand tracking (webcam only)
        if hand_detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # VIDEO mode needs a monotonically increasing timestamp.
            timestamp_ms = int(frame_number * 1000 / 30)
            hand_result = hand_detector.detect_for_video(
                mp_image, timestamp_ms
            )

            if hand_result.hand_landmarks:
                hand_status = "DETECTED"

                for landmarks in hand_result.hand_landmarks:
                    tip = draw_hand_landmarks(output, landmarks)

                    # Use the first detected hand to control the game.
                    if finger_tip is None:
                        finger_tip = tip

                if target is None:
                    target = move_target(output)

                trail.append(finger_tip)

                touched = draw_game(
                    output, target, game_score, finger_tip, trail
                )

                now = time.time()

                # Score once per touch, with a short cooldown.
                if touched and now - last_score_time > 0.7:
                    game_score += 1
                    last_score_time = now
                    target = move_target(output)
            else:
                trail.clear()

                if target is None:
                    target = move_target(output)

                draw_game(output, target, game_score, None, trail)

        # FPS calculation
        now = time.time()
        elapsed = now - previous_time

        if elapsed > 0:
            current_fps = 1.0 / elapsed
            fps = current_fps if fps == 0 else (
                0.85 * fps + 0.15 * current_fps
            )

        previous_time = now

        dashboard = build_dashboard(
            output, fps, counts, tracked_ids,
            source_name, frame_number, total_frames,
            hand_status, game_score
        )

        cv2.imshow(WINDOW_NAME, dashboard)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            filename = f"capture_{int(time.time())}.jpg"
            if cv2.imwrite(filename, dashboard):
                print(f"Dashboard screenshot saved: {filename}")
            else:
                print("Screenshot could not be saved.")

finally:
    cap.release()

    if hand_detector is not None:
        hand_detector.close()

    cv2.destroyAllWindows()

print(f"ObjectVision stopped. Final game score: {game_score}")
