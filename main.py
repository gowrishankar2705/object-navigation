import time
import cv2
import pyttsx3
import gc
from ultralytics import YOLO

# --- Configuration ---
DETECTION_INTERVAL = 5.0  # seconds between voice updates
CAMERA_ID = 0              # default webcam index
MODEL_NAME = "yolov8n.pt"  # change to your custom model if needed
KNOWN_WIDTH = 0.5          # meters (e.g., average width of a person)
FOCAL_LENGTH = 800         # pixels (calibrated for your camera)

# Initialize YOLOv8 model
model = YOLO(MODEL_NAME)

# Reliable TTS function
def speak_reliable(text):
    try:
        gc.collect()
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        print(f"[SPEAKING]: {text}")
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
        gc.collect()
    except Exception as e:
        print(f"Speech error: {e}")

# Direction logic
def get_direction(cx, frame_width):
    third = frame_width / 3
    if cx < third:
        return "to your left"
    elif cx > 2 * third:
        return "to your right"
    else:
        return "ahead of you"

# Distance estimation logic
def estimate_distance(known_width, focal_length, pixel_width):
    if pixel_width == 0:
        return None
    return (known_width * focal_length) / pixel_width

# Initialize camera
cap = cv2.VideoCapture(CAMERA_ID)
last_announce = time.time() - DETECTION_INTERVAL
announcement_count = 0

speak_reliable("Navigation system ready")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            speak_reliable("Camera error")
            break

        results = model(frame, verbose=False)[0]
        current_time = time.time()

        # Time to announce?
        if current_time - last_announce >= DETECTION_INTERVAL:
            last_announce = current_time
            announcement_count += 1
            messages = []

            for box in results.boxes:
                if len(box.xyxy) > 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if confidence > 0.5:
                        label = model.names[cls_id]
                        cx = (x1 + x2) / 2
                        direction = get_direction(cx, frame.shape[1])
                        box_width = x2 - x1
                        distance = estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)

                        if distance:
                            messages.append(f"I see a {label} {direction}, approximately {distance:.1f} meters away.")
                        else:
                            messages.append(f"I see a {label} {direction}, but distance is unclear.")

            if messages:
                announcement = f"Announcement {announcement_count}: " + " ".join(messages)
            else:
                announcement = f"Announcement {announcement_count}: I do not see any objects right now."

            speak_reliable(announcement)

        # Draw detections
        for box in results.boxes:
            if len(box.xyxy) > 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence > 0.5:
                    label = model.names[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay announcement count
        cv2.putText(frame, f"Announcements: {announcement_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Smart Glasses View", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            speak_reliable("Goodbye")
            break
        elif key == ord('t'):
            speak_reliable(f"Test number {announcement_count + 1}")

finally:
    cap.release()
    cv2.destroyAllWindows()