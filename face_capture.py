import os
import cv2
from ultralytics import YOLO

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")
DATA_DIR = os.path.join(BASE_DIR, "data", "faces")
# --------------------------------------

person_name = input("Enter person name: ").strip()
save_dir = os.path.join(DATA_DIR, person_name)
os.makedirs(save_dir, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
count = 0
MAX_IMAGES = 20

print("[INFO] Press SPACE to capture | Q to quit")

while cap.isOpened() and count < MAX_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)[0]

    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == 32 and face.size > 0:  # SPACE
                img_path = os.path.join(save_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face)
                count += 1
                print(f"[SAVED] {img_path}")

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"[DONE] Captured {count} images for {person_name}")
