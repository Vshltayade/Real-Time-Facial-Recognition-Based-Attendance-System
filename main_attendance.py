import os
import cv2
import pickle
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import mediapipe as mp
from datetime import date
import requests

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_MODEL = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")
ARCFACE_MODEL = os.path.join(BASE_DIR, "models", "glint360k_r50.onnx")
EMBED_DB = os.path.join(BASE_DIR, "data", "embeddings_db.pkl")

# ================= CONFIG =================
RECOGNITION_THRESHOLD = 0.7
REQUIRED_BLINKS = 2
EAR_THRESHOLD = 0.23
CONSEC_FRAMES = 2
# ========================================

# -------- Load models --------
face_detector = YOLO(FACE_MODEL)
arc_sess = ort.InferenceSession(ARCFACE_MODEL, providers=["CPUExecutionProvider"])
input_name = arc_sess.get_inputs()[0].name

with open(EMBED_DB, "rb") as f:
    db = pickle.load(f)

# -------- MediaPipe --------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def preprocess(face):
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face / 255.0 - 0.5) / 0.5
    face = np.transpose(face, (2, 0, 1))
    return face[np.newaxis, :].astype(np.float32)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------- Attendance caches --------
marked_today = set()        # attendance already sent
already_shown = set()       # "already marked" shown once

# -------- Camera --------
cap = cv2.VideoCapture(0)

blink_count = 0
frame_counter = 0
liveness_passed = False

print("[INFO] Blink twice to verify liveness")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------- Blink detection --------
    mesh_results = face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks:
        lm = mesh_results.multi_face_landmarks[0]
        left = np.array([(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in LEFT_EYE])
        right = np.array([(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in RIGHT_EYE])

        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
            frame_counter = 0

        if blink_count >= REQUIRED_BLINKS:
            liveness_passed = True

    # -------- Face detection --------
    results = face_detector(frame, conf=0.5, verbose=False)[0]

    if results.boxes and liveness_passed:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            emb = arc_sess.run(None, {input_name: preprocess(face)})[0][0]

            best_name = "Unknown"
            best_score = 0

            for name, embeddings in db.items():
                for ref in embeddings:
                    score = cosine_sim(emb, ref)
                    if score > best_score:
                        best_score = score
                        best_name = name

            label = "Unknown"

            if best_score >= RECOGNITION_THRESHOLD:
                # -------- Send attendance --------
                if best_name not in marked_today:
                    try:
                        response = requests.post(
                            "http://127.0.0.1:5000/mark_attendance",
                            json={"name": best_name},
                            timeout=1
                        )

                        msg = response.json().get("message", "")

                        if "already" in msg.lower():
                            if best_name not in already_shown:
                                print(f"[INFO] Attendance already marked for {best_name}")
                                already_shown.add(best_name)
                        else:
                            print(f"[INFO] Attendance sent for {best_name}")
                            marked_today.add(best_name)

                    except:
                        print("[ERROR] Backend not reachable")

                else:
                    if best_name not in already_shown:
                        print(f"[INFO] Attendance already marked for {best_name}")
                        already_shown.add(best_name)

                # -------- Label text --------
                if best_name in marked_today:
                    label = f"{best_name} - Marked"
                elif best_name in already_shown:
                    label = f"{best_name} - Already Marked"
                else:
                    label = f"{best_name} ({best_score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # -------- UI Overlay --------
    cv2.putText(frame, f"Blinks: {blink_count}/{REQUIRED_BLINKS}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    if not liveness_passed:
        cv2.putText(frame, "Blink twice to verify liveness",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

