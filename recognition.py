import os
import cv2
import pickle
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import mediapipe as mp
import requests
from collections import deque

from antispoof_check import AntiSpoof

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")
ARCFACE_MODEL = os.path.join(BASE_DIR, "models", "glint360k_r50.onnx")
ANTI_SPOOF_MODEL = os.path.join(BASE_DIR, "models", "antispoof", "best_model.onnx")
EMBED_DB = os.path.join(BASE_DIR, "data", "embeddings_db.pkl")

# ================= CONFIG =================
RECOGNITION_THRESHOLD = 0.7
REQUIRED_BLINKS = 2
EAR_THRESHOLD = 0.23
CONSEC_FRAMES = 2
SPOOF_WINDOW = 40
SPOOF_THRESHOLD = 0.5
DEBUG_MODE = True  # ← SET TO FALSE TO DISABLE DEBUG
# ========================================

# ================= STATES =================
STATE_SPOOF = 0
STATE_BLINK = 1
STATE_RECOG = 2
state = STATE_SPOOF
# ========================================

# -------- Load models --------
face_detector = YOLO(FACE_MODEL)

arc_sess = ort.InferenceSession(
    ARCFACE_MODEL,
    providers=["CPUExecutionProvider"]
)
arc_input = arc_sess.get_inputs()[0].name

antispoof_sess = ort.InferenceSession(
    ANTI_SPOOF_MODEL,
    providers=["CPUExecutionProvider"]
)
antispoof_input = antispoof_sess.get_inputs()[0].name
antispoof_output = antispoof_sess.get_outputs()[0].name

# Enhanced spoof detector with LOWER threshold
antispoof = AntiSpoof(
    ANTI_SPOOF_MODEL,
    threshold=0.7  # ← LOWERED from 0.9 to 0.7
)

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

# -------- Runtime state --------
spoof_votes = deque(maxlen=SPOOF_WINDOW)
blink_count = 0
frame_counter = 0
liveness_passed = False
current_track_id = "face_1"  # ← ADDED for temporal smoothing
frame_count = 0  # ← ADDED for debug

cap = cv2.VideoCapture(0)
print("[INFO] System started with enhanced spoof detection")
if DEBUG_MODE:
    print("[DEBUG] Debug mode ENABLED - watch console output")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detector(frame, conf=0.5, verbose=False)[0]
    if not results.boxes:
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    box = results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        continue

    # ================= STATE 0: ENHANCED SPOOF DETECTION =================
    if state == STATE_SPOOF:

        # DEBUG: Test both methods for comparison
        if DEBUG_MODE and frame_count % 30 == 0:  # Every 30 frames
            # Test 1: Original predict with YOLO crop
            face_yolo = frame[y1:y2, x1:x2]
            is_real_direct, score_direct = antispoof.predict(face_yolo)
            
            # Test 2: Enhanced predict_from_bbox
            from antispoof_check import safe_crop
            face_enhanced, is_quality = safe_crop(frame, x1, y1, x2, y2, pad=0.2)
            if face_enhanced is not None and is_quality:
                is_real_enhanced, score_enhanced = antispoof.predict(face_enhanced)
            else:
                is_real_enhanced, score_enhanced = False, 0.0
            
            print(f"\n[DEBUG Frame {frame_count}]")
            print(f"  YOLO bbox: {x2-x1}x{y2-y1}px")
            print(f"  YOLO crop result: {'REAL' if is_real_direct else 'SPOOF'} ({score_direct:.3f})")
            if face_enhanced is not None:
                print(f"  Enhanced crop size: {face_enhanced.shape}")
                print(f"  Enhanced crop result: {'REAL' if is_real_enhanced else 'SPOOF'} ({score_enhanced:.3f})")
            else:
                print(f"  Enhanced crop: REJECTED (too small/blurry)")

        # Enhanced spoof detection matching test_antispoof.py
        is_real, score = antispoof.predict_from_bbox(
            frame,
            (x1, y1, x2, y2),
            track_id=current_track_id
        )

        spoof_votes.append(1 if is_real else 0)

        label = "REAL" if is_real else "SPOOF"
        color = (0, 255, 0) if is_real else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Enhanced display
        cv2.putText(
            frame,
            f"{label} ({score:.3f})",  # ← More precision
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Show real/spoof ratio
        real_count = spoof_votes.count(1)
        spoof_count = spoof_votes.count(0)
        cv2.putText(
            frame,
            f"Verifying {len(spoof_votes)}/{SPOOF_WINDOW} (R:{real_count} S:{spoof_count})",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        if len(spoof_votes) == SPOOF_WINDOW:
            if spoof_votes.count(0) > spoof_votes.count(1):
                cv2.putText(
                    frame,
                    "SPOOF CONFIRMED - Access Denied",
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                print(f"[INFO] Spoof detected: {spoof_count}/{SPOOF_WINDOW} spoof votes")
                cv2.imshow("Recognition", frame)
                cv2.waitKey(2000)
                break
            else:
                print(f"[INFO] Spoof check passed: {real_count}/{SPOOF_WINDOW} real votes")
                state = STATE_BLINK

    # ================= STATE 1: BLINK =================
    elif state == STATE_BLINK:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            f"Blink twice ({blink_count}/{REQUIRED_BLINKS})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        mesh = face_mesh.process(rgb)
        if mesh.multi_face_landmarks:
            lm = mesh.multi_face_landmarks[0]
            left = np.array([(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in LEFT_EYE])
            right = np.array([(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in RIGHT_EYE])
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2

            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                    print(f"[INFO] Blink {blink_count}/{REQUIRED_BLINKS} detected")
                frame_counter = 0

            if blink_count >= REQUIRED_BLINKS:
                print("[INFO] Liveness verified - proceeding to recognition")
                state = STATE_RECOG

    # ================= STATE 2: RECOGNITION =================
    elif state == STATE_RECOG:
        emb = arc_sess.run(None, {arc_input: preprocess(face)})[0][0]
        best_name, best_score = "Unknown", 0

        for name, embeds in db.items():
            for ref in embeds:
                s = cosine_sim(emb, ref)
                if s > best_score:
                    best_score, best_name = s, name

        label = best_name if best_score >= RECOGNITION_THRESHOLD else "Unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] System shutdown")