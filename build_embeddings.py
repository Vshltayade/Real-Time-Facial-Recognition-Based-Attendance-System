# ==================================
# BUILD ARCFACE EMBEDDINGS (FINAL)
# ==================================

import os
import cv2
import pickle
import numpy as np
import onnxruntime as ort

# -------------------------------
# ABSOLUTE PATH SETUP (FINAL FIX)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACES_DIR = os.path.join(BASE_DIR, "data", "faces")
OUTPUT_DB = os.path.join(BASE_DIR, "data", "embeddings_db.pkl")
ARCFACE_MODEL = os.path.join(BASE_DIR, "models", "glint360k_r50.onnx")

IMAGE_SIZE = 112

# -------------------------------
# SAFETY CHECKS
# -------------------------------
if not os.path.exists(FACES_DIR):
    raise RuntimeError(f"Faces directory not found: {FACES_DIR}")

if not os.path.exists(ARCFACE_MODEL):
    raise RuntimeError(f"ArcFace model not found: {ARCFACE_MODEL}")

os.makedirs(os.path.dirname(OUTPUT_DB), exist_ok=True)

# -------------------------------
# LOAD ARCFACE MODEL
# -------------------------------
session = ort.InferenceSession(
    ARCFACE_MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess(face_img):
    face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = (face_img - 0.5) / 0.5
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# -------------------------------
# BUILD EMBEDDINGS DATABASE
# -------------------------------
embeddings_db = {}
total_faces = 0

for person_name in sorted(os.listdir(FACES_DIR)):
    person_path = os.path.join(FACES_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    person_embeddings = []

    for img_name in sorted(os.listdir(person_path)):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        blob = preprocess(img)
        embedding = session.run(None, {input_name: blob})[0][0]

        embedding = embedding / np.linalg.norm(embedding)
        person_embeddings.append(embedding)
        total_faces += 1

    if person_embeddings:
        embeddings_db[person_name] = person_embeddings

# -------------------------------
# SAVE DATABASE (GUARANTEED)
# -------------------------------
with open(OUTPUT_DB, "wb") as f:
    pickle.dump(embeddings_db, f)

print(f"[DONE] Saved embeddings_db.pkl")
print(f"[INFO] Identities: {len(embeddings_db)} | Faces: {total_faces}")
print(f"[PATH] {OUTPUT_DB}")
