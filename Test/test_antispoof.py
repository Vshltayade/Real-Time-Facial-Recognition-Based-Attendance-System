import sys
import os

#Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import cv2
import numpy as np
from collections import deque

from antispoof_check import AntiSpoof
from face_detection import FaceDetector


YOLO_MODEL_PATH = "models/yolov8n-face.pt"
ANTISPOOF_MODEL_PATH = "models/antispoof/best_model.onnx"


def safe_crop(frame, x1, y1, x2, y2, pad=0.2):
    """
    Improved face cropping with proper aspect ratio handling
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Add padding
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    face = frame[y1:y2, x1:x2]
    
    if face.size == 0 or face.shape[0] < 50 or face.shape[1] < 50:
        return None, False

    # Make square by center-cropping to maintain aspect ratio
    fh, fw = face.shape[:2]
    size = min(fh, fw)
    
    # Center crop
    y_start = (fh - size) // 2
    x_start = (fw - size) // 2
    face_square = face[y_start:y_start + size, x_start:x_start + size]
    
    # Check if face is large enough (quality check)
    is_quality = size >= 80
    
    return face_square, is_quality


def calculate_face_quality(face_crop):
    """
    Calculate face quality metrics
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    # Check sharpness using Laplacian variance
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize (100+ is sharp, <50 is blurry)
    sharpness_score = min(laplacian_var / 100.0, 1.0)
    
    return sharpness_score


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    detector = FaceDetector(YOLO_MODEL_PATH)
    antispoof = AntiSpoof(ANTISPOOF_MODEL_PATH, threshold=0.9)

    # Separate buffers for real and spoof detections
    detection_history = deque(maxlen=15)
    
    # Statistics
    frame_count = 0
    spoof_count = 0
    real_count = 0

    print("=== Anti-Spoof Testing Started ===")
    print("Press 'q' to quit")
    print("Press 'r' to reset statistics")
    print("Press 'd' to toggle debug mode")
    
    debug_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces = detector.detect(frame)

        for (x1, y1, x2, y2) in faces:
            face_crop, is_quality = safe_crop(frame, x1, y1, x2, y2, pad=0.2)
            
            if face_crop is None:
                continue
            
            # Quality check
            quality_score = calculate_face_quality(face_crop)
            
            # Get prediction
            is_real, score = antispoof.predict(face_crop)
            
            # Add to history
            detection_history.append(1 if is_real else 0)
            
            # Calculate confidence based on recent history
            if len(detection_history) >= 5:
                recent_avg = sum(list(detection_history)[-5:]) / 5
                confidence = abs(recent_avg - 0.5) * 2  # 0-1 scale
            else:
                confidence = 0.5
            
            # Determine final label with quality consideration
            if quality_score < 0.3:
                label = "LOW QUALITY"
                color = (0, 165, 255)  # Orange
            elif is_real and confidence > 0.6:
                label = "REAL"
                color = (0, 255, 0)
                real_count += 1
            elif not is_real and confidence > 0.6:
                label = "SPOOF"
                color = (0, 0, 255)
                spoof_count += 1
            else:
                label = "UNCERTAIN"
                color = (255, 255, 0)  # Yellow

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Display information
            if debug_mode:
                cv2.putText(frame, f"{label}", (int(x1), int(y1) - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Score: {score:.3f}", (int(x1), int(y1) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Quality: {quality_score:.2f} Conf: {confidence:.2f}",
                           (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                cv2.putText(frame, f"{label} ({score:.2f})", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display statistics
        cv2.putText(frame, f"Frame: {frame_count} | Real: {real_count} | Spoof: {spoof_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Anti-Spoof Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            # Reset statistics
            frame_count = 0
            spoof_count = 0
            real_count = 0
            detection_history.clear()
            print("Statistics reset")
        elif key == ord("d"):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n=== Final Statistics ===")
    print(f"Total Frames: {frame_count}")
    print(f"Real Detections: {real_count}")
    print(f"Spoof Detections: {spoof_count}")


if __name__ == "__main__":
   main()