import cv2
import numpy as np
import onnxruntime as ort
from collections import deque


class AntiSpoof:
    """Anti-spoof detector - matches original behavior exactly"""
    
    def __init__(self, model_path, threshold=0.9):
        self.threshold = threshold
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 128
        
        # ImageNet normalization (commonly used for MiniFASNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Temporal smoothing
        self.history = {}
    
    def preprocess(self, face):
        """ORIGINAL preprocessing - UNCHANGED"""
        # Resize
        face = cv2.resize(face, (self.input_size, self.input_size))
        
        # Convert to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face = face.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        face = (face - self.mean) / self.std
        
        # Transpose to CHW format
        face = np.transpose(face, (2, 0, 1))
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def softmax(self, x):
        """Apply softmax to convert logits to probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def predict(self, face):
        """ORIGINAL predict - UNCHANGED"""
        if face is None or face.size == 0:
            return False, 0.0
        
        inp = self.preprocess(face)
        output = self.session.run(None, {self.input_name: inp})[0]
        
        # Apply softmax to get probabilities
        probs = self.softmax(output[0])
        
        # Index 1 is typically "real", index 0 is "spoof"
        real_score = float(probs[1])
        
        is_real = real_score > self.threshold
        
        return is_real, real_score
    
    def predict_from_bbox(self, frame, bbox, track_id=None):
        """
        Enhanced prediction using EXACT same logic as test_antispoof.py
        """
        # Use test_antispoof.py's safe_crop function
        face_crop, is_quality = safe_crop(frame, bbox[0], bbox[1], bbox[2], bbox[3], pad=0.2)
        
        if face_crop is None or not is_quality:
            return False, 0.0
        
        # Quality check (matching test_antispoof.py)
        quality_score = calculate_face_quality(face_crop)
        if quality_score < 0.3:
            return False, 0.0
        
        # Use original predict method
        is_real, score = self.predict(face_crop)
        
        # Apply temporal smoothing if track_id provided
        if track_id:
            is_real = self._smooth(track_id, is_real)
        
        return is_real, score
    
    def _smooth(self, track_id, is_real):
        """Temporal smoothing matching test_antispoof.py"""
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=15)  # Same as test file
        
        self.history[track_id].append(1 if is_real else 0)
        
        # Calculate confidence based on recent history (last 5 frames)
        if len(self.history[track_id]) >= 5:
            recent_avg = sum(list(self.history[track_id])[-5:]) / 5
            confidence = abs(recent_avg - 0.5) * 2
            
            # Only use smoothed result if confidence is high
            if confidence > 0.6:
                return recent_avg > 0.5
        
        return is_real
    
    def predict_with_debug(self, face):
        """Debug version to check raw outputs"""
        if face is None or face.size == 0:
            return False, 0.0, {}
        
        inp = self.preprocess(face)
        raw_output = self.session.run(None, {self.input_name: inp})[0]
        
        probs = self.softmax(raw_output[0])
        
        debug_info = {
            'raw_output': raw_output[0],
            'probabilities': probs,
            'spoof_score': float(probs[0]),
            'real_score': float(probs[1])
        }
        
        is_real = float(probs[1]) > self.threshold
        
        return is_real, float(probs[1]), debug_info


# ========== EXACT COPIES FROM test_antispoof.py ==========

def safe_crop(frame, x1, y1, x2, y2, pad=0.2):
    """
    EXACT COPY from test_antispoof.py
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
    EXACT COPY from test_antispoof.py
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