import mediapipe as mp
import cv2
import numpy as np

class BlinkLiveness:
    def __init__(self, blinks_required=2, ear_thresh=0.21):
        self.blinks_required = blinks_required
        self.ear_thresh = ear_thresh
        self.blink_count = 0
        self.blink_state = False

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )

        self.LEFT = [33, 160, 158, 133, 153, 144]
        self.RIGHT = [362, 385, 387, 263, 373, 380]

    def _ear(self, lm, eye):
        p = [np.array([lm[i].x, lm[i].y]) for i in eye]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h)

    def update(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return False

        lm = res.multi_face_landmarks[0].landmark
        ear = (self._ear(lm, self.LEFT) + self._ear(lm, self.RIGHT)) / 2

        if ear < self.ear_thresh and not self.blink_state:
            self.blink_state = True
        elif ear >= self.ear_thresh and self.blink_state:
            self.blink_state = False
            self.blink_count += 1

        return self.blink_count >= self.blinks_required
