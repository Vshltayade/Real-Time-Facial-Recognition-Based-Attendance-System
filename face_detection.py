from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        result = self.model(frame, conf=0.5, verbose=False)[0]
        if result.boxes is None:
            return []
        return result.boxes.xyxy.cpu().numpy()
