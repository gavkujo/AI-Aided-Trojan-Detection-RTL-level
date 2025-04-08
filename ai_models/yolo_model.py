import torch
from ultralytics import YOLO
from PIL import Image

class YOLOTrojanDetector:
    def __init__(self, model_path, confidence_thresh=0.7, iou_thresh=0.4):
        self.model = YOLO(model_path)
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.class_map = {
            0: "RogueVia",
            1: "UnauthorizedFiller",
            2: "PowerTamper",
            3: "ThermalHotspot"
        }

    def preprocess(self, image):
        """Convert image to YOLO input format"""
        return Image.fromarray(image).convert('RGB')

    def predict(self, image):
        """Run inference on layout image"""
        results = self.model.predict(
            image,
            conf=self.confidence_thresh,
            iou=self.iou_thresh,
            imgsz=1024
        )
        return self.postprocess(results[0])

    def postprocess(self, result):
        """Format detection results"""
        detections = []
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            detections.append({
                "bbox": box.cpu().numpy().tolist(),
                "class": self.class_map[int(cls)],
                "confidence": float(conf)
            })
        return detections
