import klayout.db as kdb
import numpy as np
import torch
from PIL import Image
from ai_models.yolo_model import YOLOTrojanDetector

class LayoutProcessor:
    def __init__(self, config):
        self.layout = kdb.Layout()
        self.config = config
        
        # Initialize YOLO model
        yolo_model_path = config.get('yolo_model', 'models/yolo/best.pt')
        confidence_thresh = config.get('confidence_thresh', 0.7)
        iou_thresh = config.get('iou_thresh', 0.4)
        self.detector = YOLOTrojanDetector(
            yolo_model_path, 
            confidence_thresh=confidence_thresh,
            iou_thresh=iou_thresh
        )
        
        # Layer mapping
        self.layer_mapping = config.get('layer_mapping', {
            'METAL1': (1, 0),
            'METAL2': (2, 0),
            'VIA': (3, 0)
        })
        
        # Default class map if not provided in config
        self.class_map = config.get('class_map', {
            0: "RogueVia",
            1: "UnauthorizedFiller",
            2: "PowerTamper",
            3: "ThermalHotspot"
        })

    def load_gdsii(self, file_path):
        """Load GDSII layout file with error handling"""
        try:
            self.layout.read(file_path)
            return True
        except Exception as e:
            print(f"GDSII Loading Error: {str(e)}")
            return False

    def render_layer_to_image(self, layer_id, pixel_size=0.005):
        """Convert GDSII layer to grayscale image tensor"""
        cell = self.layout.top_cell()
        layer = self.layout.layer(layer_id, 0)
        
        # Calculate bounding box
        bbox = cell.bbox_per_layer(layer)
        if bbox.empty():
            # Return empty image if no shapes in layer
            return Image.new('RGB', (100, 100), color='black')
            
        width = max(1, int(bbox.width() / pixel_size))
        height = max(1, int(bbox.height() / pixel_size))
        
        # Create image buffer
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Render shapes
        for shape in cell.shapes(layer):
            if shape.is_box():
                box = shape.box
                x1 = int((box.left - bbox.left) / pixel_size)
                y1 = int((box.bottom - bbox.bottom) / pixel_size)
                x2 = int((box.right - bbox.left) / pixel_size)
                y2 = int((box.top - bbox.bottom) / pixel_size)
                img[y1:y2, x1:x2] = 255

        return Image.fromarray(img).convert('RGB')

    def detect_anomalies(self, layer_id=None):
        """Run YOLO detection on specified layer or all layers"""
        if layer_id is None:
            # Process all layers if none specified
            all_anomalies = []
            for layer_name, (layer_id, _) in self.layer_mapping.items():
                anomalies = self._detect_layer_anomalies(layer_id, layer_name)
                all_anomalies.extend(anomalies)
            return all_anomalies
        else:
            # Process specific layer
            return self._detect_layer_anomalies(layer_id)

    def _detect_layer_anomalies(self, layer_id, layer_name=None):
        """Detect anomalies in a specific layer"""
        img = self.render_layer_to_image(layer_id)
        detections = self.detector.predict(img)
        
        # Add layer information to detections
        if layer_name:
            for detection in detections:
                detection['layer'] = layer_name
                
        return detections
