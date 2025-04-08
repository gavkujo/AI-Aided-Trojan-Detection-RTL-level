import pytest
from PIL import Image
from data_processing.layout_processor import LayoutProcessor

@pytest.fixture
def sample_config():
    return {
        "yolo_model": "models/yolo.pt",
        "confidence_thresh": 0.7,
        "iou_thresh": 0.4,
        "layer_mapping": {"METAL1": (1, 0)}
    }

def test_gdsii_loading(sample_config):
    processor = LayoutProcessor(sample_config)
    assert processor.load_gdsii("tests/sample.gds") is True

def test_anomaly_detection(sample_config, mocker):
    processor = LayoutProcessor(sample_config)
    mocker.patch.object(processor, 'render_layer_to_image', return_value=Image.new('RGB', (100, 100)))
    mocker.patch.object(processor.detector, 'predict', return_value=[])
    
    anomalies = processor.detect_anomalies(1)
    assert isinstance(anomalies, list)
