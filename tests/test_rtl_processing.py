import pytest
from data_processing.rtl_processor import RTLProcessor
from utils.config_manager import ConfigManager

def test_verilog_parsing():
    config = ConfigManager.load_config("configs/paths.yaml")
    processor = RTLProcessor(config['rtl_params'])
    
    # Test with sample Verilog file
    graph = processor.parse_verilog("tests/sample_design.v")
    
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert 'fan_in' in graph.nodes[list(graph.nodes)[0]]
