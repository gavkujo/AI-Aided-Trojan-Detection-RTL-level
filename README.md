# AI-Aided-Trojan-Detection-RTL-level

### Dependencies:

```bash
pip install pyverilog gensim ultralytics klayout torch networkx
```

### Directory Structure:

```
/project
  ├── rtl/
  │    ├── design.v
  │    └── ast_parser.py
  ├── layout/
  │    ├── layout.gds
  │    └── yolov8_trojan_detector.pt
  └── config/
```

### Key Components:

1. RTL Analysis:
   [x] Converts Verilog to Abstract Syntax Tree (AST)
   [x] Builds directional graph of netlist components
   [x] Generates node embeddings using Word2Vec
2. Layout Inspection:
   [x] Uses Klayout's Python API for GDSII parsing
   [x] Implements YOLOv8 for object detection
   [x] Customizable layer processing

### To Run:

```bash
python hardware_trojan_detection.py \
  --verilog design.v \
  --gdsii layout.gds \
  --model yolov8n_trojan.pt
```

### Sample Output:

```json
{
  "rtl_embeddings": [["XOR", 0.92], ["MUX", 0.87]],
  "layout_anomalies": [
    [[512, 512, 600, 600], 0],  # Rogue via at coordinates
    [[1024, 0, 1200, 200], 2]   # Power grid tampering
  ]
}
```

### Expansion Points:

1. Add a GUI with Gradio/Streamlit for visualization
2. Integrate Synopsys VSO.ai API for verification automation
3. Implement quantum-resistant anomaly detection using lattice-based ML
4. Add 3D-IC analysis using OpenROAD metrics
