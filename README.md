# Hardware Trojan Detection Using AI for RTL and GDSII Designs

---

## **Overview**
This project addresses the growing threat of hardware Trojans—malicious modifications to chip designs—by leveraging Artificial Intelligence (AI) to detect anomalies in Register Transfer Level (RTL) code and post-silicon physical layouts (GDSII). The tool is designed to integrate seamlessly into semiconductor design workflows, providing a robust pre-silicon security checkpoint.

By combining **Graph Neural Networks (GNNs)** and **anomaly detection techniques**, the system identifies suspicious patterns in RTL designs and physical layouts, helping engineers mitigate risks before fabrication. The solution is tailored for IC design houses, foundries, and defense organizations, with a focus on securing Taiwan's semiconductor ecosystem.

---

## **Features**
1. **Graph-Based RTL Modeling**:
   - Converts Verilog/VHDL designs into graph representations.
   - Nodes represent logic elements (e.g., modules, gates), and edges represent connections.
   - Enables AI-based pattern recognition of Trojan circuitry.

2. **AI Anomaly Detection Engine**:
   - Combines supervised learning (GNN-based classifier) and unsupervised anomaly detection.
   - Flags rarely-triggered logic, unexpected signal dependencies, and other suspicious patterns.

3. **Physical Layout Analysis**:
   - Uses YOLOv8 for detecting anomalies in GDSII layouts (e.g., rogue vias, power grid tampering).
   - Integrates with open-source tools like Klayout for layout parsing.

4. **Trojan Test Case Library**:
   - Includes Trust-Hub benchmarks and custom Trojan-injected RTL designs for training and evaluation.

5. **Integration with Design Flows**:
   - Outputs detailed reports highlighting suspect modules or signals with risk scores.
   - Fits into existing EDA workflows as a security linting step.

---

## **Project Structure**
```
hardware_trojan_detector/
├── data_processing/
│   ├── rtl_processor.py       # Parses RTL files and builds graph representations
│   ├── layout_processor.py    # Analyzes GDSII layouts for physical anomalies
│   └── graph_builder.py       # Converts AST into graph structures
├── ai_models/
│   ├── gnn_model.py           # Implements Graph Neural Network for RTL analysis
│   ├── yolo_model.py          # YOLOv8-based detection for GDSII layouts
│   └── anomaly_detector.py    # Anomaly detection models (autoencoder, isolation forest)
├── utils/
│   ├── logger.py              # Logging utility
│   └── config_manager.py      # Manages configuration files
├── configs/
│   ├── paths.yaml             # Paths to data, models, and parameters
│   └── model_params.yaml      # Model-specific hyperparameters
├── tests/
│   ├── test_rtl_processing.py # Unit tests for RTL processing
│   └── test_layout_processing.py # Unit tests for layout processing
├── scripts/
│   ├── train_model.py         # Script to train AI models
│   └── run_detection.py       # Main script to perform Trojan detection
├── Dockerfile                 # Docker container definition
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## **Getting Started**

### **1. Prerequisites**
- Python 3.8+
- NVIDIA CUDA Toolkit (for GPU acceleration)
- Docker (optional for containerized execution)
- Install Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### **2. Prepare Your RTL Project**
Place your RTL design files in a directory (e.g., `rtl_project/`):
```
rtl_project/
├── top_module.v
├── crypto_core.v
├── communication_interface.v
```

Update `configs/paths.yaml` to point to your RTL directory:
```yaml
paths:
  rtl_samples: "rtl_project/"
  model_weights: "models/pretrained/gnn_model.pth"
rtl_params:
  embedding_size: 128
  window_size: 5
layout_params:
  yolo_model_path: "models/yolo/best.pt"
```

---

## **How to Run**

### **1. Train Models**
Train the Graph Neural Network (GNN) or YOLO model using the provided training script:
```bash
python scripts/train_model.py --config configs/model_params.yaml
```

This will:
- Parse RTL files into graph representations.
- Train the GNN on known Trojan examples.
- Save the trained model weights to `models/pretrained/gnn_model.pth`.

### **2. Perform Detection**
Run Trojan detection on your RTL project or GDSII layout files:

#### For RTL Files:
```bash
python scripts/run_detection.py \
    --rtl rtl_project/top_module.v \
    --config configs/paths.yaml
```

#### For GDSII Files:
```bash
python scripts/run_detection.py \
    --gdsii layout.gds \
    --config configs/paths.yaml
```

### **3. Batch Processing**
To analyze all RTL files in a directory:
```bash
for file in rtl_project/*.v; do
    python scripts/run_detection.py --rtl $file --config configs/paths.yaml;
done;
```

---

## **Outputs**
The tool generates detailed reports highlighting potential Trojans:

#### Example Output:
```plaintext
[INFO] Processing RTL file: rtl_project/top_module.v...
[INFO] Detected potential Trojan in module 'crypto_core':
       - Trigger Condition: Rarely activated always block at line 120.
       - Risk Score: HIGH

[INFO] Processing GDSII file: layout.gds...
[INFO] Detected anomalies in layer METAL1:
       - Class 'RogueVia', Confidence: 0.92, Bounding Box: [512, 256, 600, 320]
       - Class 'PowerTamper', Confidence: 0.87, Bounding Box: [1024, 512, 1200, 640]

[INFO] Detection completed. Report saved to reports/detection_report.txt.
```

The report (`reports/detection_report.txt`) includes flagged modules/signals with explanations.

---

## **Testing**
Run unit tests to validate individual components:
```bash
pytest tests/ -v
```

---

## **Using Docker**
To run the project in a containerized environment:

### Build Docker Image:
```bash
docker build -t hardware_trojan_detector .
```

### Run Container:
```bash
docker run --rm -it \
    -v $(pwd)/samples:/app/samples \
    -v $(pwd)/configs:/app/configs \
    hardware_trojan_detector \
    python scripts/run_detection.py \
        --rtl samples/top_module.v \
        --config configs/paths.yaml
```

---

## **Key Features**
- **Graph Neural Network (GNN)**: Detects structural anomalies in RTL designs.
- **YOLOv8 Layout Analysis**: Identifies physical anomalies in GDSII files.
- **Trojan Test Case Library**: Includes Trust-Hub benchmarks and custom examples.
- **Seamless Integration**: Fits into existing EDA workflows as a security checkpoint.

---

## **Target Users**
1. **IC Design Houses**: Vet third-party IP blocks during the design phase.
2. **Semiconductor Foundries**: Ensure customer-provided designs are Trojan-free before fabrication.
3. **Defense Organizations**: Evaluate chip designs for critical infrastructure projects.

---

## **Future Work**
- Expand the Trojan library with more real-world examples.
- Integrate formal verification techniques for flagged regions.
- Optimize the GNN architecture for larger designs.

---

## **Contributing**
We welcome contributions! Please follow these steps:
1. Fork the repository on GitHub.
2. Create a new branch (`feature/new-feature`).
3. Commit changes and submit a pull request.

For questions or feedback, please contact us at [garv001@e.ntu.edu.sg], [ygong005@e.ntu.edu.sg].

---

### Authors
1. Garv Sachdev [garv001@e.ntu.edu.sg]
2. 

## **License**
This project is licensed under the MIT License.
