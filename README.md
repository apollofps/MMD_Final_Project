# Distributed Motion Prediction on Waymo Open Dataset

## Overview
Multi-Trajectory Prediction (MTP) pipeline for the Mining Massive Datasets final project. We process the **2.17 TB Waymo Open Motion Dataset** using a hybrid architecture (Cloud Spark + Local MPS) and achieve a **70% reduction in displacement error**.

## Repository Structure
```
├── src/                      # All source code
│   ├── model_mtp.py          # Multi-Trajectory Prediction Network
│   ├── model_lstm.py         # Baseline LSTM
│   ├── train_local_mtp.py    # Training script (MPS/CUDA)
│   ├── eval_local_mtp.py     # Evaluation script
│   ├── preprocess_maps.py    # Spark ETL pipeline
│   ├── download_data.py      # GCS data fetcher
│   └── generate_report_plots.py
├── output/                   # Results and plots
│   ├── model_comparison.png
│   └── scaling_efficiency.png
├── Final_Report.md           # Research report
├── run_local_mtp_pipeline.sh # Pipeline runner
└── video_script.md           # Presentation script
```

## Quick Start

### Prerequisites
- Python 3.10+
- PyTorch with MPS (Mac) or CUDA (Linux)
- Google Cloud SDK (for data download)

### Installation
```bash
git clone https://github.com/apollofps/MMD_Final_Project.git
cd MMD_Final_Project
pip install torch pandas numpy matplotlib seaborn
```

## Running the Pipeline

### Step 1: Download Data
```bash
python src/download_data.py
```

### Step 2: Train the Model
```bash
python src/train_local_mtp.py
```

### Step 3: Evaluate
```bash
python src/eval_local_mtp.py
```
**Expected output:**
```
minADE: 2.07 meters
minFDE: 2.93 meters
```

### Run Full Pipeline
```bash
./run_local_mtp_pipeline.sh
```

## Results

| Model | minADE | minFDE | Improvement |
|-------|--------|--------|-------------|
| Constant Velocity | 4.07m | 11.28m | - |
| Single-Mode LSTM | 3.58m | 9.89m | 12% |
| **MTP (Ours)** | **2.07m** | **2.93m** | **70%** |

## License
MIT
