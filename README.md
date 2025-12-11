# Distributed Motion Prediction on Waymo Open Dataset

## Overview
Multi-Trajectory Prediction (MTP) pipeline for the Mining Massive Datasets final project. We process the **2.17 TB Waymo Open Motion Dataset** using a hybrid architecture (Cloud Spark + Local MPS) and achieve a **70% reduction in displacement error**.

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
First, configure your GCS credentials, then run:
```bash
python download_data.py
```
This downloads 100 preprocessed batches (~50GB) to `./data/processed_enriched/`.

**Note**: If you don't have GCS access, you can use a subset of sample data or run the preprocessing yourself (see Step 1b).

### Step 1b: Preprocess from Raw (Optional)
If you have access to raw Waymo TFRecords on Dataproc:
```bash
# Submit to Dataproc cluster
gcloud dataproc jobs submit pyspark preprocess_maps.py \
    --cluster=waymo-cluster \
    --region=us-central1 \
    -- --start_batch 0 --end_batch 100
```

### Step 2: Train the Model
```bash
python train_local_mtp.py
```
**Expected output:**
- Training runs for 10 epochs across 100 batches
- Loss decreases from ~1000 to ~650
- Model saved to `mtp_v5_local.pth`

**Hardware**: Automatically detects MPS (Mac) or CUDA (Linux). Falls back to CPU.

### Step 3: Evaluate
```bash
python eval_local_mtp.py
```
**Expected output:**
```
RESULTS (Local MTP - 3 Modes):
minADE: 2.07 meters
minFDE: 2.93 meters
```

### Step 4: Generate Plots (Optional)
```bash
python generate_report_plots.py
```
Creates `model_comparison.png` and `scaling_efficiency.png`.

## Repository Structure
```
├── src/
│   ├── model_mtp.py       # Multi-Trajectory Prediction Network
│   └── model_lstm.py      # Baseline LSTM
├── train_local_mtp.py     # Training script (MPS/CUDA)
├── eval_local_mtp.py      # Evaluation script
├── preprocess_maps.py     # Spark ETL pipeline
├── download_data.py       # GCS data fetcher
├── Final_Report.md        # Research report
└── video_script.md        # Presentation script
```

## Results

| Model | minADE | minFDE | Improvement |
|-------|--------|--------|-------------|
| Constant Velocity | 4.07m | 11.28m | - |
| Single-Mode LSTM | 3.58m | 9.89m | 12% |
| **MTP (Ours)** | **2.07m** | **2.93m** | **70%** |

## License
MIT
