# Distributed Motion Prediction on Waymo Open Dataset

## Overview
This repository contains the implementation of a **Multi-Trajectory Prediction (MTP)** pipeline for the Mining Massive Datasets (MMD) final project. We address the challenge of training deep learning models on the **2.17 TB Waymo Open Motion Dataset** using a constrained hybrid architecture (Cloud Spark + Edge Compute).

Our approach achieves a **70% reduction in Final Displacement Error (FDE)** compared to physics baselines by leveraging a multi-modal output head trained with a Winner-Takes-All loss function.

## Architecture
The system is designed as a decoupled pipeline:
1.  **Distributed ETL (Google Cloud Dataproc)**: 
    *   Apache Spark Micro-batching to process raw TFRecords.
    *   Map feature extraction (Lane centerlines & geometry).
    *   Optimized Parquet serialization (60% storage reduction).
2.  **Edge Training (Local M4 Silicon)**:
    *   Iterative Streaming of Parquet batches to constant RAM.
    *   Metal Performance Shaders (MPS) for accelerated PyTorch training.

## Repository Structure
```
.
├── src/
│   ├── model_mtp.py       # Multi-Trajectory Prediction Network (PyTorch)
│   └── model_lstm.py      # Baseline LSTM Network
├── Final_Report.md        # Detailed Research Report
├── train_local_mtp.py     # Main Training Script (MPS-Accelerated)
├── eval_local_mtp.py      # Evaluation Script (minADE/minFDE)
├── preprocess_maps.py     # PySpark ETL Script
├── download_data.py       # Data Fetching Utility
└── requirements.txt       # Dependencies
```

## Setup & Installation

### Prerequisites
*   Python 3.10+
*   PyTorch (MPS enabled for Mac, or CUDA for Linux)
*   Apache Spark (for ETL only)

### Installation
```bash
pip install torch pandas numpy pyspark matplotlib
```

## Usage

### 1. Data Processing (Cloud)
To run the extraction pipeline on Dataproc:
```bash
python preprocess_maps.py --start_batch 0 --end_batch 100
```
*Note: This requires GCS bucket access configured in the script.*

### 2. Training (Local)
Train the MTP model on the downloaded batches:
```bash
python train_local_mtp.py
```
This script will automatically detect MPS/CUDA and start streaming data from `./data/processed_enriched`.

### 3. Evaluation
Run the validation on the unseen test set (Batch 50):
```bash
python eval_local_mtp.py
```
Expected output:
> minADE: ~2.07m  
> minFDE: ~2.93m

## Results
We compared our MTP approach against Single-Mode LSTM and constant velocity baselines.

| Model | minADE (m) | minFDE (m) |
|-------|------------|------------|
| Constant Velocity | 4.07 | 11.28 |
| Single-Mode LSTM | 3.58 | 9.89 |
| **MTP (Ours)** | **2.07** | **2.93** |

See `Final_Report.md` for the complete ablation study and scaling analysis.

## License
MIT
