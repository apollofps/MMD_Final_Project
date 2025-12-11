import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os
from src.model_mtp import MotionMTP

def calculate_mtp_metrics(pred_trajs, pred_logits, gt_traj, valid_mask):
    """
    Calculate minADE and minFDE (Oracle metrics).
    pred_trajs: (batch, 3, 80, 2)
    gt_traj: (batch, 80, 2)
    """
    batch_size, num_modes, steps, _ = pred_trajs.shape
    
    # Expand GT: (batch, 1, 80, 2)
    gt_expand = gt_traj.unsqueeze(1)
    
    # Diff: (batch, 3, 80, 2)
    diff = pred_trajs - gt_expand
    # Dist: (batch, 3, 80)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=3))
    
    # Mask
    mask_expand = valid_mask.unsqueeze(1)
    dist = dist * mask_expand
    
    # ADE for each mode: (batch, 3)
    # Sum over time / Count valid steps
    valid_steps = torch.sum(valid_mask, dim=1).unsqueeze(1) # (batch, 1)
    mode_ade = torch.sum(dist, dim=2) / (valid_steps + 1e-6)
    
    # minADE: (batch,)
    min_ade, _ = torch.min(mode_ade, dim=1)
    
    # FDE for each mode: (batch, 3)
    # Get last valid index
    # (Simplified: assume last step is 79, else we need advanced indexing)
    # Waymo data usually has valid mask for full future or not.
    # We'll take the simple approach: last step distance.
    last_dist = dist[:, :, -1] 
    min_fde, _ = torch.min(last_dist, dim=1)
    
    return torch.mean(min_ade).item(), torch.mean(min_fde).item()

def eval_mtp():
    print("=========================================")
    print("   Local MTP Evaluation (MPS) 🔮         ")
    print("=========================================")
    
    BATCH_SIZE = 512
    # Evaluate on Batch 50 (Unseen test set)
    BATCH_NAME = "batch_50"
    DATA_DIR = f"./data/processed_enriched/{BATCH_NAME}/agents"
    MODEL_PATH = "mtp_v5_local.pth"
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
        
    # Load Model (3 Modes) - Scaled to 512
    model = MotionMTP(input_size=6, hidden_size=512, num_modes=3).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    model.eval()
    
    files = glob.glob(f"{DATA_DIR}/*.parquet")
    if not files:
        print(f"Error: No files in {DATA_DIR}")
        return

    print(f"Reading {len(files)} parquet files from {BATCH_NAME}...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    xs = np.stack(df["x"].values)
    ys = np.stack(df["y"].values)
    vxs = np.stack(df["vx"].values)
    vys = np.stack(df["vy"].values)
    hs = np.stack(df["heading"].values)
    map_dists = np.stack(df["map_dist"].values)
    valids = np.stack(df["valid"].values)
    
    mask = valids[:, 10] == 1
    xs, ys, vxs, vys, hs, map_dists, valids = xs[mask], ys[mask], vxs[mask], vys[mask], hs[mask], map_dists[mask], valids[mask]
    
    hist_x = (xs[:, :11] - xs[:, 10:11]) * valids[:, :11]
    hist_y = (ys[:, :11] - ys[:, 10:11]) * valids[:, :11]
    hist_vx = vxs[:, :11] * valids[:, :11]
    hist_vy = vys[:, :11] * valids[:, :11]
    hist_h = hs[:, :11] * valids[:, :11]
    hist_map = np.repeat(map_dists[:, np.newaxis], 11, axis=1) * valids[:, :11]

    inputs = np.stack([hist_x, hist_y, hist_vx, hist_vy, hist_h, hist_map], axis=2)
    
    targets = np.stack([
        (xs[:, 11:] - xs[:, 10:11]) * valids[:, 11:],
        (ys[:, 11:] - ys[:, 10:11]) * valids[:, 11:]
    ], axis=2)
    
    inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
    targets_t = torch.tensor(targets, dtype=torch.float32).to(device)
    valid_t = torch.tensor(valids[:, 11:], dtype=torch.float32).to(device)
    
    print(f"Evaluating on {len(inputs_t)} agents...")
    
    total_ade = 0
    total_fde = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(inputs_t), BATCH_SIZE):
            batch_in = inputs_t[i:i+BATCH_SIZE]
            batch_target = targets_t[i:i+BATCH_SIZE]
            batch_mask = valid_t[i:i+BATCH_SIZE]
            
            pred_trajs, pred_logits = model(batch_in)
            
            ade, fde = calculate_mtp_metrics(pred_trajs, pred_logits, batch_target, batch_mask)
            
            total_ade += ade
            total_fde += fde
            num_batches += 1
            
    print(f"\nRESULTS (Local MTP - 3 Modes):")
    print(f"minADE: {total_ade/num_batches:.4f} meters")
    print(f"minFDE: {total_fde/num_batches:.4f} meters")

if __name__ == "__main__":
    eval_mtp()
