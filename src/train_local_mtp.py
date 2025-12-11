import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import os
import gc
import time
from src.model_mtp import MotionMTP, mtp_loss

def train_mtp():
    print("========================================")
    print("   Local MTP Training (MPS) 🔮          ")
    print("========================================")
    print("Predicting 3 Modes (Left/Straight/Right)")
    
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    EPOCHS = 10 # More epochs to learn multiple modes
    DATA_DIR = "./data/processed_enriched"
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not found")
        
    # Scaling Up: Hidden Size 128 -> 512
    model = MotionMTP(input_size=6, hidden_size=512, num_modes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Batches 00 to 99 (Full Scale)
    batches = [f"batch_{i:02d}" for i in range(100)]
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_reg_loss = 0
        batch_count = 0
        
        for batch_name in batches:
            path = f"{DATA_DIR}/{batch_name}/agents"
            files = glob.glob(f"{path}/*.parquet")
            if not files: continue
                
            try:
                dfs = [pd.read_parquet(f) for f in files]
                df = pd.concat(dfs, ignore_index=True)
                if "map_dist" not in df.columns: continue

                # Preprocess (Same as before)
                xs = np.stack(df["x"].values)
                ys = np.stack(df["y"].values)
                vxs = np.stack(df["vx"].values)
                vys = np.stack(df["vy"].values)
                hs = np.stack(df["heading"].values)
                map_dists = np.stack(df["map_dist"].values)
                valids = np.stack(df["valid"].values)
                
                mask = valids[:, 10] == 1
                xs, ys, vxs, vys, hs, map_dists, valids = xs[mask], ys[mask], vxs[mask], vys[mask], hs[mask], map_dists[mask], valids[mask]
                
                if len(xs) == 0: continue

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
                
                target_valid = valids[:, 11:]
                
                # To MPS
                inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
                targets_t = torch.tensor(targets, dtype=torch.float32).to(device)
                valid_t = torch.tensor(target_valid, dtype=torch.float32).to(device)
                
                model.train()
                indices = torch.randperm(len(inputs_t))
                
                batch_reg = 0
                steps = 0
                
                for i in range(0, len(inputs_t), BATCH_SIZE):
                    idx = indices[i:i+BATCH_SIZE]
                    optimizer.zero_grad()
                    
                    pred_trajs, pred_logits = model(inputs_t[idx])
                    
                    loss, reg, cls = mtp_loss(pred_trajs, pred_logits, targets_t[idx], valid_t[idx])
                    
                    loss.backward()
                    optimizer.step()
                    
                    batch_reg += reg
                    steps += 1
                
                avg_reg = batch_reg/steps
                epoch_reg_loss += avg_reg
                batch_count += 1
                
                if batch_count % 10 == 0:
                    print(f"  Batch {batch_name}: Reg Loss {avg_reg:.4f}")
                
                del df, inputs_t, targets_t, valid_t
                gc.collect()
            except Exception as e:
                print(f"Error {batch_name}: {e}")

        print(f"Epoch {epoch+1} Avg Reg Loss: {epoch_reg_loss/batch_count:.4f}")

    torch.save(model.state_dict(), "mtp_v5_local.pth")
    print("Model saved to mtp_v5_local.pth")

if __name__ == "__main__":
    train_mtp()
