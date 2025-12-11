import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionMTP(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, num_modes=3, future_steps=80):
        super(MotionMTP, self).__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        
        # Encoder (Same as v3)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Decoder Head
        # Instead of 1 path, we predict 'num_modes' paths
        # Output dim = num_modes * (future_steps * 2)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes * future_steps * 2)
        )
        
        # Probability Head (Classifies which mode is likely)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes) # Logits for each mode
        )
        
    def forward(self, x):
        # x: (batch, seq, input)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :] # (batch, hidden)
        
        # Regress 3 Trajectories
        # (batch, num_modes * steps * 2)
        raw_trajs = self.reg_head(last_hidden) 
        # Reshape to (batch, num_modes, steps, 2)
        pred_trajs = raw_trajs.view(-1, self.num_modes, self.future_steps, 2)
        
        # Predict Probabilities
        logits = self.cls_head(last_hidden) # (batch, num_modes)
        
        return pred_trajs, logits

def mtp_loss(pred_trajs, pred_logits, gt_traj, valid_mask):
    """
    Winner-Takes-All Loss.
    pred_trajs: (batch, 3, 80, 2)
    pred_logits: (batch, 3)
    gt_traj: (batch, 80, 2)
    valid_mask: (batch, 80)
    """
    batch_size, num_modes, steps, _ = pred_trajs.shape
    
    # 1. Calculate L2 distance for ALL modes
    # gt_traj expanded: (batch, 1, 80, 2)
    gt_expand = gt_traj.unsqueeze(1) 
    
    # Diff: (batch, 3, 80, 2)
    diff = pred_trajs - gt_expand
    # Squared Error: (batch, 3, 80)
    dist_sq = torch.sum(diff ** 2, dim=3) 
    
    # Mask invalid steps: (batch, 1, 80)
    mask_expand = valid_mask.unsqueeze(1)
    dist_sq = dist_sq * mask_expand
    
    # Sum over time -> Average Displacement Error for each mode
    # (batch, 3)
    mode_losses = torch.sum(dist_sq, dim=2) 
    
    # 2. Select Winner (Minimum Loss)
    # best_mode_idx: (batch,)
    min_loss, best_mode_idx = torch.min(mode_losses, dim=1)
    
    # 3. Classification Loss (Teach logits to predict the best mode)
    cls_loss = F.cross_entropy(pred_logits, best_mode_idx)
    
    # 4. Total Loss = Regression Loss of Winner + Classification Loss
    # We take average over batch
    reg_loss = torch.mean(min_loss)
    
    # Alpha balances the two. 
    total_loss = reg_loss + 1.0 * cls_loss
    
    return total_loss, reg_loss.item(), cls_loss.item()
