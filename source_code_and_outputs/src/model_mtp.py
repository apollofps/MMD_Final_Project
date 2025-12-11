import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionMTP(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, num_modes=3, future_steps=80):
        super(MotionMTP, self).__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes * future_steps * 2)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        
        raw_trajs = self.reg_head(last_hidden) 
        pred_trajs = raw_trajs.view(-1, self.num_modes, self.future_steps, 2)
        
        logits = self.cls_head(last_hidden)
        
        return pred_trajs, logits

def mtp_loss(pred_trajs, pred_logits, gt_traj, valid_mask):
    batch_size, num_modes, steps, _ = pred_trajs.shape
    
    gt_expand = gt_traj.unsqueeze(1) 
    
    diff = pred_trajs - gt_expand
    dist_sq = torch.sum(diff ** 2, dim=3) 
    
    mask_expand = valid_mask.unsqueeze(1)
    dist_sq = dist_sq * mask_expand
    
    mode_losses = torch.sum(dist_sq, dim=2) 
    
    min_loss, best_mode_idx = torch.min(mode_losses, dim=1)
    
    cls_loss = F.cross_entropy(pred_logits, best_mode_idx)
    
    reg_loss = torch.mean(min_loss)
    
    total_loss = reg_loss + 1.0 * cls_loss
    
    return total_loss, reg_loss.item(), cls_loss.item()
