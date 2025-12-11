import torch
import torch.nn as nn

class MotionLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=2, future_steps=80):
        super(MotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.future_steps = future_steps
        
        # Encoder: Takes history (x, y, vx, vy, heading)
        # Input shape: (batch_size, seq_len=11, input_size=5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Decoder: Predicts future (dx, dy)
        # Simple MLP decoder from the last hidden state
        # In a real SOTA models, we would use a Decoder LSTM or Transformer.
        # Here, we project the final hidden state to the entire future trajectory (flat)
        # Output: future_steps * 2 (flatted array of dx, dy)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, future_steps * 2) # Predict all 80 steps at once (Simplification)
        )
        
    def forward(self, x):
        # x: (batch, history_len, input_size)
        
        # LSTM Encoder
        # out: (batch, seq_len, hidden)
        # h_n: (num_layers, batch, hidden)
        out, (h_n, c_n) = self.lstm(x)
        
        # Take the last time-step output from the encoder
        last_hidden = out[:, -1, :] # (batch, hidden)
        
        # Decode
        prediction = self.fc(last_hidden) # (batch, 160)
        
        # Reshape to (batch, future_steps, 2)
        prediction = prediction.view(-1, self.future_steps, 2)
        
        return prediction
