import torch
import torch.nn as nn

class MotionLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=2, future_steps=80):
        super(MotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.future_steps = future_steps
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, future_steps * 2) 
        )
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        
        last_hidden = out[:, -1, :]
        
        prediction = self.fc(last_hidden)
        
        prediction = prediction.view(-1, self.future_steps, 2)
        
        return prediction
