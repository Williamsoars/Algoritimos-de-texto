import torch
import torch.nn as nn

class Biaffine(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.start_mlp = nn.Linear(hidden_size, hidden_size)
        self.end_mlp = nn.Linear(hidden_size, hidden_size)
        self.biaffine = nn.Bilinear(hidden_size, hidden_size, num_labels)

    def forward(self, hidden_states):
        start = self.start_mlp(hidden_states)
        end = self.end_mlp(hidden_states)
        # Atenção: aqui simplificado, falta expandir para todos spans
        scores = self.biaffine(start, end)
        return scores
