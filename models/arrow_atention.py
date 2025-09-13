import torch
import torch.nn as nn
import torch.nn.functional as F

class ArrowAttention(nn.Module):
    def __init__(self, hidden_size: int, window_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size

    def forward(self, hidden_states, attention_mask=None):
        # TODO: implementar arrow attention + LogN-scaling no [CLS]
        # Por enquanto, devolve direto
        return hidden_states
