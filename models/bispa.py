import torch
import torch.nn as nn

class BiSPAModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, span_repr):
        # Espera um tensor [batch, seq, seq, hidden]
        # Aplicação simplificada de conv2D
        x = span_repr.permute(0, 3, 1, 2)  # [B, H, seq, seq]
        x = self.conv(x)
        return x.permute(0, 2, 3, 1)       # [B, seq, seq, H]
