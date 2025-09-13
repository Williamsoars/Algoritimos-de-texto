# my_sener_lib/models/biaffine.py
"""
Biaffine span representation.

Implementa a Eqns (4)-(5) do artigo:
Hs = MLP_start(H)
He = MLP_end(H)
Si,j = Hs_i^T W1 He_j + W2 [Hs_i || He_j] + b

Outputs: token-pair span tensor S of shape [B, L, L, c]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiaffineSpan(nn.Module):
    def __init__(self, hidden_size: int, c: int = 256, dropout: float = 0.1):
        """
        hidden_size: D from encoder
        c: output channel dimension for token-pair span tensor
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.c = c
        self.start_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # W1: (hidden, c, hidden)
        # We'll store as parameter of shape (c, hidden, hidden) for einsum convenience
        self.W1 = nn.Parameter(torch.randn(c, hidden_size, hidden_size) * (hidden_size ** -0.5))
        # W2: linear combining concatenation -> c dims
        self.W2 = nn.Linear(2*hidden_size, c)
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [B, L, D]
        returns: S [B, L, L, c] (upper triangle and lower triangle both computed)
        """
        B, L, D = hidden_states.shape
        Hs = self.start_mlp(hidden_states)  # [B, L, D]
        He = self.end_mlp(hidden_states)    # [B, L, D]

        # Compute bilinear term efficiently:
        # For each c: S_c = Hs @ W1_c @ He^T
        # Using einsum: 'b i d, c d e, b j e -> b i j c'
        S_bilinear = torch.einsum('bid,cde,bje->bijc', Hs, self.W1, He)  # [B, L, L, c]

        # Compute linear concat term:
        # Expand Hs and He to pairwise grid
        Hs_exp = Hs.unsqueeze(2).expand(B, L, L, D)  # [B, L, L, D]
        He_exp = He.unsqueeze(1).expand(B, L, L, D)  # [B, L, L, D]
        concat = torch.cat([Hs_exp, He_exp], dim=-1)  # [B, L, L, 2D]
        S_linear = self.W2(concat)  # [B, L, L, c]

        S = S_bilinear + S_linear + self.bias  # [B, L, L, c]
        return S

