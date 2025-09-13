# my_sener_lib/models/sener.py
"""
SeNER model assembled from modules:
- Encoder (Transformers AutoModel)
- ArrowAttention (arrow + LogN-scaling on [CLS])
- BiaffineSpan -> token-pair span tensor
- BiSPAModule -> BiSPA blocks + conv + recover
- Final MLP per token-pair to produce scores per entity type

This implementation follows the equations and architecture described
in "Small Language Model Makes an Effective Long Text Extractor".
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from .arrow_attention import ArrowAttention
from .biaffine import BiaffineSpan
from .bispa import BiSPAModule

class SeNER(nn.Module):
    def __init__(self,
                 plm_name: str = "microsoft/deberta-v3-large",
                 encoder_out_dim: int = 1024,
                 span_c: int = 256,
                 num_entity_types: int = 12,
                 arrow_heads: int = 8,
                 arrow_window: int = 128,
                 bispa_wprime: int = 128,
                 bispa_heads: int = 8):
        super().__init__()
        # Encoder
        self.encoder = AutoModel.from_pretrained(plm_name, return_dict=True)
        # If encoder hidden size differs from expected, adapt
        enc_hidden = self.encoder.config.hidden_size

        # Project encoder outputs if necessary to a target D (we keep enc_hidden)
        self.proj = nn.Identity() if enc_hidden == encoder_out_dim else nn.Linear(enc_hidden, encoder_out_dim)

        # Modules
        self.arrow = ArrowAttention(d_model=encoder_out_dim, n_heads=arrow_heads, window_size=arrow_window)
        self.biaffine = BiaffineSpan(hidden_size=encoder_out_dim, c=span_c)
        self.bispa = BiSPAModule(c=span_c, w_prime=bispa_wprime, n_heads=bispa_heads)

        # Final MLP classifier: maps c -> num_entity_types (binary per (i,j, type) in article they use BCE)
        self.final_mlp = nn.Sequential(
            nn.Linear(span_c, span_c),
            nn.ReLU(),
            nn.Linear(span_c, num_entity_types)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        returns: logits tensor [B, L, L, R] where R = num_entity_types
        """
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        H = enc.last_hidden_state  # [B, L, enc_hidden]
        H = self.proj(H)           # [B, L, D]

        # Arrow attention (applies LogN-scaling on CLS inside)
        H2 = self.arrow(H, attention_mask)  # [B, L, D]

        # Biaffine token-pair span tensor
        S = self.biaffine(H2)  # [B, L, L, c]

        # BiSPA transformer block(s)
        Sprime = self.bispa(S)  # [B, L, L, c]

        # Final predicted scores per entity type
        logits = self.final_mlp(Sprime)  # [B, L, L, R]

        # As article: final score uses average of upper & lower triangular; we return full logits
        return logits

