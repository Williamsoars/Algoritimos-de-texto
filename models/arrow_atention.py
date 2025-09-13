# my_sener_lib/models/arrow_attention.py
"""
ArrowAttention module.

Implementa a ideia do artigo:
- [CLS] usa attention global.
- Demais tokens usam sliding-window attention (unilateral window w each side)
  + também podem consultar [CLS] (como sink).
- LogN-Scaling aplicado ao cálculo de atenção do [CLS] (interpretação: multiplicador = log_base512(L)).
Notes/assumptions:
- CLS token index assumed = 0
- Input: hidden_states [B, L, D], attention_mask [B, L] (1 for valid tokens)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArrowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        # projectors
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        # x: [B, L, D] -> [B, n_heads, L, head_dim]
        B, L, D = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1,2)

    def _combine_heads(self, x):
        # x: [B, n_heads, L, head_dim] -> [B, L, D]
        B, H, L, hd = x.shape
        return x.transpose(1,2).contiguous().view(B, L, H*hd)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        hidden_states: [B, L, D]
        attention_mask: [B, L] with 1 for valid tokens, 0 for padding (optional)
        returns: same shape [B, L, D]
        """
        B, L, D = hidden_states.size()
        q = self.q_proj(hidden_states)  # [B, L, D]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        qh = self._split_heads(q)  # [B, H, L, hd]
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        device = hidden_states.device
        out = torch.zeros_like(qh)  # [B, H, L, hd]

        # Precompute attention_mask for broadcasting
        if attention_mask is None:
            attn_mask = torch.ones(B, L, device=device, dtype=torch.bool)
        else:
            attn_mask = attention_mask.bool().to(device)

        # Precompute scale
        scale = 1.0 / math.sqrt(self.head_dim)

        # LogN-Scaling factor for [CLS] attention: factor = log_base512(L)
        # if L <= 1, factor = 1.0
        logn_factor = math.log(max(L, 1), 512) if L > 1 else 1.0
        # guard small values (if L < 512, log_base512(L) < 1) -> ok per paper
        logn_factor = float(logn_factor)

        # We'll compute attention per head for efficiency across batch
        # Strategy:
        # - For query idx = 0 (CLS), allow keys=0..L-1 (global).
        # - For queries i>0: allow keys j where |i-j| <= window_size and also key 0 (CLS).
        # Implement via masked matmuls.
        # Convert to [B*H, L, hd] for batched matmul
        qh_b = qh.reshape(B*self.n_heads, L, self.head_dim)
        kh_b = kh.reshape(B*self.n_heads, L, self.head_dim)
        vh_b = vh.reshape(B*self.n_heads, L, self.head_dim)
        attn_out = torch.zeros_like(qh_b)

        # Precompute key padding mask for each example (B, L)
        key_padding = attn_mask  # True for valid tokens

        # compute full QK^T once (batched), then mask where needed
        # Full scores: [B*H, L, L]
        scores_full = torch.matmul(qh_b, kh_b.transpose(-1, -2)) * scale  # [B*H, L, L]

        # apply logn scaling to CLS query rows (row index 0)
        # rows correspond to queries; for CLS query row we multiply by logn_factor
        # Indexing: rows for each batch*head: row 0 corresponds to query index 0
        if logn_factor != 1.0:
            scores_full[:, 0, :] = scores_full[:, 0, :] * logn_factor

        # Build per-query attention masks
        # mask_matrix: True where allowed to attend
        # Start with False and enable allowed positions
        mask_matrix = torch.zeros(B, L, L, device=device, dtype=torch.bool)
        for i in range(L):
            if i == 0:
                # CLS can attend to all positions that are valid
                # We'll set allowed positions according to key_padding per batch
                mask_matrix[:, i, :] = key_padding
            else:
                left = max(0, i - self.window_size)
                right = min(L - 1, i + self.window_size)
                # allow local window
                mask_matrix[:, i, left:right+1] = True
                # always allow CLS key (index 0) if valid
                mask_matrix[:, i, 0] = key_padding[:, 0]

        # Expand mask to heads dimension when flattening
        mask_flat = mask_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [B, H, L, L]
        mask_flat = mask_flat.reshape(B*self.n_heads, L, L)  # [B*H, L, L]

        # Apply key padding: ensure positions that are padding are masked out for all queries
        key_pad_exp = (~key_padding).unsqueeze(1).expand(-1, L, -1)  # [B, L, L] (True where pad)
        key_pad_exp = key_pad_exp.unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(B*self.n_heads, L, L)

        # final_allow = mask_flat & (~key_pad_exp)
        final_allow = mask_flat & (~key_pad_exp)

        # set disallowed positions to -inf
        scores_full = scores_full.masked_fill(~final_allow, -1e9)

        # Softmax and matmul
        attn_probs = torch.softmax(scores_full, dim=-1)  # [B*H, L, L]
        attn_out = torch.matmul(attn_probs, vh_b)  # [B*H, L, hd]

        # reshape back
        attn_out = attn_out.view(B, self.n_heads, L, self.head_dim)
        out = self._combine_heads(attn_out)  # [B, L, D]
        out = self.out_proj(out)
        return out

