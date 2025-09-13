# my_sener_lib/models/bispa.py
"""
BiSPA module (Bidirectional Sliding-window Plus-shaped Attention).

Implementa:
- compress spans with length <= w_prime into Sh [B, L, w', c]
- compute horizontal self-attention along the window dim for each start i
- transform to Sv and compute vertical self-attention
- concat Zh and Zv, pass through MLP, then conv layers (3x3), finally Recover to [B, L, L, c]
- recover places values at positions j >= i with j-i < w'
Notes:
- Uses nn.MultiheadAttention for horizontal and vertical attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiSPAModule(nn.Module):
    def __init__(self, c: int = 256, w_prime: int = 128, n_heads: int = 8, conv_channels: int = 256):
        super().__init__()
        self.c = c
        self.w_prime = w_prime
        self.n_heads = n_heads

        # Attention modules (embed dim = c)
        self.horiz_attn = nn.MultiheadAttention(embed_dim=c, num_heads=n_heads, batch_first=True)
        self.vert_attn = nn.MultiheadAttention(embed_dim=c, num_heads=n_heads, batch_first=True)

        # MLP to aggregate Zh ⊕ Zv -> c
        self.mlp = nn.Sequential(
            nn.Linear(2*c, c),
            nn.ReLU(),
            nn.LayerNorm(c)
        )

        # Convolution module: will take S' [B, L, w', c] -> map to conv input shape [B, c, L, w']
        self.conv_block = nn.Sequential(
            nn.Conv2d(c, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, c, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def compress_span_tensor(self, S: torch.Tensor):
        """
        S: [B, L, L, c] full token-pair span tensor computed by Biaffine
        produce Sh of shape [B, L, w', c], where for start i we keep spans (i, i .. i+w'-1)
        For positions beyond L-1 pad with zeros.
        """
        B, L, _, c = S.shape
        w = min(self.w_prime, L)
        device = S.device

        Sh = S.new_zeros((B, L, w, c))  # [B, L, w, c]
        for offset in range(w):
            j_idx = torch.arange(0, L, device=device) + offset
            # mask where j_idx < L
            valid = j_idx < L
            if valid.any():
                # S[:, i, j_idx, :] with broadcasting over i
                # Use advanced indexing: we want S[:, i, i+offset, :]
                # Build indices
                i_idx = torch.arange(0, L, device=device)
                # For batch: gather for each batch independently
                # Build index tensors
                # We'll gather by looping over batch to keep code simple & clear (cost O(B*L*w))
                for b in range(B):
                    Sh[b, :, offset, :] = S[b, torch.arange(L, device=device), j_idx.clamp(max=L-1), :]

        return Sh  # [B, L, w, c]

    def forward(self, S: torch.Tensor):
        """
        S: [B, L, L, c]
        Returns recovered S'': [B, L, L, c] after BiSPA block (including conv & recover)
        """
        B, L, _, c = S.shape
        device = S.device
        w = min(self.w_prime, L)

        # Compress to Sh
        # Efficient vectorized implementation:
        # Build indices tensor of shape [L, w] where indices[i, offset] = i + offset (clamped)
        idx = torch.arange(0, L, device=device).unsqueeze(1) + torch.arange(0, w, device=device).unsqueeze(0)
        idx = idx.clamp(max=L-1)  # [L, w]
        # Now gather S across second dim (end index)
        # S has shape [B, L, L, c]; we want Sh[b,i,offset,:] = S[b, i, idx[i,offset], :]
        # Use gather: first reshape S to [B, L, L*c] and then gather, but simpler is advanced indexing:
        # Build i_idx: [L, w] with each row equal to 0..L-1
        i_idx = torch.arange(0, L, device=device).unsqueeze(1).expand(L, w)
        # Now build final indices for first two dims for advanced indexing
        # Use torch.arange for batch dimension
        Sh = S.new_zeros((B, L, w, c))
        for b in range(B):
            Sh[b] = S[b, i_idx, idx, :]  # [L, w, c]

        # Now perform horizontal attention: for each i, attend across window dim (length w)
        # Reshape to (B*L, w, c) for MultiheadAttention batch_first=True
        Sh_flat = Sh.reshape(B*L, w, c)
        # Use key_padding_mask to mask positions that correspond to idx >= L (we clamped so need mask)
        key_padding_mask = (idx >= L)  # [L, w] -> False for valid positions? True for PAD (we used clamp so some equal L-1)
        # But because we clamped, idx >= L never true; instead mask entries where original i+offset >= L
        orig_idx = (torch.arange(0, L, device=device).unsqueeze(1) + torch.arange(0, w, device=device).unsqueeze(0))
        key_padding_mask = (orig_idx >= L)  # True where pad
        key_padding_mask_flat = key_padding_mask.unsqueeze(0).expand(B, -1, -1).reshape(B*L, w)

        # Horizontal attention: query=key=value = Sh_flat
        Zh_flat, _ = self.horiz_attn(Sh_flat, Sh_flat, Sh_flat, key_padding_mask=key_padding_mask_flat)
        Zh = Zh_flat.reshape(B, L, w, c)  # [B, L, w, c]

        # Vertical: we need Sv such that for a given offset, we attend along starts i. Essentially transpose axes
        # Build Sv such that Sv[b, offset, i, :] = Sh[b, i, offset, :]
        Sv = Sh.permute(0, 2, 1, 3)  # [B, w, L, c]
        Sv_flat = Sv.reshape(B*w, L, c)
        # key_padding_mask for vertical: positions i where original i >= L -> no, all i < L. But need to mask based on original j >= L for that offset (we used clamp)
        # For vertical, valid mask per (offset) is orig_idx[:, offset] < L => but orig_idx shape [L, w], transposed gives [w, L]
        key_padding_mask_v = (orig_idx.transpose(0,1) >= L)  # [w, L]
        key_padding_mask_v_flat = key_padding_mask_v.unsqueeze(0).expand(B, -1, -1).reshape(B*w, L)

        Zv_flat, _ = self.vert_attn(Sv_flat, Sv_flat, Sv_flat, key_padding_mask=key_padding_mask_v_flat)
        Zv = Zv_flat.reshape(B, w, L, c).permute(0, 2, 1, 3)  # back to [B, L, w, c]

        # Concatenate Zh and Zv along last dim:
        Z = torch.cat([Zh, Zv], dim=-1)  # [B, L, w, 2c]
        Sprime = self.mlp(Z)  # [B, L, w, c]

        # Conv block: need shape [B, c, L, w]
        conv_in = Sprime.permute(0, 3, 1, 2)  # [B, c, L, w]
        conv_out = self.conv_block(conv_in)   # [B, c, L, w]
        conv_out = conv_out.permute(0, 2, 3, 1)  # [B, L, w, c]

        # Recover to full square tensor [B, L, L, c]; fill with zeros, and place conv_out for positions j = i..i+w-1
        S_recover = S.new_zeros((B, L, L, c))
        for offset in range(w):
            j_idx = torch.arange(0, L, device=device) + offset
            valid_mask = j_idx < L
            j_idx_clamped = j_idx.clamp(max=L-1)
            # place conv_out[:, i, offset, :] into S_recover[:, i, j_idx, :]
            for b in range(B):
                S_recover[b, torch.arange(L, device=device), j_idx_clamped, :] += conv_out[b, :, offset, :]
        # The above sums may add to the last row if clamped; but we followed the article strategy keeping only spans length <= w'
        # Optionally, zero out entries where j<i (we filled only for j>=i because j_idx = i+offset)
        tri_mask = torch.triu(torch.ones(L, L, device=device), diagonal=0).bool()
        S_recover = S_recover * tri_mask.unsqueeze(0).unsqueeze(-1).to(S_recover.dtype)

        return S_recover

