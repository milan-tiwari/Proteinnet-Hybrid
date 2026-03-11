from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal position encoding (no learned parameters)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x.

        Args:
            x: (B, L, d_model)
        """
        L = x.size(1)
        if L > self.max_len:
            # Fallback: extend on the fly (rare). This keeps the module safe for long proteins.
            self._extend(L)
        return x + self.pe[:, :L]

    def _extend(self, new_max_len: int) -> None:
        max_len = int(new_max_len * 1.1)  # small buffer
        d_model = self.d_model

        pe = torch.zeros(max_len, d_model, device=self.pe.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.pe.device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.max_len = max_len
        self.register_buffer("pe", pe, persistent=False)


@dataclass
class ProteinTransformerConfig:
    vocab_size: int = 22
    evo_dim: int = 21
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 4096


class ProteinTransformer(nn.Module):
    """Transformer encoder that predicts per-residue Cα coordinates."""

    def __init__(self, cfg: ProteinTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.evo_proj = nn.Linear(cfg.evo_dim, cfg.d_model)

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 3),
        )

    def forward(self, tokens: torch.Tensor, evo: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: (B, L) long
            evo: (B, L, 21) float
            pad_mask: (B, L) bool, True where padding (for Transformer attention masking)

        Returns:
            pred_ca: (B, L, 3)
        """
        x = self.token_emb(tokens) + self.evo_proj(evo)
        x = self.pos_enc(x)
        x = self.dropout(x)

        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(h)
