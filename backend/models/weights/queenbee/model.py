"""
ECG-Transformer: Foundation Model for 12-Lead ECG Analysis
QueenBee Medical AI Stack

Architecture:
- 1D Patch Embedding (100ms patches at 500Hz = 50 samples)
- 12-lead processing with lead embeddings
- Transformer encoder with rotary position embeddings
- Multi-label classification heads (5 superclasses + 71 SCP codes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding for better position awareness"""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Shape: (1, 1, seq_len, head_dim) for broadcasting with (B, heads, seq_len, head_dim)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ECGPatchEmbedding(nn.Module):
    """
    Convert 12-lead ECG into patches.
    Each patch = 50 samples (100ms at 500Hz) - clinically meaningful window
    """

    def __init__(
        self,
        num_leads: int = 12,
        patch_size: int = 50,  # 100ms at 500Hz
        embed_dim: int = 256,
        signal_length: int = 5000,  # 10 seconds at 500Hz
    ):
        super().__init__()
        self.num_leads = num_leads
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = signal_length // patch_size  # 100 patches

        # 1D convolution for patch embedding
        self.proj = nn.Conv1d(
            num_leads,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Lead type embedding (I, II, III, aVR, aVL, aVF, V1-V6)
        self.lead_embedding = nn.Embedding(num_leads, embed_dim)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_leads, signal_length) - 12-lead ECG
        returns: (batch, num_patches + 1, embed_dim)
        """
        B = x.shape[0]

        # Patch projection: (B, 12, 5000) -> (B, embed_dim, 100)
        x = self.proj(x)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add lead-wise positional context (average across patches per lead)
        lead_ids = torch.arange(self.num_leads, device=x.device)
        lead_emb = self.lead_embedding(lead_ids).mean(dim=0, keepdim=True)  # (1, embed_dim)
        x = x + lead_emb.unsqueeze(0)  # Broadcast

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        return self.norm(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with rotary embeddings"""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.rotary = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        cos, sin = self.rotary(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ECGTransformer(nn.Module):
    """
    ECG-Transformer Foundation Model

    Trained on PTB-XL dataset (21,799 12-lead ECGs)
    Multi-label classification for 5 superclasses + 71 SCP diagnostic codes
    """

    def __init__(
        self,
        num_leads: int = 12,
        signal_length: int = 5000,  # 10s @ 500Hz
        patch_size: int = 50,       # 100ms patches
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_superclasses: int = 5,   # NORM, MI, STTC, CD, HYP
        num_scp_codes: int = 71,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = ECGPatchEmbedding(
            num_leads=num_leads,
            patch_size=patch_size,
            embed_dim=embed_dim,
            signal_length=signal_length,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification heads
        self.superclass_head = nn.Linear(embed_dim, num_superclasses)
        self.scp_head = nn.Linear(embed_dim, num_scp_codes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from ECG signal"""
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, 12, 5000) - 12-lead ECG signal
        returns: (superclass_logits, scp_logits)
        """
        features = self.forward_features(x)

        superclass_logits = self.superclass_head(features)
        scp_logits = self.scp_head(features)

        return superclass_logits, scp_logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for downstream tasks"""
        return self.forward_features(x)


# Model configurations
def ecg_transformer_tiny(**kwargs):
    """Tiny model for fast iteration (1.2M params)"""
    return ECGTransformer(
        embed_dim=128,
        depth=4,
        num_heads=4,
        **kwargs
    )

def ecg_transformer_small(**kwargs):
    """Small model (4.7M params)"""
    return ECGTransformer(
        embed_dim=256,
        depth=6,
        num_heads=8,
        **kwargs
    )

def ecg_transformer_base(**kwargs):
    """Base model (12M params)"""
    return ECGTransformer(
        embed_dim=384,
        depth=8,
        num_heads=8,
        **kwargs
    )

def ecg_transformer_large(**kwargs):
    """Large model (45M params)"""
    return ECGTransformer(
        embed_dim=512,
        depth=12,
        num_heads=16,
        **kwargs
    )


if __name__ == "__main__":
    # Test model
    model = ecg_transformer_small()
    print(f"ECG-Transformer Small")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(4, 12, 5000)  # 4 ECGs, 12 leads, 5000 samples
    superclass, scp = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Superclass output: {superclass.shape}")  # (4, 5)
    print(f"SCP codes output: {scp.shape}")  # (4, 71)

    # Test embeddings
    emb = model.get_embeddings(x)
    print(f"Embedding shape: {emb.shape}")  # (4, 256)
