"""
HeatFormer — PyTorch model definition
======================================
Architecture (matches Notebook 3, extended for spatial risk maps):

  Stream 1 : SpatialEncoder   — ResNet-50 (5-ch input) → 16 spatial tokens
  Stream 2 : TemporalEncoder  — 2-layer LSTM over 8-day met window
  Fusion   : CrossAttentionFusion — sat tokens attend to met context
  Decoder  : SpatialDecoder   — 16 tokens → 64×64 × 4-class risk map

Output risk classes:
  0 = Low   1 = Moderate   2 = High   3 = Extreme
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from einops import rearrange


# ── Stream 1: Spatial Encoder ────────────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """ResNet-50 adapted for 5-channel satellite input → 16 spatial tokens."""

    def __init__(self, in_ch: int = 5, out_dim: int = 256):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)

        # Adapt first conv: keep pretrained-style init for ch 0-2, replicate for 3-4
        old_w = resnet.conv1.weight.data          # (64, 3, 7, 7)
        new_conv = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_w
            extra = old_w[:, : in_ch - 3].mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = extra.expand(-1, in_ch - 3, -1, -1)
        resnet.conv1 = new_conv

        # Drop avgpool + fc; keep feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])   # → (B, 2048, H, W)
        self.pool     = nn.AdaptiveAvgPool2d((4, 4))                    # → (B, 2048, 4, 4)
        self.proj     = nn.Linear(2048, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 5, 64, 64) → (B, 16, out_dim)"""
        feat = self.backbone(x)                              # (B, 2048, H, W)
        feat = self.pool(feat)                               # (B, 2048, 4, 4)
        feat = rearrange(feat, "b c h w -> b (h w) c")      # (B, 16, 2048)
        return self.proj(feat)                               # (B, 16, 256)


# ── Stream 2: Temporal Encoder ───────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """Two-layer LSTM over the 8-day meteorological window."""

    def __init__(self, in_feat: int = 11, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(in_feat, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, 11) → (B, 8, 128)"""
        out, _ = self.lstm(x)
        return self.norm(out)


# ── Cross-Attention Fusion ────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """Satellite spatial tokens attend to meteorological time context."""

    def __init__(self, sat_dim: int = 256, met_dim: int = 128, heads: int = 8):
        super().__init__()
        self.kv_proj = nn.Linear(met_dim, sat_dim)
        self.attn    = nn.MultiheadAttention(sat_dim, heads,
                                              batch_first=True, dropout=0.1)
        self.norm1   = nn.LayerNorm(sat_dim)
        self.norm2   = nn.LayerNorm(sat_dim)
        self.ff      = nn.Sequential(
            nn.Linear(sat_dim, sat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(sat_dim * 2, sat_dim),
        )

    def forward(self, sat_tok: torch.Tensor,
                met_ctx: torch.Tensor) -> torch.Tensor:
        """sat_tok: (B, 16, 256)  met_ctx: (B, 8, 128)  → (B, 16, 256)"""
        kv          = self.kv_proj(met_ctx)          # (B, 8, 256)
        attn_out, _ = self.attn(sat_tok, kv, kv)
        x = self.norm1(sat_tok + attn_out)
        return self.norm2(x + self.ff(x))


# ── Spatial Decoder (token → 64×64 risk map) ─────────────────────────────────

class SpatialDecoder(nn.Module):
    """Decode 4×4 token grid → 64×64 risk map via transposed convolutions."""

    def __init__(self, in_dim: int = 256, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            # 4×4 → 8×8
            nn.ConvTranspose2d(in_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(32, n_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 16, 256) → logits: (B, n_classes, 64, 64)"""
        x = rearrange(tokens, "b (h w) c -> b c h w", h=4, w=4)
        return self.net(x)


# ── Full HeatFormer ───────────────────────────────────────────────────────────

class HeatFormer(nn.Module):
    """
    HeatFormer: dual-stream transformer for urban heat risk mapping.

    Inputs
    ------
    sat : (B, 5, 64, 64)  — satellite channels [LST_C, NDVI, NDBI, NDWI, SAVI]
    met : (B, 8, 11)      — 8-day meteorological window (11 features)

    Output
    ------
    logits : (B, 4, 64, 64) — per-pixel class logits
    """

    def __init__(self, in_ch: int = 5, met_feat: int = 11, n_classes: int = 4):
        super().__init__()
        self.spatial  = SpatialEncoder(in_ch=in_ch, out_dim=256)
        self.temporal = TemporalEncoder(in_feat=met_feat, hidden=128)
        self.fusion   = CrossAttentionFusion(sat_dim=256, met_dim=128, heads=8)
        self.decoder  = SpatialDecoder(in_dim=256, n_classes=n_classes)

    def forward(self, sat: torch.Tensor, met: torch.Tensor) -> torch.Tensor:
        sat_tok = self.spatial(sat)              # (B, 16, 256)
        met_ctx = self.temporal(met)             # (B, 8, 128)
        fused   = self.fusion(sat_tok, met_ctx)  # (B, 16, 256)
        return self.decoder(fused)               # (B, 4, 64, 64)

    @torch.no_grad()
    def predict(self, sat: torch.Tensor,
                met: torch.Tensor) -> tuple[list[list[int]], list[list[float]]]:
        """
        Returns
        -------
        risk_map : List[List[int]]   shape (64, 64)  argmax classes 0-3
        prob_map : List[List[float]] shape (64, 64)  max class probability
        """
        self.eval()
        logits    = self.forward(sat, met)              # (B, 4, 64, 64)
        probs     = logits.softmax(dim=1)               # (B, 4, 64, 64)
        risk_map  = logits.argmax(dim=1)                # (B, 64, 64)
        prob_map  = probs.max(dim=1).values             # (B, 64, 64)

        # Return first sample as nested Python lists
        return (
            risk_map[0].cpu().tolist(),
            [[round(v, 3) for v in row]
             for row in prob_map[0].cpu().tolist()],
        )


# ── Loader helper ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cpu") -> HeatFormer:
    """Load HeatFormer from a checkpoint (.pth/.pt)."""
    model = HeatFormer()
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Accept common checkpoint layouts.
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break

    if not isinstance(state, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format at {checkpoint_path}: "
            f"{type(state).__name__}"
        )

    # Strip DataParallel prefixes when present.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
