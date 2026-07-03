"""
================================================================================
model.py -- Multi-Stream Cross-Attention (MS-CA) Architecture
================================================================================

Purpose
-------
This module contains ONLY the model architecture for the MS-CA satellite
data fusion network:

    Stream A (Environment Proxy)          : [Batch, 3, H, W]
        Channels: NO2, CO, SO2
    Stream B (Socio-Infrastructure Proxy) : [Batch, 5, H, W]
        Channels: Nightlight, Population, Power Plants, Road Network,
                  Land Cover

Design goals
------------
1. STRICT MODULARITY: Encoders are built behind an abstract `BaseEncoder`
   interface. The current backbone is a lightweight Vision Transformer (ViT)
   patch-embedding encoder. To later swap in a Vision Mamba backbone, you
   only need to write a new class that subclasses `BaseEncoder` and
   implements `forward()` / `num_patches` / `grid_size`. Nothing in the
   fusion layer, task heads, or any downstream training code needs to
   change, AS LONG AS the new encoder still emits tokens shaped
   [Batch, Sequence_Length, Embed_Dim].

2. EXPLAINABILITY: The cross-attention fusion layer explicitly returns the
   raw attention score matrix (per-head) so it can be reshaped back into a
   2D spatial grid for visual "where is the model looking" maps via the
   `reshape_attention_to_spatial_grid` helper.

3. MULTI-TASK LEARNING: The fused representation is pooled and passed to two
   independent MLP heads -- a 5-class hotspot classifier and a scalar
   greenhouse-gas regression head.

Note
----
This file intentionally contains NO dataset, training loop, or inference
demo code -- it is pure architecture, meant to be imported by separate
training/evaluation scripts (e.g. `from model import MSCANet`).

Requirements
------------
    torch >= 1.12 (for `average_attn_weights` kwarg on nn.MultiheadAttention)
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


# ==============================================================================
# SECTION 1: MODULAR ENCODER INTERFACE
# ==============================================================================
class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class that every stream-encoder backbone must implement.

    Contract
    --------
    forward(x) : x is a raw image tensor [Batch, Channels, H, W] and MUST
                 return a token sequence [Batch, Sequence_Length, Embed_Dim].
    num_patches: the fixed sequence length produced by the encoder.
    grid_size  : (grid_h, grid_w) so the token sequence can be reshaped back
                 into a 2D spatial map -- this is what makes the attention
                 matrices interpretable/visualizable later.

    Why this matters
    -----------------
    The Cross-Attention fusion layer, the pooling step, and the task heads
    ONLY ever see [Batch, Sequence_Length, Embed_Dim] tensors. They never
    know or care whether the tokens came from a ViT, a Vision Mamba scan,
    a CNN + flatten, etc. This is what lets you swap backbones later
    without touching any downstream code -- just implement a new subclass
    of BaseEncoder (e.g. `VisionMambaEncoder`) with the same output
    contract and drop it into `MSCANet.__init__`.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_patches(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def grid_size(self) -> Tuple[int, int]:
        raise NotImplementedError


class TransformerEncoderBlock(nn.Module):
    """
    A single pre-norm Transformer encoder block (self-attention + MLP).
    Used internally by the ViT backbone. Kept as its own class so it can
    also be reused by a future backbone if useful.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Self-attention sub-block (pre-norm + residual) ---
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attn_out

        # --- MLP sub-block (pre-norm + residual) ---
        x = x + self.mlp(self.norm2(x))
        return x


class ViTPatchEncoder(BaseEncoder):
    """
    Lightweight Vision Transformer encoder.

    Pipeline: Conv2d patch-embedding -> add learned positional embeddings
    -> N Transformer encoder blocks -> final LayerNorm.

    Note: no [CLS] token is used on purpose. Keeping the output sequence
    purely spatial (one token per patch, nothing else) means the sequence
    length always equals grid_h * grid_w, which is required to cleanly
    reshape attention maps back into a 2D grid for XAI visualization.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int],
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim)
        img_h, img_w = img_size
        assert img_h % patch_size == 0 and img_w % patch_size == 0, (
            "Image dimensions must be divisible by patch_size."
        )

        self.patch_size = patch_size
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size
        self._num_patches = self.grid_h * self.grid_w

        # Patch embedding: a single strided conv turns each non-overlapping
        # patch_size x patch_size patch into one embed_dim-length vector.
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learned positional embeddings, one per patch/token.
        self.pos_embed = nn.Parameter(torch.zeros(1, self._num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self.grid_h, self.grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.patch_embed(x)                 # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)         # [B, N, embed_dim]
        x = x + self.pos_embed                   # broadcast add positional info
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x                                 # [B, N, embed_dim]


# ==============================================================================
# SECTION 2: CROSS-ATTENTION FUSION LAYER
# ==============================================================================
class CrossAttentionFusion(nn.Module):
    """
    Fuses the two encoded streams via cross-attention.

        Query (Q)      <- Stream B tokens (Socio-Infrastructure)
        Key/Value (K,V) <- Stream A tokens (Environment)

    This design means the model asks: "given each socio-infrastructure
    patch, which environmental (NO2/CO/SO2) patches are most relevant?" --
    which is intuitive for hotspot attribution (e.g. "this power-plant
    patch is driven by emission concentration in these specific patches").

    The raw attention weight matrix is returned unmodified (per attention
    head) for downstream Explainable-AI visualization.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self, tokens_b: torch.Tensor, tokens_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens_b: [Batch, Lq, Embed_Dim]  (Query source: Stream B)
            tokens_a: [Batch, Lk, Embed_Dim]  (Key/Value source: Stream A)

        Returns:
            fused_tokens : [Batch, Lq, Embed_Dim]
            attn_weights : [Batch, Num_Heads, Lq, Lk]  (explicit, per-head,
                            for XAI -- NOT averaged across heads)
        """
        q = self.norm_q(tokens_b)
        kv = self.norm_kv(tokens_a)

        # need_weights=True + average_attn_weights=False -> per-head weight
        # matrix of shape [Batch, Num_Heads, Lq, Lk]. Requires torch>=1.12.
        fused, attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=False,
        )

        # Residual connection back onto the query stream (Stream B),
        # so the fused output still carries the socio-infra identity.
        fused_tokens = self.norm_out(fused + tokens_b)
        return fused_tokens, attn_weights


# ==============================================================================
# SECTION 3: MULTI-TASK OUTPUT HEADS
# ==============================================================================
class ClassificationHead(nn.Module):
    """5-class emission hotspot grading head (Class 0 .. Class 4)."""

    def __init__(self, embed_dim: int, num_classes: int = 5, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # [Batch, num_classes]


class RegressionHead(nn.Module):
    """Scalar absolute greenhouse-gas emission value regression head."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # [Batch, 1]


# ==============================================================================
# SECTION 4: FULL MS-CA MODEL
# ==============================================================================
class MSCANet(nn.Module):
    """
    Full Multi-Stream Cross-Attention network.

    Composition (all injected as sub-modules, nothing hard-wired):
        encoder_a  : BaseEncoder subclass for Stream A (Environment)
        encoder_b  : BaseEncoder subclass for Stream B (Socio-Infra)
        fusion     : CrossAttentionFusion
        cls_head   : ClassificationHead
        reg_head   : RegressionHead

    To swap the backbone later (e.g. to Vision Mamba), only the
    `encoder_a_cls` / `encoder_b_cls` constructor arguments need to change
    -- everything else is untouched.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 64),
        patch_size: int = 16,
        embed_dim: int = 128,
        encoder_depth: int = 2,
        num_heads: int = 4,
        num_classes: int = 5,
        dropout: float = 0.1,
        encoder_a_cls=ViTPatchEncoder,
        encoder_b_cls=ViTPatchEncoder,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Independent encoder instances -> weights are NOT shared between
        # Stream A and Stream B, since the two modalities have different
        # statistics.
        #
        # CRITICAL FIX: Stream A is finalized to carry 3 channels
        # (NO2, CO, SO2), so encoder_a now uses in_channels=3 (previously
        # incorrectly set to 1). Stream B remains at in_channels=5
        # (Nightlight, Population, Power Plants, Road Network, Land Cover).
        self.encoder_a = encoder_a_cls(
            in_channels=3,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.encoder_b = encoder_b_cls(
            in_channels=5,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(embed_dim, num_heads=num_heads, dropout=dropout)

        self.cls_head = ClassificationHead(embed_dim, num_classes=num_classes, dropout=dropout)
        self.reg_head = RegressionHead(embed_dim, dropout=dropout)

    def forward(
        self,
        stream_a: torch.Tensor,
        stream_b: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            stream_a: [Batch, 3, H, W]  Environment proxy (NO2, CO, SO2)
            stream_b: [Batch, 5, H, W]  Socio-infrastructure proxy
            return_attention: if True, also return the raw XAI attention
                               weight matrix from the fusion layer.

        Returns:
            cls_logits : [Batch, num_classes]
            reg_out    : [Batch, 1]
            (optional) attn_weights : [Batch, Num_Heads, Lq, Lk]
        """
        tokens_a = self.encoder_a(stream_a)   # [B, Na, D]
        tokens_b = self.encoder_b(stream_b)   # [B, Nb, D]

        fused_tokens, attn_weights = self.fusion(tokens_b, tokens_a)  # [B, Nb, D]

        # Global average pooling over the sequence dimension to get a
        # single feature vector per sample before the task heads.
        pooled = fused_tokens.mean(dim=1)     # [B, D]

        cls_logits = self.cls_head(pooled)    # [B, num_classes]
        reg_out = self.reg_head(pooled)       # [B, 1]

        if return_attention:
            return cls_logits, reg_out, attn_weights
        return cls_logits, reg_out


# ==============================================================================
# SECTION 5: XAI HELPER -- RESHAPE ATTENTION BACK TO A 2D SPATIAL MAP
# ==============================================================================
def reshape_attention_to_spatial_grid(
    attn_weights: torch.Tensor,
    query_grid: Tuple[int, int],
    key_grid: Tuple[int, int],
    average_heads: bool = True,
) -> torch.Tensor:
    """
    Converts the flat cross-attention matrix into a 4D/5D spatial tensor
    suitable for visual mapping (e.g. overlaying on the input raster).

    Args:
        attn_weights : [Batch, Num_Heads, Lq, Lk] as returned by
                       CrossAttentionFusion.
        query_grid   : (grid_h, grid_w) of the Query source encoder
                       (Stream B / socio-infra), with grid_h*grid_w == Lq.
        key_grid     : (grid_h, grid_w) of the Key/Value source encoder
                       (Stream A / environment), with grid_h*grid_w == Lk.
        average_heads: if True, average across attention heads first.

    Returns:
        If average_heads:
            [Batch, gh_q, gw_q, gh_k, gw_k]
        else:
            [Batch, Num_Heads, gh_q, gw_q, gh_k, gw_k]

        Interpretation: element [b, i, j, m, n] (post head-averaging) is
        "how much attention does socio-infra patch (i, j) pay to
        environment patch (m, n)". For a single query patch (i, j), slicing
        [:, i, j, :, :] gives a (gh_k, gw_k) heatmap that can be resized
        and overlaid on the original Stream A raster.
    """
    gh_q, gw_q = query_grid
    gh_k, gw_k = key_grid

    if average_heads:
        attn = attn_weights.mean(dim=1)  # [B, Lq, Lk]
        B, Lq, Lk = attn.shape
        assert Lq == gh_q * gw_q, "Query grid does not match sequence length."
        assert Lk == gh_k * gw_k, "Key grid does not match sequence length."
        return attn.view(B, gh_q, gw_q, gh_k, gw_k)
    else:
        B, num_heads, Lq, Lk = attn_weights.shape
        assert Lq == gh_q * gw_q, "Query grid does not match sequence length."
        assert Lk == gh_k * gw_k, "Key grid does not match sequence length."
        return attn_weights.view(B, num_heads, gh_q, gw_q, gh_k, gw_k)
