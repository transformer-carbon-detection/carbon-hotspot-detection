"""
================================================================================
model.py -- Multi-Stream Cross-Attention (MS-CA) Architecture (VMamba backbone)
================================================================================

Purpose
-------
Model architecture for the MS-CA satellite data fusion network:

    Stream A (Environment Proxy)          : [Batch, 3, H, W]
        Channels: NO2, CO, SO2
    Stream B (Socio-Infrastructure Proxy) : [Batch, 5, H, W]
        Channels: Nightlight, Population, Power Plants, Road Network,
                  Land Cover

Backbone upgrade: ViT -> Vision Mamba (VMamba)
-----------------------------------------------
The previous `ViTPatchEncoder` backbone has been replaced with
`VisionMambaEncoder`, which processes patch tokens using a Selective State
Space Model (SSM / "S6", the core operator behind Mamba) instead of
self-attention. Because an SSM is inherently a 1D sequential/causal
operator, naively flattening a 2D image into a single row-major sequence
would break spatial adjacency (e.g. a pixel's vertical neighbor ends up
`grid_w` steps away in the sequence, not 1 step away). VMamba solves this
with a **Cross-Scan Module (CSM)**: the same 2D grid of tokens is unrolled
into FOUR different 1D sequences (horizontal, vertical, and their reverses),
each SSM-processed independently (with shared weights), and then the four
outputs are un-scanned back to the original 2D layout and merged. This
lets every token receive contextual information "from all four cardinal
reading directions", approximating the 2D spatial adjacency that
self-attention gets for free but an SSM does not.

Design goals (unchanged from the ViT version)
----------------------------------------------
1. STRICT MODULARITY: Encoders are built behind the abstract `BaseEncoder`
   interface. `VisionMambaEncoder` is a drop-in implementation of that
   interface. To swap in yet another backbone later, subclass `BaseEncoder`,
   implement `forward()` / `num_patches` / `grid_size`, and pass the new
   class into `MSCANet(encoder_a_cls=..., encoder_b_cls=...)`. Nothing in
   the fusion layer or task heads needs to change.
2. EXPLAINABILITY: `CrossAttentionFusion` still explicitly returns the raw
   per-head attention score matrix for XAI, reshapeable via
   `reshape_attention_to_spatial_grid`.
3. MULTI-TASK LEARNING: unchanged -- pooled fused tokens feed a 5-class
   classification head and a scalar regression head.

A note on the SSM implementation
---------------------------------
This file ships a self-contained, pure-PyTorch **simplified** Selective
State Space block (`SimplifiedSSM`), so the model runs with zero external
dependencies beyond `torch`. If the official `mamba_ssm` package (with its
hardware-optimized fused CUDA selective-scan kernel) is available in your
training environment, you can swap `SimplifiedSSM`'s internals for
`mamba_ssm.Mamba` without touching `CrossScanModule`, `VisionMambaEncoder`,
or anything downstream -- the surrounding module boundary is intentionally
kept identical to `BaseEncoder`'s contract for exactly this reason.

This file intentionally contains NO dataset, training loop, or inference
demo code -- it is pure architecture, meant to be imported by separate
training/evaluation scripts (e.g. `from model import MSCANet`).
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    a CNN + flatten, etc. This is what lets the backbone be swapped
    (as was just done here, ViT -> VisionMamba) without touching any
    downstream code.
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


# ==============================================================================
# SECTION 2: CROSS-SCAN MODULE (CSM)
# ==============================================================================
class CrossScanModule:
    """
    Implements the 2D -> 1D Cross-Scan / Cross-Merge operators from the
    VMamba paper.

    An SSM processes a single 1D sequence causally/sequentially, so a plain
    row-major flatten of a 2D patch grid would only preserve horizontal
    adjacency, not vertical adjacency. The Cross-Scan Module fixes this by
    unrolling the SAME 2D grid into four different 1D sequences, each
    reading the image in a different direction:

        1. Horizontal        : row-major,        left -> right, top -> bottom
        2. Vertical           : column-major,     top -> bottom, left -> right
        3. Reverse-horizontal : direction 1, reversed end-to-start
        4. Reverse-vertical   : direction 2, reversed end-to-start

    Each of the four sequences is passed through the (weight-shared) SSM
    stack independently. `merge()` then undoes each direction's unrolling
    and averages the four results back into a single [B, N, D] tensor in
    the original horizontal (row-major) token order, so every output token
    has effectively "seen" its 2D neighborhood from all four cardinal
    directions.

    Implemented as plain static methods (no learnable parameters) since the
    Cross-Scan/Cross-Merge operators are pure reshaping/reindexing logic --
    all learnable weights live in the SSM blocks that process the scanned
    sequences.
    """

    @staticmethod
    def scan(x_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_grid: [Batch, grid_h, grid_w, Embed_Dim] -- spatial token grid.

        Returns:
            [Batch, 4, grid_h * grid_w, Embed_Dim] -- the 4 directional
            sequences, stacked along a new "direction" axis (index 1).
        """
        B, gh, gw, D = x_grid.shape

        # --- Direction 1: horizontal (row-major) scan ---
        seq_horizontal = x_grid.reshape(B, gh * gw, D)

        # --- Direction 2: vertical (column-major) scan ---
        # Swap the H and W axes, then flatten row-major over the swapped
        # grid -- this is equivalent to reading the image column by column.
        seq_vertical = x_grid.permute(0, 2, 1, 3).contiguous().reshape(B, gh * gw, D)

        # --- Direction 3: reverse-horizontal (mirror of Direction 1) ---
        seq_horizontal_rev = torch.flip(seq_horizontal, dims=[1])

        # --- Direction 4: reverse-vertical (mirror of Direction 2) ---
        seq_vertical_rev = torch.flip(seq_vertical, dims=[1])

        # Stack along a new "scan direction" dimension -> [B, 4, N, D]
        return torch.stack(
            [seq_horizontal, seq_vertical, seq_horizontal_rev, seq_vertical_rev],
            dim=1,
        )

    @staticmethod
    def merge(scanned_outputs: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Inverse of `scan()`: takes the four SSM-processed directional
        sequences (still in their own scan order) and merges them back
        into a single [Batch, N, Embed_Dim] tensor in the original
        horizontal row-major token order.

        Args:
            scanned_outputs: [Batch, 4, grid_h * grid_w, Embed_Dim]
            grid_size      : (grid_h, grid_w) of the original token grid.

        Returns:
            [Batch, grid_h * grid_w, Embed_Dim]
        """
        gh, gw = grid_size
        B, num_dirs, N, D = scanned_outputs.shape
        assert num_dirs == 4, "Expected exactly 4 cross-scan directions."

        out_h, out_v, out_h_rev, out_v_rev = scanned_outputs.unbind(dim=1)

        # Undo the reversal applied in scan() for directions 3 and 4, so
        # every sequence below is aligned to its own forward scan order.
        out_h_rev = torch.flip(out_h_rev, dims=[1])
        out_v_rev = torch.flip(out_v_rev, dims=[1])

        # out_h / out_h_rev are already in horizontal row-major order.
        # out_v / out_v_rev are in vertical column-major order and must be
        # transposed back to horizontal row-major order before merging.
        out_v = out_v.reshape(B, gw, gh, D).permute(0, 2, 1, 3).reshape(B, gh * gw, D)
        out_v_rev = out_v_rev.reshape(B, gw, gh, D).permute(0, 2, 1, 3).reshape(B, gh * gw, D)

        # Average the four directional views -> every token now carries
        # context accumulated from all four scan directions.
        merged = (out_h + out_v + out_h_rev + out_v_rev) / 4.0
        return merged  # [B, N, D], horizontal row-major order


# ==============================================================================
# SECTION 3: SIMPLIFIED SELECTIVE STATE SPACE MODEL (SSM) BLOCK
# ==============================================================================
class SimplifiedSSM(nn.Module):
    """
    A self-contained, pure-PyTorch simplified Selective State Space block
    (the "S6" operator behind Mamba). Processes ONE 1D sequence causally.

    Pipeline per token step t:
        1. Input projection splits the token into a main branch and a
           gating branch (like Mamba's gated MLP-style block).
        2. A short depthwise causal Conv1d mixes local neighboring tokens
           along the sequence before the SSM (as in the original Mamba
           block).
        3. Input-DEPENDENT SSM parameters (delta / B / C) are produced from
           the token itself -- this "selectivity" is what distinguishes
           Mamba from a plain time-invariant linear SSM (e.g. S4).
        4. The continuous-time SSM is discretized (zero-order-hold
           approximation) and unrolled sequentially over the sequence
           length to produce the per-token output.

    Note: this uses an explicit Python loop over the sequence length for
    the recurrence, which is simple and correct but not as fast as a
    fused/parallel hardware kernel. For heavy SSH training on large images
    or long sequences, swap this class's internals for the official
    `mamba_ssm.Mamba` CUDA kernel -- the input/output shapes
    ([Batch, Length, Embed_Dim] in, [Batch, Length, Embed_Dim] out) are
    kept identical so the swap requires no changes anywhere else.
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.inner_dim = expand * embed_dim
        # Rank of the input-dependent "delta" (timestep) projection --
        # a small bottleneck, as in the original Mamba design.
        self.dt_rank = max(embed_dim // 16, 4)

        # Projects the token into [main branch | gate branch], each of
        # size inner_dim (Mamba-style gated block).
        self.in_proj = nn.Linear(embed_dim, self.inner_dim * 2)

        # Depthwise causal-style local conv for short-range token mixing
        # before the SSM recurrence (padding trimmed to preserve length).
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.inner_dim,
        )

        # Produces the input-dependent delta (timestep), B, and C matrices.
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * state_dim)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim)

        # Learnable, input-INDEPENDENT state transition base (A). Stored in
        # log-space and negated at use-time (-exp(A_log)) to guarantee a
        # stable (strictly negative real part) continuous-time system,
        # following the Mamba/S4 initialization convention.
        A_log_init = torch.log(
            torch.arange(1, state_dim + 1, dtype=torch.float32)
        ).unsqueeze(0).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(A_log_init)  # [inner_dim, state_dim]

        # Skip/residual scaling term (as in Mamba's "D" parameter).
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        self.out_proj = nn.Linear(self.inner_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Length, Embed_Dim]
        Returns:
            [Batch, Length, Embed_Dim]
        """
        batch_size, seq_len, _ = x.shape

        # --- Gated input projection ---
        x_and_gate = self.in_proj(x)                          # [B, L, 2*inner]
        x_main, gate = x_and_gate.chunk(2, dim=-1)             # each [B, L, inner]

        # --- Local depthwise conv (causal: trim the extra right padding) ---
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = self.activation(x_conv)                       # [B, L, inner]

        # --- Input-dependent (selective) SSM parameters ---
        ssm_params = self.x_proj(x_conv)                       # [B, L, dt_rank + 2*state]
        delta_raw, B_mat, C_mat = torch.split(
            ssm_params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))            # [B, L, inner], > 0

        A = -torch.exp(self.A_log)                             # [inner, state], stable

        # --- Discretize (zero-order hold) ---
        # deltaA: [B, L, inner, state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # deltaB_x: [B, L, inner, state] -- input injected into the state
        deltaB_x = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * x_conv.unsqueeze(-1)

        # --- Sequential selective-scan recurrence ---
        # state_t = deltaA_t * state_{t-1} + deltaB_x_t
        # y_t     = C_t . state_t
        state = torch.zeros(
            batch_size, self.inner_dim, self.state_dim, device=x.device, dtype=x.dtype
        )
        outputs = []
        for t in range(seq_len):
            state = deltaA[:, t] * state + deltaB_x[:, t]       # [B, inner, state]
            y_t = (state * C_mat[:, t].unsqueeze(1)).sum(dim=-1)  # [B, inner]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)                         # [B, L, inner]

        # --- Skip connection (Mamba's D term) + gating ---
        y = y + x_conv * self.D
        y = y * self.activation(gate)

        out = self.out_proj(y)                                  # [B, L, embed_dim]
        out = self.dropout(out)
        return out


class MambaBlock(nn.Module):
    """Pre-norm residual wrapper around a single SimplifiedSSM layer."""

    def __init__(self, embed_dim: int, state_dim: int = 16, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.ssm = SimplifiedSSM(embed_dim, state_dim=state_dim, expand=expand, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ==============================================================================
# SECTION 4: VISION MAMBA ENCODER (Patch Embed + Cross-Scan + SSM stack)
# ==============================================================================
class VisionMambaEncoder(BaseEncoder):
    """
    Vision Mamba (VMamba) backbone.

    Pipeline:
        1. Conv2d patch embedding: [B, C, H, W] -> [B, grid_h, grid_w, D]
        2. Add a 2D learned positional embedding (added BEFORE scanning, so
           positional information travels with each token regardless of
           which of the 4 directions later reads it).
        3. Cross-Scan Module unrolls the grid into 4 directional sequences:
           [B, 4, N, D].
        4. All 4 sequences are processed by the SAME (weight-shared) stack
           of `MambaBlock`s, batched together as [B*4, N, D] for
           efficiency.
        5. Cross-Merge Module folds the 4 processed sequences back into a
           single [B, N, D] spatial token sequence.
        6. Final LayerNorm + dropout.

    Output contract matches `BaseEncoder`: [Batch, Sequence_Length,
    Embed_Dim], with Sequence_Length = grid_h * grid_w, identical to what
    the previous ViT encoder produced -- so `CrossAttentionFusion` and the
    task heads require zero changes.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int],
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 2,
        state_dim: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
        **_unused_kwargs,
    ):
        """
        Args:
            in_channels : number of input channels for this stream
                          (3 for Stream A, 5 for Stream B).
            img_size    : (H, W) of the raw input raster.
            patch_size  : side length of each square patch.
            embed_dim   : token embedding dimension.
            depth       : number of stacked MambaBlocks.
            state_dim   : SSM latent state size per channel.
            expand      : inner expansion factor inside each MambaBlock
                          (mirrors Mamba's "expand" hyperparameter).
            dropout     : dropout probability used throughout the block.
            **_unused_kwargs: absorbs any extra keyword arguments (e.g.
                          `num_heads`) that a caller written for a
                          different backbone (such as a ViT) might still
                          pass in -- keeps `MSCANet` fully backbone-agnostic
                          without requiring every encoder to share an
                          identical constructor signature.
        """
        super().__init__(embed_dim)
        img_h, img_w = img_size
        assert img_h % patch_size == 0 and img_w % patch_size == 0, (
            "Image dimensions must be divisible by patch_size."
        )

        self.patch_size = patch_size
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size
        self._num_patches = self.grid_h * self.grid_w

        # Patch embedding: strided conv turns each non-overlapping
        # patch_size x patch_size patch into one embed_dim-length vector.
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2D learned positional embedding, kept in grid shape so it can be
        # added directly to the [B, gh, gw, D] tensor BEFORE cross-scanning
        # (adding it after scanning would misalign positions for the
        # vertical/reverse scan orders).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_h, self.grid_w, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # A single shared stack of MambaBlocks is applied to all 4 scan
        # directions (weight sharing across directions), consistent with
        # VMamba's "SS2D" design -- this keeps parameter count in line
        # with a comparable single-direction sequence model.
        self.mamba_blocks = nn.ModuleList(
            [
                MambaBlock(embed_dim, state_dim=state_dim, expand=expand, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self.grid_h, self.grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        batch_size = x.shape[0]

        x = self.patch_embed(x)                      # [B, D, gh, gw]
        x = x.permute(0, 2, 3, 1)                     # [B, gh, gw, D]
        x = x + self.pos_embed                        # broadcast add 2D positions

        # --- Cross-Scan: 2D grid -> 4 directional 1D sequences ---
        scanned = CrossScanModule.scan(x)              # [B, 4, N, D]

        # Fold the "direction" axis into the batch axis so all 4 scans are
        # processed by the shared MambaBlock stack in a single batched pass.
        _, num_dirs, seq_len, embed_dim = scanned.shape
        scanned = scanned.reshape(batch_size * num_dirs, seq_len, embed_dim)

        for block in self.mamba_blocks:
            scanned = block(scanned)
        scanned = self.norm(scanned)

        # Restore the [B, 4, N, D] shape for the Cross-Merge step.
        scanned = scanned.reshape(batch_size, num_dirs, seq_len, embed_dim)

        # --- Cross-Merge: 4 directional sequences -> single spatial sequence ---
        merged = CrossScanModule.merge(scanned, (self.grid_h, self.grid_w))  # [B, N, D]
        merged = self.dropout(merged)
        return merged


# ==============================================================================
# SECTION 5: CROSS-ATTENTION FUSION LAYER
# ==============================================================================
class CrossAttentionFusion(nn.Module):
    """
    Fuses the two encoded streams via cross-attention.

        Query (Q)       <- Stream B tokens (Socio-Infrastructure)
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
# SECTION 6: MULTI-TASK OUTPUT HEADS
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
# SECTION 7: FULL MS-CA MODEL
# ==============================================================================
class MSCANet(nn.Module):
    """
    Full Multi-Stream Cross-Attention network, now defaulting to the
    Vision Mamba backbone for both streams.

    Composition (all injected as sub-modules, nothing hard-wired):
        encoder_a  : BaseEncoder subclass for Stream A (Environment)
        encoder_b  : BaseEncoder subclass for Stream B (Socio-Infra)
        fusion     : CrossAttentionFusion
        cls_head   : ClassificationHead
        reg_head   : RegressionHead

    To swap the backbone again later (e.g. back to a ViT, or to some future
    architecture), only the `encoder_a_cls` / `encoder_b_cls` constructor
    arguments need to change -- the fusion layer and task heads are
    completely backbone-agnostic.
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
        encoder_a_cls=VisionMambaEncoder,
        encoder_b_cls=VisionMambaEncoder,
        **encoder_kwargs,
    ):
        """
        Args:
            img_size, patch_size, embed_dim, encoder_depth, dropout: shared
                hyperparameters forwarded to both stream encoders.
            num_heads: number of attention heads used ONLY by the
                cross-attention fusion layer (the Mamba encoders do not use
                multi-head attention internally).
            num_classes: number of hotspot grading classes.
            encoder_a_cls / encoder_b_cls: `BaseEncoder` subclasses used to
                build Stream A's and Stream B's encoders respectively --
                this is the single point of control for backbone swaps.
            **encoder_kwargs: extra backbone-specific hyperparameters
                (e.g. `state_dim`, `expand` for VisionMambaEncoder) that
                are forwarded verbatim to both encoder constructors. This
                keeps MSCANet decoupled from any one backbone's specific
                hyperparameter set.
        """
        super().__init__()
        self.embed_dim = embed_dim

        common_encoder_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            dropout=dropout,
            **encoder_kwargs,
        )

        # Independent encoder instances -> weights are NOT shared between
        # Stream A and Stream B, since the two modalities have different
        # statistics.
        #
        # Stream A is finalized to carry 3 channels (NO2, CO, SO2), so
        # encoder_a uses in_channels=3. Stream B remains at in_channels=5
        # (Nightlight, Population, Power Plants, Road Network, Land Cover).
        self.encoder_a = encoder_a_cls(in_channels=3, **common_encoder_kwargs)
        self.encoder_b = encoder_b_cls(in_channels=5, **common_encoder_kwargs)

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
# SECTION 8: XAI HELPER -- RESHAPE ATTENTION BACK TO A 2D SPATIAL MAP
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
