"""
================================================================================
train.py -- Cloud-Ready Training Loop for the MS-CA (Vision Mamba) Network
================================================================================

Purpose
-------
End-to-end training script for `MSCANet` (from `model.py`) on
`SpatialGridDataset` (from `spatial_grid_dataset.py`), designed to run
unmodified on a local machine, Colab, or a remote AWS/GCP instance.

Design goals
------------
1. NO HARDCODED PATHS: every filesystem path (data, checkpoint output) is
   supplied via `argparse`. The script never assumes a specific mount
   point, drive letter, or working directory.
2. SYNCHRONIZED D4 AUGMENTATION: Stream A and Stream B receive the exact
   same random rotation/flip per sample, applied via a custom
   `collate_fn`, so spatial alignment between the two modalities is never
   broken (critical for the cross-attention fusion layer).
3. UNCERTAINTY-WEIGHTED MULTI-TASK LOSS: `DynamicMultiTaskLoss` implements
   Kendall et al. (2018)'s homoscedastic uncertainty weighting, combining
   Focal Loss (classification) and Huber/Smooth-L1 Loss (regression) with
   two learnable log-variance parameters.
4. MAMBA-AWARE OPTIMIZER: AdamW parameter groups explicitly exclude the
   SSM's `A_log` / `D` parameters, Conv1d biases, and all other
   non-`nn.Linear` weights from weight decay.
5. CLOUD-SAFE CHECKPOINTING: every epoch (and whenever validation loss
   improves) the model/optimizer/scheduler/loss state is saved locally
   AND synced to Weights & Biases, so a preemptible cloud instance can be
   killed and resumed without losing progress (`--resume_from`).

Usage
-----
    python train.py \\
        --data_path /path/to/monthly_grid_data.csv \\
        --output_dir /path/to/checkpoints \\
        --wandb_project ms-ca-fusion \\
        --epochs 50 --batch_size 32

Run `python train.py --help` for the full list of configurable options.
================================================================================
"""

import argparse
import math
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MSCANet
from spatial_grid_dataset import SpatialGridDataset

# WandB is treated as an optional dependency: if it isn't installed (or the
# user passes --disable_wandb), the script degrades gracefully to
# console-only logging instead of crashing.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ==============================================================================
# SECTION 0: ARGUMENT PARSING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cloud-ready training script for the MS-CA Vision Mamba network."
    )

    # --- Data / I/O (no hardcoded paths -- everything supplied by the caller) ---
    parser.add_argument("--data_path", type=str, required=True,
                         help="Path to the joined CSV file consumed by SpatialGridDataset.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                         help="Local directory to write checkpoints to.")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Optional path to a checkpoint (.pth) to resume training from.")

    # --- Dataset / augmentation ---
    parser.add_argument("--patch_size", type=int, default=16,
                         help="Side length of the square crop drawn by SpatialGridDataset.")
    parser.add_argument("--train_samples", type=int, default=10000,
                         help="Virtual epoch size for the training split.")
    parser.add_argument("--val_samples", type=int, default=2000,
                         help="Virtual epoch size for the validation split.")

    # --- Model architecture (forwarded to MSCANet / VisionMambaEncoder) ---
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4,
                         help="Attention heads used ONLY by the cross-attention fusion layer.")
    parser.add_argument("--state_dim", type=int, default=16,
                         help="SSM latent state size (VisionMambaEncoder).")
    parser.add_argument("--expand", type=int, default=2,
                         help="Inner expansion factor inside each MambaBlock.")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)

    # --- Optimization ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=5.0,
                         help="Linear warm-up duration, in epochs (may be fractional).")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                         help="Max gradient norm for clipping (<=0 disables clipping).")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                         help="Focusing parameter for Focal Loss.")
    parser.add_argument("--use_amp", action="store_true",
                         help="Enable automatic mixed precision (torch.cuda.amp).")

    # --- Cloud / DataLoader ---
    parser.add_argument("--num_workers", type=int, default=4,
                         help="DataLoader worker processes (tune per cloud VM's vCPU count).")
    parser.add_argument("--device", type=str, default=None,
                         help="Force a device string (e.g. 'cuda:0', 'cpu'). "
                              "Defaults to CUDA if available, else CPU.")
    parser.add_argument("--seed", type=int, default=42)

    # --- Weights & Biases ---
    parser.add_argument("--wandb_project", type=str, default="ms-ca-satellite-fusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true",
                         help="Run with console-only logging, no WandB calls at all.")

    return parser.parse_args()


# ==============================================================================
# SECTION 1: REPRODUCIBILITY
# ==============================================================================
def set_global_seed(seed: int) -> None:
    """Seeds every RNG the training pipeline touches for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# SECTION 2: SYNCHRONIZED D4 AUGMENTATION
# ==============================================================================
def _apply_d4_transform(
    stream_a: torch.Tensor, stream_b: torch.Tensor, k_rot: int, flip_h: bool, flip_v: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies ONE randomly-chosen D4 symmetry transform (0/90/180/270 degree
    rotation, optional horizontal flip, optional vertical flip) IDENTICALLY
    to both streams.

    CRITICAL: stream_a and stream_b are transformed with the exact same
    (k_rot, flip_h, flip_v) triple, so a pixel at position (i, j) in both
    tensors always ends up at the same new position (i', j') after the
    transform -- preserving the spatial alignment the cross-attention
    fusion layer depends on.

    Args:
        stream_a, stream_b: [C, patch_size, patch_size] tensors.
        k_rot  : number of 90-degree rotations to apply (0-3).
        flip_h : whether to flip along the height axis.
        flip_v : whether to flip along the width axis.
    """
    if k_rot != 0:
        stream_a = torch.rot90(stream_a, k=k_rot, dims=(-2, -1))
        stream_b = torch.rot90(stream_b, k=k_rot, dims=(-2, -1))
    if flip_h:
        stream_a = torch.flip(stream_a, dims=(-2,))
        stream_b = torch.flip(stream_b, dims=(-2,))
    if flip_v:
        stream_a = torch.flip(stream_a, dims=(-1,))
        stream_b = torch.flip(stream_b, dims=(-1,))
    return stream_a, stream_b


def d4_augmentation_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]
):
    """
    Custom `collate_fn` for the TRAINING DataLoader.

    For every sample in the batch, draws one random D4 symmetry transform
    (8 possible combinations: 4 rotations x 2 flip states, since a
    horizontal+vertical flip is equivalent to a 180-degree rotation
    composed with a single flip -- sampling both flip flags independently
    still covers the full D4 group with correct probabilities) and applies
    it identically to that sample's Stream A and Stream B crop before
    batching.

    Returns:
        stream_a_batch : [B, 3, patch, patch]
        stream_b_batch : [B, 5, patch, patch]
        label_batch    : [B]              (long)
        reg_batch      : [B, 1]           (float)
    """
    augmented_a, augmented_b, labels, regs = [], [], [], []

    for stream_a, stream_b, label, reg_target in batch:
        k_rot = random.randint(0, 3)
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5

        a_aug, b_aug = _apply_d4_transform(stream_a, stream_b, k_rot, flip_h, flip_v)
        augmented_a.append(a_aug)
        augmented_b.append(b_aug)
        labels.append(label)
        regs.append(reg_target)

    stream_a_batch = torch.stack(augmented_a, dim=0)
    stream_b_batch = torch.stack(augmented_b, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.long)
    reg_batch = torch.stack(regs, dim=0)
    return stream_a_batch, stream_b_batch, label_batch, reg_batch


def plain_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]
):
    """
    Un-augmented collate function used for the VALIDATION DataLoader, so
    validation metrics reflect the model's performance on un-distorted
    crops (standard practice: augment train, not val/test).
    """
    stream_a, stream_b, labels, regs = zip(*batch)
    stream_a_batch = torch.stack(stream_a, dim=0)
    stream_b_batch = torch.stack(stream_b, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.long)
    reg_batch = torch.stack(regs, dim=0)
    return stream_a_batch, stream_b_batch, label_batch, reg_batch


# ==============================================================================
# SECTION 3: DYNAMIC MULTI-TASK LOSS (Kendall et al., 2018)
# ==============================================================================
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss, used to counter emission-hotspot class
    imbalance (most patches are "no hotspot", a few are severe).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    where p_t is the model's predicted probability for the TRUE class.
    Down-weights easy, already-well-classified examples so the loss
    focuses on hard/rare examples during training.
    """

    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        # Optional per-class weight vector (the "alpha" term), e.g.
        # inverse class frequency -- left as None (uniform) by default.
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy per-sample (no reduction yet), optionally
        # class-weighted.
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction="none"
        )
        p_t = torch.exp(-ce_loss)  # probability assigned to the true class
        focal_term = (1.0 - p_t).pow(self.gamma)
        return (focal_term * ce_loss).mean()


class DynamicMultiTaskLoss(nn.Module):
    """
    Homoscedastic uncertainty-weighted multi-task loss, following
    Kendall, Gal & Cipolla (2018) "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics".

    Target formulation (as specified):

        TotalLoss = (1 / (2*sigma_1^2)) * FocalLoss  + log(sigma_1)
                  + (1 / (2*sigma_2^2)) * HuberLoss   + log(sigma_2)

    Numerically-stable log-variance parameterization
    --------------------------------------------------
    Directly learning sigma_1/sigma_2 as raw parameters risks them
    drifting to zero or negative values (undefined for 1/sigma^2 and
    log(sigma)). Instead, following the reference implementation's
    standard trick, we learn:

        s_i = log(sigma_i^2)                     (a free real-valued parameter)

    which gives, algebraically:

        1 / (2*sigma_i^2) = 0.5 * exp(-s_i)
        log(sigma_i)      = 0.5 * s_i

    so each task term becomes:

        0.5 * exp(-s_i) * Loss_i + 0.5 * s_i

    This is EXACTLY the requested formula, just re-expressed through
    `s_i = log(sigma_i^2)` for unconstrained, stable optimization --
    `s_i` initialized to 0.0 corresponds to sigma_i = 1.0 (a "sensible
    default" that starts both tasks with equal, unit weighting).
    """

    def __init__(
        self,
        num_classes: int,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        huber_beta: float = 1.0,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
        # SmoothL1Loss with beta=1.0 is mathematically equivalent to the
        # classic Huber Loss with delta=1.0 -- used here for its
        # robustness to the extreme emission-value outliers mentioned in
        # the spec, and for broad PyTorch-version compatibility.
        self.huber_loss = nn.SmoothL1Loss(beta=huber_beta, reduction="mean")

        # Learnable log-variance parameters: s_i = log(sigma_i^2).
        # Init to 0.0 -> sigma_1 = sigma_2 = 1.0 at the start of training.
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_out: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
    ):
        loss_cls_raw = self.focal_loss(cls_logits, cls_targets)
        loss_reg_raw = self.huber_loss(reg_out, reg_targets)

        precision_cls = torch.exp(-self.log_var_cls)  # = 1 / sigma_cls^2
        precision_reg = torch.exp(-self.log_var_reg)  # = 1 / sigma_reg^2

        term_cls = 0.5 * precision_cls * loss_cls_raw + 0.5 * self.log_var_cls
        term_reg = 0.5 * precision_reg * loss_reg_raw + 0.5 * self.log_var_reg

        total_loss = (term_cls + term_reg).squeeze()

        # Return the raw (unweighted) sub-losses too, purely for logging/
        # interpretability -- detached so they don't hold onto the graph.
        diagnostics = {
            "loss_cls_raw": loss_cls_raw.detach(),
            "loss_reg_raw": loss_reg_raw.detach(),
            "sigma_cls": torch.exp(0.5 * self.log_var_cls).detach(),
            "sigma_reg": torch.exp(0.5 * self.log_var_reg).detach(),
        }
        return total_loss, diagnostics


# ==============================================================================
# SECTION 4: MAMBA-AWARE OPTIMIZER PARAMETER GROUPING
# ==============================================================================
def build_param_groups(
    model: nn.Module, loss_fn: nn.Module, weight_decay: float
) -> List[dict]:
    """
    Splits ALL trainable parameters (model + loss module) into two AdamW
    parameter groups:

        decay_params    : weight decay applied. ONLY `nn.Linear.weight`
                           tensors qualify -- this is the "standard linear
                           layer weights" the spec asks to decay.
        no_decay_params : weight decay explicitly disabled. This
                           automatically and correctly captures:
                             - `SimplifiedSSM.A_log` (the SSM's state
                               transition matrix, in log-space)
                             - `SimplifiedSSM.D` (the skip-connection term)
                             - all Conv1d/Conv2d weights AND biases
                               (including the depthwise causal Conv1d
                               biases called out in the spec)
                             - all LayerNorm weights/biases
                             - all nn.Linear biases
                             - positional embeddings (`pos_embed`)
                             - the loss module's learnable log-variance
                               parameters (`log_var_cls`, `log_var_reg`)

    Walking `named_modules()` and pulling only `recurse=False` parameters
    off each module (rather than iterating `named_parameters()` directly)
    is what lets us key the decay decision off the PARENT MODULE'S TYPE
    (`isinstance(module, nn.Linear)`) instead of fragile name-string
    matching -- while still visiting every parameter exactly once.
    """
    decay_params, no_decay_params = [], []

    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, nn.Linear) and param_name == "weight":
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    # The loss module's learnable uncertainty-weighting parameters are
    # scalars -- weight decay on them would fight against the very
    # uncertainty estimation Kendall et al.'s formulation relies on, so
    # they always go in the no-decay group.
    for param in loss_fn.parameters():
        if param.requires_grad:
            no_decay_params.append(param)

    print(
        f"[Optimizer] {len(decay_params)} tensors WITH weight decay "
        f"(nn.Linear weights), {len(no_decay_params)} tensors WITHOUT "
        f"(A_log, D, conv weights/biases, norms, embeddings, loss params)."
    )

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# ==============================================================================
# SECTION 5: LINEAR WARM-UP + COSINE ANNEALING SCHEDULER
# ==============================================================================
def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Builds a per-STEP (not per-epoch) learning-rate multiplier schedule:
      - Steps [0, warmup_steps)      : linear ramp from ~0 -> 1.0x base LR.
      - Steps [warmup_steps, total)  : cosine decay from 1.0x -> ~0x base LR.

    Implemented as a single LambdaLR so it works identically across PyTorch
    versions without depending on `SequentialLR` (added in more recent
    releases), which matters for reproducibility across heterogeneous
    cloud images.
    """
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ==============================================================================
# SECTION 6: CHECKPOINTING (LOCAL + WANDB SYNC)
# ==============================================================================
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: nn.Module,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    wandb_run,
) -> None:
    """
    Saves a full training checkpoint locally, then syncs it to WandB (if
    enabled) so training state survives a preemptible-instance eviction.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, path)

    if wandb_run is not None:
        # wandb.save() sets up a symlink/watch that uploads the file
        # (and any future overwrites of it) to the run's cloud storage --
        # this is what protects the checkpoint if the local disk is on a
        # preemptible/ephemeral cloud instance.
        wandb_run.save(path, base_path=os.path.dirname(path))


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[int, float]:
    """Restores full training state from a checkpoint. Returns (start_epoch, best_val_loss)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    print(f"[Resume] Loaded checkpoint from '{path}' -> resuming at epoch {start_epoch}.")
    return start_epoch, best_val_loss


# ==============================================================================
# SECTION 7: TRAIN / VALIDATE FOR ONE EPOCH
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    grad_clip: float,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    epoch: int,
    wandb_run,
    log_every: int = 50,
) -> dict:
    model.train()
    running_total, running_cls, running_reg = 0.0, 0.0, 0.0
    num_batches = len(loader)

    for step, (stream_a, stream_b, cls_label, reg_target) in enumerate(loader):
        stream_a = stream_a.to(device, non_blocking=True)
        stream_b = stream_b.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)
        reg_target = reg_target.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        # Training forward pass intentionally skips `return_attention` --
        # the attention weights aren't needed for the loss, and computing
        # need_weights=True is extra overhead we don't want on every step.
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            cls_logits, reg_out = model(stream_a, stream_b)
            total_loss, diagnostics = loss_fn(cls_logits, reg_out, cls_label, reg_target)

        if use_amp:
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()  # per-step schedule (warmup + cosine)

        running_total += total_loss.item()
        running_cls += diagnostics["loss_cls_raw"].item()
        running_reg += diagnostics["loss_reg_raw"].item()

        if wandb_run is not None and step % log_every == 0:
            wandb_run.log({
                "train/step_total_loss": total_loss.item(),
                "train/step_focal_loss": diagnostics["loss_cls_raw"].item(),
                "train/step_huber_loss": diagnostics["loss_reg_raw"].item(),
                "train/sigma_cls": diagnostics["sigma_cls"].item(),
                "train/sigma_reg": diagnostics["sigma_reg"].item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })

        if step % log_every == 0:
            print(
                f"  [Epoch {epoch}] step {step}/{num_batches} "
                f"total_loss={total_loss.item():.4f} "
                f"focal={diagnostics['loss_cls_raw'].item():.4f} "
                f"huber={diagnostics['loss_reg_raw'].item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return {
        "avg_total_loss": running_total / num_batches,
        "avg_focal_loss": running_cls / num_batches,
        "avg_huber_loss": running_reg / num_batches,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Tuple[dict, torch.Tensor]:
    """
    Runs a full validation pass. Also exercises the XAI attention-return
    path (`return_attention=True`) once per epoch on the first batch, to
    continuously verify that:
      (a) the raw attention weight tensor comes back with the expected
          shape, and
      (b) it is immediately detached + moved to CPU so it never keeps a
          live autograd graph or GPU tensor alive beyond this function --
          preventing the kind of silent memory leak that can otherwise
          creep in when XAI/visualization hooks are bolted onto a
          training loop.
    """
    model.eval()
    running_total, running_cls, running_reg = 0.0, 0.0, 0.0
    num_batches = len(loader)
    sample_attention_weights = None

    for step, (stream_a, stream_b, cls_label, reg_target) in enumerate(loader):
        stream_a = stream_a.to(device, non_blocking=True)
        stream_b = stream_b.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)
        reg_target = reg_target.to(device, non_blocking=True).float()

        if step == 0:
            # XAI hook readiness check -- see docstring above.
            cls_logits, reg_out, attn_weights = model(stream_a, stream_b, return_attention=True)
            sample_attention_weights = attn_weights.detach().cpu()
            del attn_weights  # drop the GPU-resident reference explicitly
        else:
            cls_logits, reg_out = model(stream_a, stream_b)

        total_loss, diagnostics = loss_fn(cls_logits, reg_out, cls_label, reg_target)

        running_total += total_loss.item()
        running_cls += diagnostics["loss_cls_raw"].item()
        running_reg += diagnostics["loss_reg_raw"].item()

    metrics = {
        "avg_total_loss": running_total / num_batches,
        "avg_focal_loss": running_cls / num_batches,
        "avg_huber_loss": running_reg / num_batches,
    }

    print(
        f"[Epoch {epoch}] VALIDATION: "
        f"total_loss={metrics['avg_total_loss']:.4f} "
        f"focal={metrics['avg_focal_loss']:.4f} "
        f"huber={metrics['avg_huber_loss']:.4f} "
        f"| XAI attention shape check: {tuple(sample_attention_weights.shape)} "
        f"(expected [B, num_heads, Lq, Lk])"
    )

    return metrics, sample_attention_weights


# ==============================================================================
# SECTION 8: MAIN
# ==============================================================================
def main():
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Setup] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    best_ckpt_path = os.path.join(args.output_dir, "checkpoint_best.pth")

    # ---------------------------------------------------------------- #
    # WandB setup (gracefully degrades to console-only if unavailable)
    # ---------------------------------------------------------------- #
    wandb_run = None
    if not args.disable_wandb:
        if _WANDB_AVAILABLE:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
        else:
            print(
                "[Warning] WandB is not installed but --disable_wandb was not set. "
                "Continuing with console-only logging. Run `pip install wandb` to enable it."
            )

    # ---------------------------------------------------------------- #
    # Datasets / DataLoaders
    # ---------------------------------------------------------------- #
    # NOTE: both splits are drawn (via random crops) from the SAME
    # underlying monthly raster, just with different RNG seeds and
    # virtual sizes. This is a pragmatic choice for a single-month, dense
    # spatial dataset -- for a stricter evaluation protocol later, split
    # the raw DataFrame by spatial region (e.g. holding out a contiguous
    # block of rows/cols) before constructing two SpatialGridDataset
    # instances.
    train_dataset = SpatialGridDataset(
        data_source=args.data_path,
        patch_size=args.patch_size,
        num_samples=args.train_samples,
        seed=args.seed,
    )
    val_dataset = SpatialGridDataset(
        data_source=args.data_path,
        patch_size=args.patch_size,
        num_samples=args.val_samples,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=d4_augmentation_collate_fn,  # synchronized A/B augmentation
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=plain_collate_fn,  # no augmentation at validation time
        persistent_workers=args.num_workers > 0,
    )

    print(
        f"[Data] Grid dims (H, W) = ({train_dataset.H}, {train_dataset.W}) | "
        f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} | "
        f"steps/epoch(train)={len(train_loader)}"
    )

    # ---------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------- #
    model = MSCANet(
        img_size=(train_dataset.H, train_dataset.W),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        dropout=args.dropout,
        state_dim=args.state_dim,
        expand=args.expand,
    )

    if torch.cuda.device_count() > 1:
        print(f"[Setup] Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)
        
    model = model.to(device)

    loss_fn = DynamicMultiTaskLoss(
        num_classes=args.num_classes, focal_gamma=args.focal_gamma
    ).to(device)

    if wandb_run is not None:
        wandb_run.watch(model, log="gradients", log_freq=200)

    # ---------------------------------------------------------------- #
    # Optimizer + Scheduler
    # ---------------------------------------------------------------- #
    param_groups = build_param_groups(model, loss_fn, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)

    # ---------------------------------------------------------------- #
    # Optionally resume from a checkpoint (cloud preemption recovery)
    # ---------------------------------------------------------------- #
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_from is not None and os.path.isfile(args.resume_from):
        start_epoch, best_val_loss = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, loss_fn, device
        )

    # ---------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------- #
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(
            model, loss_fn, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, scaler=scaler, use_amp=args.use_amp,
            epoch=epoch, wandb_run=wandb_run,
        )
        val_metrics, _sample_attn = validate_one_epoch(
            model, loss_fn, val_loader, device, epoch=epoch
        )

        epoch_time = time.time() - epoch_start_time
        print(
            f"[Epoch {epoch}] DONE in {epoch_time:.1f}s | "
            f"train_loss={train_metrics['avg_total_loss']:.4f} | "
            f"val_loss={val_metrics['avg_total_loss']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/epoch_total_loss": train_metrics["avg_total_loss"],
                "train/epoch_focal_loss": train_metrics["avg_focal_loss"],
                "train/epoch_huber_loss": train_metrics["avg_huber_loss"],
                "val/epoch_total_loss": val_metrics["avg_total_loss"],
                "val/epoch_focal_loss": val_metrics["avg_focal_loss"],
                "val/epoch_huber_loss": val_metrics["avg_huber_loss"],
                "epoch_time_sec": epoch_time,
            })

        # --- Cloud-safe checkpointing: always save "latest", and save a
        # separate "best" copy whenever validation loss improves. Both are
        # synced to WandB so a preempted instance never loses progress. ---
        save_checkpoint(
            latest_ckpt_path, model, optimizer, scheduler, loss_fn,
            epoch, best_val_loss, args, wandb_run,
        )

        if val_metrics["avg_total_loss"] < best_val_loss:
            best_val_loss = val_metrics["avg_total_loss"]
            save_checkpoint(
                best_ckpt_path, model, optimizer, scheduler, loss_fn,
                epoch, best_val_loss, args, wandb_run,
            )
            print(f"[Checkpoint] New best val loss {best_val_loss:.4f} -> saved '{best_ckpt_path}'.")

    print("[Done] Training complete.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
