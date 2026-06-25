"""
src/models.py
=============
SegFormer architecture integration for 4-channel BraTS-PEDs segmentation.

Public API
----------
get_segformer(model_checkpoint, num_classes) → SegFormerWrapper
SegFormerWrapper                              → nn.Module (drop-in compatible)

4-channel adaptation
--------------------
The pretrained ImageNet first patch embedding projection has shape
[C_out, 3, k, k] (RGB input). We expand this to [C_out, 4, k, k] by
keeping the three existing RGB channel weights unchanged and initialising
the new 4th channel as the mean of the three RGB channels. This preserves
the pretrained feature representations for the existing channels while giving
the new MRI-specific channel a sensible starting point.

Drop-in compatibility with train_utils.py
------------------------------------------
SegFormerWrapper exposes:
  - .encoder     → model.segformer.encoder   (for set_encoder_trainable)
  - .decode_head → model.decode_head         (for differential LR in Phase 2)
  - forward()    upsamples logits to input resolution [B, C, H, W]
    so train_one_epoch() and evaluate() work without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class SegFormerWrapper(nn.Module):
    """Thin wrapper around HuggingFace SegformerForSemanticSegmentation.

    Makes SegFormer a drop-in replacement for segmentation-models-pytorch
    models inside the BraTS training pipeline (train_utils.py).

    Adaptations
    -----------
    * forward() upsamples SegFormer logits from [B, C, H/4, W/4] back to
      [B, C, H, W] so CombinedLoss and metric functions see the correct
      spatial resolution.
    * .encoder property exposes segformer.encoder for set_encoder_trainable().
    * .decode_head property exposes decode_head for differential LR setup.
    """

    def __init__(self, hf_model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = hf_model

    # ------------------------------------------------------------------
    # Convenience properties for compatibility with train_utils.py
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> nn.Module:
        """Encoder backbone — used by set_encoder_trainable().

        transformers >=5 flattened the SegFormer backbone: the old
        ``segformer.encoder`` no longer exists (the stages live directly under
        ``segformer``). Fall back to ``segformer`` itself in that case.
        """
        seg = self.model.segformer
        return seg.encoder if hasattr(seg, "encoder") else seg

    @property
    def decode_head(self) -> nn.Module:
        """Decode head + classifier — used for differential LR in Phase 2."""
        return self.model.decode_head

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run SegFormer and upsample logits to the input spatial resolution.

        SegFormer natively produces logits at 1/4 of the input resolution
        due to its hierarchical transformer patch embeddings. This method
        bilinearly upsamples them back to (H, W) so the output shape matches
        what CombinedLoss and _compute_batch_dice() expect.

        Args:
            pixel_values: Float tensor of shape [B, 4, H, W].

        Returns:
            Logits float tensor of shape [B, num_classes, H, W].
        """
        _, _, H, W = pixel_values.shape
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits  # [B, num_classes, H, W]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_segformer(
    model_checkpoint: str = "nvidia/mit-b1",
    num_classes: int = 4,
) -> SegFormerWrapper:
    """Build a SegFormer model adapted for 4-channel MRI input.

    Steps
    -----
    1. Load pretrained SegformerForSemanticSegmentation from HuggingFace Hub.
       The decode head classifier is re-initialised for ``num_classes`` outputs
       (``ignore_mismatched_sizes=True`` handles the ADE20K→4-class mismatch).
    2. Expand the first patch embedding projection from 3 → 4 input channels:
       - Channels 0-2 keep their original pretrained RGB weights exactly.
       - Channel 3 is initialised as the mean of channels 0-2, providing a
         sensible starting response for the additional MRI modality.
    3. Wrap in SegFormerWrapper for drop-in compatibility with train_utils.py.

    Args:
        model_checkpoint: HuggingFace model identifier (default ``"nvidia/mit-b1"``).
        num_classes:       Number of segmentation output classes (default 4).

    Returns:
        SegFormerWrapper ready for training on 4-channel [B, 4, H, W] inputs.
    """
    # ── 1. Load pretrained weights; re-init head for num_classes outputs ──
    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # ── 2. Expand first patch embedding: [C_out, 3, k, k] → [C_out, 4, k, k] ──
    # transformers <5 : segformer.encoder.patch_embeddings[0] (ModuleList)
    # transformers >=5: segformer.stages[0].patch_embeddings (per-stage)
    seg = hf_model.segformer
    if hasattr(seg, "stages"):
        patch_embed = seg.stages[0].patch_embeddings
    else:
        patch_embed = seg.encoder.patch_embeddings[0]
    proj = patch_embed.proj
    old_weight = proj.weight.data.clone()  # [C_out, 3, kH, kW]
    C_out, _, kH, kW = old_weight.shape

    new_weight = torch.zeros(C_out, 4, kH, kW, dtype=old_weight.dtype)
    new_weight[:, :3, :, :] = old_weight               # preserve RGB weights
    new_weight[:, 3:4, :, :] = old_weight.mean(dim=1, keepdim=True)  # 4th = mean(RGB)

    new_proj = nn.Conv2d(
        in_channels=4,
        out_channels=C_out,
        kernel_size=(kH, kW),
        stride=proj.stride,
        padding=proj.padding,
        bias=proj.bias is not None,
    )
    new_proj.weight = nn.Parameter(new_weight)
    if proj.bias is not None:
        new_proj.bias = nn.Parameter(proj.bias.data.clone())

    patch_embed.proj = new_proj

    return SegFormerWrapper(hf_model)
