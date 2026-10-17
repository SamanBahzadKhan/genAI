"""SFT and DPO training utilities for the small-model alignment assignment."""

from alignment.batching import build_dpo_batch, build_lm_batch
from alignment.losses import (
    compute_dpo_loss,
    compute_dpo_loss_from_logps,
    compute_response_logprobs,
    compute_sft_loss,
)

__all__ = [
    "build_dpo_batch",
    "build_lm_batch",
    "compute_dpo_loss",
    "compute_dpo_loss_from_logps",
    "compute_response_logprobs",
    "compute_sft_loss",
]
