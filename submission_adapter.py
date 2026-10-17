"""Stable grading interface for the SFT and DPO implementation.

The function signatures below are part of the assignment contract. Each one
delegates to the training code in ``alignment``. SFT loss is the mean over
unmasked response tokens, not the mean of per-example sums.
"""

from __future__ import annotations

from alignment.batching import build_dpo_batch as _build_dpo_batch
from alignment.batching import build_lm_batch
from alignment.losses import compute_dpo_loss as _compute_dpo_loss
from alignment.losses import compute_dpo_loss_from_logps as _compute_dpo_loss_from_logps
from alignment.losses import compute_response_logprobs as _compute_response_logprobs
from alignment.losses import compute_sft_loss as _compute_sft_loss


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return _compute_response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return the mean response-token negative log-likelihood."""

    return _compute_sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return _build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return _compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return _compute_dpo_loss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )
