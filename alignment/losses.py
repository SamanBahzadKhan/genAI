"""Response-only SFT loss and the from-scratch DPO objective.

SFT reduction: mean negative log-likelihood over unmasked response tokens
(prompt and padding tokens are ignored). This is one of the two reductions
accepted by the assignment.

DPO uses summed response-token log-probabilities, then

    L = -log sigmoid(beta * ((log pi_w - log pi_l) - (log ref_w - log ref_l)))

averaged over the batch. Reference log-probabilities are computed under
``torch.no_grad``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.batching import build_dpo_batch, build_lm_batch


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _logits_from_output(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    return outputs[0]


def _response_token_logprobs(logits, labels):
    """Sum log p(response tokens) for each row, using the standard LM shift."""

    labels = labels.to(device=logits.device)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    if shift_logits.shape[1] == 0:
        empty = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
        mask = torch.zeros(logits.shape[0], 0, device=logits.device, dtype=torch.bool)
        return empty, mask

    mask = shift_labels.ne(-100)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp(min=0).unsqueeze(-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels).squeeze(-1)
    token_logps = token_logps.masked_fill(~mask, 0.0)
    return token_logps.sum(dim=-1), mask


def response_logprobs_from_batch(model, batch):
    """Summed response log-probability for one already-tokenized LM batch."""

    device = _model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = _logits_from_output(outputs)
    summed, mask = _response_token_logprobs(logits, labels)
    return summed, mask


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length, max_prompt_length=None):
    """Return one summed response-token log-probability per input row."""

    batch = build_lm_batch(
        tokenizer,
        prompts,
        responses,
        max_length,
        max_prompt_length=max_prompt_length,
    )
    summed, _mask = response_logprobs_from_batch(model, batch)
    return summed


def compute_sft_loss(model, tokenizer, prompts, responses, max_length, max_prompt_length=None):
    """Mean response-token NLL. Prompt and padding positions are excluded."""

    batch = build_lm_batch(
        tokenizer,
        prompts,
        responses,
        max_length,
        max_prompt_length=max_prompt_length,
    )
    summed, mask = response_logprobs_from_batch(model, batch)
    token_count = mask.sum().clamp_min(1)
    return -summed.sum() / token_count


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Scalar batch-mean DPO loss from per-example summed log-probabilities."""

    device = policy_chosen_logps.device
    dtype = torch.float32
    policy_chosen = policy_chosen_logps.to(device=device, dtype=dtype)
    policy_rejected = policy_rejected_logps.to(device=device, dtype=dtype)
    reference_chosen = reference_chosen_logps.to(device=device, dtype=dtype)
    reference_rejected = reference_rejected_logps.to(device=device, dtype=dtype)
    beta_value = torch.as_tensor(beta, device=device, dtype=dtype)
    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen - reference_rejected
    logits = beta_value * (policy_gap - reference_gap)
    return -F.logsigmoid(logits).mean()


def preference_rewards(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Implicit DPO rewards r(y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))."""

    device = policy_chosen_logps.device
    dtype = torch.float32
    beta_value = torch.as_tensor(beta, device=device, dtype=dtype)
    chosen = beta_value * (
        policy_chosen_logps.to(device=device, dtype=dtype) - reference_chosen_logps.to(device=device, dtype=dtype)
    )
    rejected = beta_value * (
        policy_rejected_logps.to(device=device, dtype=dtype) - reference_rejected_logps.to(device=device, dtype=dtype)
    )
    return chosen, rejected


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
    max_prompt_length=None,
):
    """Scalar mean DPO loss. The reference forward does not build a graph."""

    batch = build_dpo_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length,
        max_prompt_length=max_prompt_length,
    )
    was_training = reference_model.training
    reference_model.eval()
    try:
        with torch.no_grad():
            reference_chosen, _ = response_logprobs_from_batch(reference_model, batch["chosen"])
            reference_rejected, _ = response_logprobs_from_batch(reference_model, batch["rejected"])
    finally:
        reference_model.train(was_training)

    policy_chosen, _ = response_logprobs_from_batch(policy_model, batch["chosen"])
    policy_rejected, _ = response_logprobs_from_batch(policy_model, batch["rejected"])
    return compute_dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        reference_chosen.detach(),
        reference_rejected.detach(),
        beta,
    )
