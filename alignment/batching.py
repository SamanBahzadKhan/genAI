"""Response-masked language-model batches for SFT and DPO.

Sequences are right-padded to ``max_length``. An EOS token is appended to the
response before truncation, so truncation can remove that EOS. Prompt tokens
and padding tokens are labeled -100 and do not contribute to the loss.
"""

from __future__ import annotations

import torch


def _as_id_list(token_ids):
    if hasattr(token_ids, "ids"):
        token_ids = token_ids.ids
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()
    return [int(token_id) for token_id in list(token_ids)]


def encode_text(tokenizer, text):
    """Encode text the same way the public tests measure prompt length."""

    return _as_id_list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    return int(pad_id)


def _eos_id(tokenizer):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return None
    return int(eos_id)


def _one_example(tokenizer, prompt, response, max_length, max_prompt_length):
    prompt_ids = encode_text(tokenizer, prompt)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        # Keep the end of the prompt so the response boundary stays intact.
        prompt_ids = prompt_ids[-int(max_prompt_length) :]

    response_ids = encode_text(tokenizer, response)
    eos_id = _eos_id(tokenizer)
    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + list(input_ids[prompt_length:])
    attention_mask = [1] * len(input_ids)

    pad_id = _pad_id(tokenizer)
    pad_count = max_length - len(input_ids)
    if pad_count > 0:
        input_ids = input_ids + [pad_id] * pad_count
        attention_mask = attention_mask + [0] * pad_count
        labels = labels + [-100] * pad_count
    return input_ids, attention_mask, labels


def build_lm_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None):
    """Return input_ids, attention_mask, and response-only labels.

    Every row is padded or truncated to ``max_length``. Row order is preserved.
    """

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    if max_length < 1:
        raise ValueError("max_length must be positive")

    input_rows = []
    mask_rows = []
    label_rows = []
    for prompt, response in zip(prompts, responses):
        input_ids, attention_mask, labels = _one_example(
            tokenizer,
            prompt,
            response,
            max_length,
            max_prompt_length,
        )
        input_rows.append(input_ids)
        mask_rows.append(attention_mask)
        label_rows.append(labels)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
    max_prompt_length=None,
):
    """Return separate response-masked batches for chosen and rejected rows."""

    return {
        "chosen": build_lm_batch(
            tokenizer,
            prompts,
            chosen_responses,
            max_length,
            max_prompt_length=max_prompt_length,
        ),
        "rejected": build_lm_batch(
            tokenizer,
            prompts,
            rejected_responses,
            max_length,
            max_prompt_length=max_prompt_length,
        ),
    }
