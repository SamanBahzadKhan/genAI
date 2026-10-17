"""Edge cases beyond the public adapter smoke tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment.data import first_ultrachat_turn, parse_hh_dialogue, single_turn_messages
from alignment.losses import compute_dpo_loss_from_logps
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_batches_are_padded_to_max_length_and_keep_row_order():
    tokenizer = ToyTokenizer()
    prompts = ["A: ", "BB: "]
    responses = ["a", "longer-response"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=24)

    assert batch["input_ids"].shape == (2, 24)
    first_prompt = len(tokenizer.encode(prompts[0], add_special_tokens=False))
    second_prompt = len(tokenizer.encode(prompts[1], add_special_tokens=False))
    assert batch["labels"][0, first_prompt] != batch["labels"][1, second_prompt]
    assert batch["attention_mask"][0, :first_prompt].eq(1).all()
    assert batch["labels"][0, :first_prompt].eq(-100).all()


def test_truncation_can_remove_the_eos_token():
    tokenizer = ToyTokenizer()
    prompt = "abc"
    response = "defghijkl"
    batch = adapter.build_sft_batch(tokenizer, [prompt], [response], max_length=8)

    assert batch["input_ids"].shape == (1, 8)
    assert tokenizer.eos_token_id not in batch["input_ids"][0].tolist()
    assert batch["labels"][0, :3].eq(-100).all()
    assert batch["labels"][0, 3:].ne(-100).all()


def test_full_sequence_keeps_eos_when_it_fits():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["ab"], ["cd"], max_length=5)
    assert batch["labels"][0, 4] == tokenizer.eos_token_id
    assert batch["attention_mask"][0].eq(1).all()


def test_prompt_longer_than_context_has_zero_response_logprob():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.4)
    logps = adapter.compute_response_logprobs(model, tokenizer, ["abcdefghij"], ["z"], max_length=4)
    assert logps.shape == (1,)
    torch.testing.assert_close(logps, torch.zeros_like(logps))


def test_dpo_loss_matches_the_written_formula():
    policy_chosen = torch.tensor([0.0, -1.0])
    policy_rejected = torch.tensor([-1.0, -1.0])
    reference_chosen = torch.tensor([0.0, -1.0])
    reference_rejected = torch.tensor([0.0, -1.0])
    loss = compute_dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    expected = (-F.logsigmoid(torch.tensor(0.1)) + torch.log(torch.tensor(2.0))) / 2
    torch.testing.assert_close(loss, expected)


def test_sft_optimizer_step_decreases_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.2)
    prompts = ["Do: ", "Say: "]
    responses = ["a", "longer"]
    before = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=16).detach()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=16)
    loss.backward()
    optimizer.step()
    after = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=16).detach()
    assert after < before


def test_dpo_chosen_and_rejected_rows_are_not_swapped():
    tokenizer = ToyTokenizer()
    batch = adapter.build_dpo_batch(
        tokenizer,
        ["Task: "],
        ["good"],
        ["bad"],
        max_length=12,
    )
    chosen_ids = batch["chosen"]["input_ids"][0].tolist()
    rejected_ids = batch["rejected"]["input_ids"][0].tolist()
    assert chosen_ids != rejected_ids
    assert tokenizer.encode("good", add_special_tokens=False)[0] in chosen_ids
    assert tokenizer.encode("bad", add_special_tokens=False)[0] in rejected_ids


def test_preference_filters_keep_only_single_turn_pairs():
    assert parse_hh_dialogue("\n\nHuman: hello\n\nAssistant: hi there") == ("hello", "hi there")
    assert parse_hh_dialogue("\n\nHuman: one\n\nAssistant: a\n\nHuman: two\n\nAssistant: b") is None
    assert first_ultrachat_turn(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "second"},
        ]
    ) == ("question", "first")
    messages = [
        {"role": "user", "content": "only ask"},
        {"role": "assistant", "content": "only answer"},
    ]
    assert single_turn_messages(messages) == ("only ask", "only answer")
    assert single_turn_messages(messages + [{"role": "user", "content": "again"}]) is None
