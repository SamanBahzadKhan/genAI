"""Greedy generation on a fixed evaluation file.

The Qwen tracks use the same Instruction/Response prefix as training.
The LoRA persona track uses the tokenizer chat template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.formatting import format_prompt
from alignment.runtime import prepare_causal_tokenizer, set_seed


def load_eval_rows(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload["examples"]
    return payload


def render_prompt(tokenizer, instruction, prompt_format):
    if prompt_format == "plain":
        return format_prompt(instruction)
    if prompt_format == "chat":
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError(f"Unknown prompt format: {prompt_format}")


def load_model(model_path, adapter_path, dtype):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--prompt_format", choices=("plain", "chat"), default="plain")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N prompts. 0 means all.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer_source = args.adapter_path or args.model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prepare_causal_tokenizer(tokenizer)
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model = load_model(args.model_path, args.adapter_path, dtype)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    rows = load_eval_rows(args.eval_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations = []
    done = set()
    if args.resume and output_path.exists():
        generations = load_eval_rows(output_path)
        done = {row["id"] for row in generations}
        print(f"resuming with {len(done)} existing generations")

    pending = []
    for index, row in enumerate(rows):
        row_id = row.get("id", f"{args.dataset}-{index:04d}")
        if row_id not in done:
            pending.append((row_id, row))

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=args.repetition_penalty,
    )
    if args.no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    def texts_for(prompts):
        token_rows = []
        for prompt in prompts:
            token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(token_ids) > args.max_prompt_length:
                token_ids = token_ids[-args.max_prompt_length :]
            token_rows.append(token_ids)
        width = max(len(token_ids) for token_ids in token_rows)
        pad_id = tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        for token_ids in token_rows:
            pad = width - len(token_ids)
            input_ids.append([pad_id] * pad + token_ids)
            attention_mask.append([0] * pad + [1] * len(token_ids))
        encoded = {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attention_mask, device=device),
        }
        with torch.no_grad():
            output_ids = model.generate(**encoded, **generate_kwargs)
        decoded = []
        for row_ids in output_ids:
            decoded.append(tokenizer.decode(row_ids[width:], skip_special_tokens=True).strip())
        return decoded

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [
            render_prompt(tokenizer, row["instruction"], args.prompt_format) for _, row in batch
        ]
        texts = texts_for(prompts)
        for (row_id, row), text in zip(batch, texts):
            record = {
                "id": row_id,
                "instruction": row["instruction"],
                "output": text,
                "generator": args.generator,
                "dataset": args.dataset,
            }
            if "category" in row:
                record["category"] = row["category"]
            generations.append(record)
        temporary = output_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(generations, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(output_path)
        print(f"generated {len(generations)}/{len(rows)}", flush=True)

    print(f"wrote {len(generations)} generations to {output_path}", flush=True)


if __name__ == "__main__":
    main()
