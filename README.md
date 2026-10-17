# Small-Model Alignment

Post-training pipeline for `Qwen/Qwen2.5-0.5B`: supervised fine-tuning, two from-scratch DPO runs, and a TRL LoRA persona run. The public adapter tests check the SFT and DPO math without a GPU.

## Layout

```text
submission_adapter.py          grading wrappers
alignment/                     batching, response-only loss, DPO loss, data filters
configs/                       hyperparameters for each track
scripts/prepare_data.py        30k/1k SFT and 15k/1k DPO subsets
scripts/train_sft.py           from-scratch SFT loop
scripts/train_dpo.py           from-scratch DPO loop, used for both preference datasets
scripts/train_lora_dpo.py      TRL DPOTrainer + LoRA on the Ramsay subset
scripts/generate.py            greedy generation, resumable
scripts/build_pairs.py         blinded A/B pairs
scripts/judge_pairwise_kaggle.py
scripts/plot_metrics.py
scripts/summarize_judgments.py
eval_sets/hh_rlhf_eval_300_seed42.json
eval_sets/persona_style_eval.json
data/persona_lora/ramsay_style_preferences.json
notebooks/kaggle.ipynb         full Kaggle run
report/REPORT.md               report
```

## Prompt format

Qwen SFT, both from-scratch DPO runs, and their evaluations use one prefix:

```text
Instruction:
{instruction}

Response:
```

Loss and sequence log-probabilities use response tokens only, including the EOS token when it still fits in `max_length`. The LoRA track uses SmolLM2's chat template through TRL, and generation for that model uses the same chat template.

SFT loss is the mean negative log-likelihood over unmasked response tokens. DPO uses summed response log-probabilities in the assignment formula with `beta = 0.1`.

## Local check

```bash
python -m pip install -q pytest pyyaml
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py tests/test_implementation.py
```

## Kaggle

Use a notebook with the GPU T4 x2 accelerator. From the repo root:

```bash
bash scripts/kaggle_setup.sh
python scripts/prepare_data.py --config configs/data.yaml
python scripts/make_alpaca_subset.py
python scripts/train_sft.py --config configs/sft.yaml
python scripts/train_dpo.py --config configs/dpo_hh.yaml
python scripts/train_dpo.py --config configs/dpo_ultrafeedback.yaml
python scripts/train_lora_dpo.py --config configs/lora_persona.yaml
```

`notebooks/kaggle.ipynb` runs setup, data preparation, training, generation, and judging in that order. Training writes `metrics.jsonl` plus `best` and `final` checkpoints under `outputs/`.

Generation is greedy: `do_sample=False`, `repetition_penalty=1.0`, `max_new_tokens=256`. The same settings are used for every model in a comparison.

Judging:

```bash
python scripts/judge_pairwise_kaggle.py \
  --pairs_path outputs/eval/dpo_ultra_vs_sft_pairs.jsonl \
  --output_path outputs/eval/dpo_ultra_vs_sft_judged.jsonl \
  --judge_model google/gemini-3.6-flash \
  --batch_size 5 \
  --sleep_seconds 5 \
  --max_retries 3 \
  --reuse_judgments
```

The judge prompt labels outputs only as Response A and Response B. Model identity stays in `_model_map`.
