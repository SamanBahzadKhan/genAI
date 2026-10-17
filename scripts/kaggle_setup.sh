#!/usr/bin/env bash
# Install assignment dependencies without replacing Kaggle's CUDA PyTorch.
set -euo pipefail

python -m pip uninstall -y -q torchao || true
BEFORE="$(python -c 'import torch; print(torch.__version__)')"

python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" \
  huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest matplotlib \
  "peft>=0.11.0" "trl>=0.11.0"
python -m pip install -q -e . --no-deps

AFTER="$(python -c 'import torch; print(torch.__version__)')"
if [[ "${BEFORE}" != "${AFTER}" ]]; then
  echo "PyTorch changed from ${BEFORE} to ${AFTER}. Reinstall the Kaggle CUDA build before training."
  exit 1
fi

python scripts/print_env.py
