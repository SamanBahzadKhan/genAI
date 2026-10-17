"""Prompt formatting shared by SFT, from-scratch DPO, and Qwen evaluation.

The same prefix is used at train and test time. Loss and log-probabilities
are computed only on the response that follows this prefix.
"""

from __future__ import annotations


def format_prompt(instruction: str) -> str:
    """Return the prompt prefix. The response string is appended after it."""

    return f"Instruction:\n{instruction.strip()}\n\nResponse:\n"
