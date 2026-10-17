"""Print the accelerator and package versions required by Task 1."""

from __future__ import annotations

import subprocess
import sys


def version(module_name):
    module = __import__(module_name)
    return getattr(module, "__version__", "unknown")


def main():
    import torch

    print("python", sys.version.split()[0])
    print("torch", version("torch"))
    print("cuda_available", torch.cuda.is_available())
    print("gpu_count", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(f"gpu_{index}", torch.cuda.get_device_name(index))
    for module_name in ("transformers", "datasets", "peft", "trl"):
        try:
            print(module_name, version(module_name))
        except Exception as exc:
            print(module_name, f"unavailable: {exc}")
    subprocess.run(["nvidia-smi"], check=False)


if __name__ == "__main__":
    main()
