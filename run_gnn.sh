#!/bin/bash

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset}"

echo "=== NVIDIA status ==="
nvidia-smi || true

echo "=== PyTorch status ==="
/home/hamishra/.local/bin/uv run python - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("test tensor device:", x.device)
PY

echo "=== Run application ==="
/home/hamishra/.local/bin/uv run gnn cora