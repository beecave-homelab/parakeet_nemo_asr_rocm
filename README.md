# Parakeet NeMo ASR ROCm

Automatic Speech Recognition inference service for the NVIDIA Parakeet-TDT 0.6B v2 model running on AMD GPUs via ROCm.

## Quick start (Docker)

```bash
# Build image (first time ~10-15 min)
make build      # wraps: docker compose build

# Run container in foreground
make run        # wraps: docker compose up
```

Open another terminal for an interactive shell inside the running container:

```bash
make shell      # or ./scripts/dev_shell.sh
```

### Smoke-test inside container

```bash
python - <<'PY'
import torch, nemo.collections.asr as nemo_asr, os
print("HIP:", torch.version.hip, "CUDA avail:", torch.cuda.is_available())
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").eval().to("cuda")
print(model.transcribe(["/data/samples/sample.wav"], batch_size=1)[0])
PY
```

## Local dev (without Docker)

Prereqs: Python 3.10, ROCm 6.4.1, PDM ≥2.15, ROCm PyTorch wheels in your `--find-links`.

```bash
# Create lockfile + install deps (incl. rocm extras)
make lock      # pdm lock && pdm export …
pdm install -G rocm

# Run unit tests
make test      # pytest -q

# Transcribe a wav locally
python -m parakeet_nemo_asr_rocm.app data/samples/sample.wav
```

## Troubleshooting

• Verify `/opt/rocm` bind-mount and device permissions (`/dev/kfd`, `/dev/dri`).  
• Export `HSA_OVERRIDE_GFX_VERSION=10.3.0` for RX 6600 if `rocminfo` shows unknown GPU.  
• OOM? Call `model = model.half()`.
