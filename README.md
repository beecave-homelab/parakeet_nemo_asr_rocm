# Parakeet-ROCm

[![Version](https://img.shields.io/badge/Version-v0.2.1-informational)](./VERSIONS.md)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-6.4.1-red)](https://rocm.docs.amd.com/)

Containerised, GPU-accelerated Automatic Speech Recognition (ASR) inference service for the NVIDIA **Parakeet-TDT 0.6B v2** model, running on **AMD ROCm** GPUs.

## 🚀 What's Included

- **CLI Tool**: Typer-based command-line interface with rich progress tracking
- **Docker Support**: Pre-configured container with ROCm, NeMo 2.2, and all dependencies
- **Batch Processing**: Efficient batch transcription with configurable chunking
- **Multiple Output Formats**: TXT, SRT, VTT, and JSON transcription outputs
- **Timestamp Alignment**: Word-level timestamp generation and intelligent subtitle segmentation
- **GPU Acceleration**: Optimized for AMD GPUs via ROCm platform

## Key Features

- **🎯 High Accuracy**: Leverages NVIDIA's state-of-the-art Parakeet-TDT 0.6B v2 model
- **⚡ GPU Accelerated**: Runs efficiently on AMD GPUs through ROCm platform
- **📦 Containerised**: Dockerized deployment with ROCm support
- **📋 Multiple Formats**: Export transcriptions in TXT, SRT, VTT, or JSON formats
- **⏱️ Timestamp Support**: Word-level timestamps with intelligent segmentation
- **🔄 Batch Processing**: Process multiple files efficiently with configurable batch sizes
- **🔧 Configurable**: Environment-based configuration for all key parameters

## 🎯 Why This Project?

This project bridges the gap between NVIDIA's cutting-edge ASR models and AMD GPU hardware through the ROCm platform. While NVIDIA's NeMo framework is primarily optimized for CUDA, this project enables running the powerful Parakeet-TDT model on AMD hardware with minimal performance impact.

## Badges

[![Version](https://img.shields.io/badge/Version-v0.2.1-informational)](./VERSIONS.md)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-6.4.1-red)](https://rocm.docs.amd.com/)
[![CLI](https://img.shields.io/badge/CLI-Typer-yellow)](#cli)

---

## 📚 Table of Contents

- [What's Included](#-whats-included)
- [Key Features](#key-features)
- [Why This Project?](#-why-this-project)
- [Installation](#installation)
  - [Recommended: Docker Compose](#recommended-docker-compose)
  - [Alternative: Local Development](#alternative-local-development)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI](#cli)
  - [API Parameters](#api-parameters)
  - [Output Files](#output-files)
- [Development](#development)
- [License](#license)
- [Contributing](#contributing)

## Installation

### Recommended: Docker Compose

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/parakeet_nemo_asr_rocm.git
    cd parakeet_nemo_asr_rocm
    ```

2. Build the Docker image (first time ~10-15 min):

    ```bash
    make build
    # or: docker compose build
    ```

3. Run the container:

    ```bash
    make run
    # or: docker compose up
    ```

4. Open another terminal for an interactive shell inside the running container:

    ```bash
    make shell
    # or: ./scripts/dev_shell.sh
    ```

### Alternative: Local Development

Prerequisites: Python 3.10, ROCm 6.4.1, PDM ≥2.15, ROCm PyTorch wheels in your `--find-links`.

1. Create lockfile and install dependencies (including ROCm extras):

    ```bash
    make lock      # pdm lock && pdm export …
    pdm install -G rocm
    ```

2. Run unit tests:

    ```bash
    make test      # pytest -q
    ```

3. Transcribe a wav file locally:

    ```bash
    python -m parakeet_nemo_asr_rocm.cli data/samples/sample.wav
    
    # Or use the installed CLI script
    parakeet-rocm data/samples/sample.wav
    ```

## Configuration

The project uses environment variables for configuration. See `.env.example` for all options:

```bash
cp .env.example .env
```

Key configuration variables:

| Variable | Description | Default |
|---------|-------------|---------|
| `CHUNK_LEN_SEC` | Audio chunk size in seconds for processing long files | 300 |
| `BATCH_SIZE` | Batch size for model inference | 16 |
| `MAX_CPS` | Max characters per second for subtitle readability | 17 |
| `MIN_CPS` | Min characters per second for subtitle readability | 12 |
| `MAX_LINE_CHARS` | Maximum characters per subtitle line | 42 |

For ROCm-specific configuration, the following environment variables are set by default:

- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (mitigates ROCm memory fragmentation)
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` (required for some AMD GPUs)

## Usage

### CLI

The primary interface is a Typer-based CLI with rich help messages:

```bash
# Basic transcription
parakeet-rocm data/samples/sample.wav

# Transcribe multiple files
parakeet-rocm file1.wav file2.wav

# Specify output directory and format
parakeet-rocm --output-dir ./transcripts --output-format srt file.wav

# Adjust batch size for performance
parakeet-rocm --batch-size 8 file.wav

# Enable word-level timestamps
parakeet-rocm --word-timestamps file.wav

# Get help
parakeet-rocm --help
parakeet-rocm transcribe --help
```

### API Parameters

| Parameter | Type | Description | Default |
|----------|------|-------------|---------|
| `audio_files` | List[pathlib.Path] | Path to one or more audio files to transcribe | required |
| `--model` | str | Hugging Face Hub or local path to the NeMo ASR model | nvidia/parakeet-tdt-0.6b-v2 |
| `--output-dir` | pathlib.Path | Directory to save the transcription outputs | ./output |
| `--output-format` | str | Format for the output file(s) (txt, srt, vtt, json) | txt |
| `--batch-size` | int | Batch size for transcription inference | 16 (from env) |
| `--chunk-len-sec` | int | Segment length in seconds for chunked transcription | 300 (from env) |
| `--word-timestamps` | bool | Enable word-level timestamp generation | False |
| `--overwrite` | bool | Overwrite existing output files | False |
| `--verbose` | bool | Enable verbose output | False |
| `--quiet` | bool | Suppress console output except progress bar | False |
| `--no-progress` | bool | Disable the Rich progress bar while still showing created file paths | False |
| `--fp16` | bool | Enable half-precision (FP16) inference | False |

### Output Files

Transcriptions are saved in the specified output directory with filenames based on the input files. Supported formats include:

- **TXT**: Plain text transcription
- **SRT**: SubRip subtitle format with timestamps
- **VTT**: Web Video Text Tracks format
- **JSON**: Structured output with detailed timing information

## Development

See [`project-overview.md`](./project-overview.md) for complete architecture and developer documentation.

For local development:

1. Install development dependencies:

    ```bash
    pdm install -G rocm -G dev
    ```

2. Run tests:

    ```bash
    make test
    # or: pytest -q
    ```

3. Code formatting and linting:

    ```bash
    # Format code
    pdm run format
    
    # Check code style
    pdm run lint
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your proposal.

See [`project-overview.md`](./project-overview.md) for complete architecture and developer documentation.
