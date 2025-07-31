# 🛠️ Port MLX Features to ROCm Parakeet Wrapper

This plan outlines the steps to implement feature parity between the Apple MLX implementation and the ROCm/Nemo wrapper. The goal is to add:

- Robust audio loading
- Smarter chunking & merging
- Streaming inference
- Enhanced CLI with multiple output formats
- Environment-driven configuration

> All while maintaining **NeMo** as the backend.

---

## ✅ Tasks

### 🔍 Analysis Phase

#### [ ] Research FFmpeg audio decoding and integration

- **Path:** `parakeet_nemo_asr_rocm/utils/audio_io.py`  
- **Reference:** [audio.py FFmpeg logic (lines 52–76)](https://github.com/senstella/parakeet-mlx/blob/b2dc18a3a6e6bf7cd9dda5d2e197d15090e55769/parakeet_mlx/audio.py#L52-L76)

**Action**:  
Investigate how [`audio.py`](https://github.com/senstella/parakeet-mlx/blob/main/parakeet_mlx/audio.py) uses `subprocess.run` to decode arbitrary audio into 16-bit PCM. Compare with the current `load_audio` implementation.

**Analysis Results**:

- Determine whether FFmpeg provides more robust format support than `librosa`/`soundfile`.
- Identify licensing/deployment considerations for bundling FFmpeg.

**Acceptance Criteria**:

- Decide whether FFmpeg should be the default decoder.
- Document decision and required changes.

---

#### [ ] Evaluate NeMo’s long-audio and streaming capabilities

- **Path:** `parakeet_nemo_asr_rocm/transcribe.py`  
- **Docs:** NeMo ASR documentation

**Action**:  
Check if `nemo_asr.models.ASRModel` exposes streaming inference or long-audio support. Compare with MLX's `transcribe_stream()` API.

**Analysis Results**:

- Does NeMo support encoder state caching?
- If not, calculate effective receptive field using `_calc_time_stride`.

**Acceptance Criteria**:

- Recommend manual chunking or rely on native NeMo support.

---

#### [ ] Study token-based merging algorithms

- **Path:** [`alignment.py`](https://raw.githubusercontent.com/senstella/parakeet-mlx/refs/heads/master/parakeet_mlx/alignment.py)  
- **Refs:**  
  - [merge_longest_contiguous](https://raw.githubusercontent.com/senstella/parakeet-mlx/refs/heads/master/parakeet_mlx/alignment.py#L81-L104)  
  - [merge_longest_common_subsequence](https://raw.githubusercontent.com/senstella/parakeet-mlx/refs/heads/master/parakeet_mlx/alignment.py#L186-L252)

**Action**:  
Understand how token alignment is done across overlapping segments using token IDs and timestamps.

**Analysis Results**:

- Evaluate applicability of MLX logic to NeMo hypotheses and `Word` model.
- Compare MLX token IDs (SentencePiece) vs NeMo (WordPiece or others).

**Acceptance Criteria**:

- Decide on merge strategy (contiguous vs LCS).
- Design interface for `AlignedResult.word_segments`.

---

#### [ ] Assess CLI feature gaps

- **Path:**  
  - ROCm CLI: `parakeet_nemo_asr_rocm/cli.py`  
  - MLX CLI: [`cli.py`](https://raw.githubusercontent.com/senstella/parakeet-mlx/refs/heads/master/parakeet_mlx/cli.py)

**Action**:  
Compare available CLI options between ROCm and MLX:

**Missing Features**:

- Output formats
- Templates
- Word highlighting
- Precision flags
- Chunk durations & overlaps
- Progress bars

**Acceptance Criteria**:

- Checklist of CLI gaps with prioritization.

---

### 🔧 Implementation Phase

#### [ ] Integrate FFmpeg Audio Decoding  

- Add FFmpeg loader with fallback
- Update CLI and `transcribe_paths()`

#### [ ] Implement Chunking & Merging  

- Implement in:  
  - `chunking/chunker.py`  
  - `merge.py` (new)  
- Handle word timestamp offsets

#### [ ] Add Streaming Inference Mode  

- Use NeMo's streaming APIs (if available)  
- Otherwise, implement manual encoder state caching  
- Inspired by MLX's [`parakeet.py`](https://raw.githubusercontent.com/senstella/parakeet-mlx/refs/heads/master/parakeet_mlx/parakeet.py#L195-L234)

#### [ ] Expand CLI Functionality  

Add support for:

- `--model`  
- `--output-dir`, `--output-template`  
- `--output-format`: `txt`, `srt`, `vtt`, `json`, `all`  
- `--chunk-duration`, `--overlap-duration`  
- `--batch-size`, `--fp32`, `--fp16`  
- `--highlight-words`, `--verbose`

Use formatters from: `formatting/_*.py`

---

#### [ ] Finalize Timestamp Adaptation and Formatters

- **Path:**  
  - `timestamps/adapt.py`  
  - `formatting/_srt.py`, `_vtt.py`, `_json.py`, `_txt.py`

**Action**:  

- Finalize `adapt_nemo_hypotheses()`  
- Implement VTT formatter  
- Expose `.env.example` vars from [`constant.py`](https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/main/parakeet_nemo_asr_rocm/utils/constant.py)

---

#### [ ] Add Precision Casting Support

**Action**:  

- Call `model.float()` or `model.half()` after model load  
- Ensure input tensor casting

---

### 🧪 Testing Phase

#### [ ] Unit Tests for Merging & Segmentation  

- **Path:** `tests/test_merge.py`, `test_segmentation.py`  
- Create synthetic `AlignedResult` objects

#### [ ] Formatter Tests  

- **Path:** `tests/test_formatting.py`  
- Test all formats + `highlight_words=True`

#### [ ] CLI Integration Tests  

- **Path:** `tests/test_cli.py`  
- Validate output files, formats, flags (`--fp16`, `--output-format all`)

---

### 📚 Documentation Phase

#### [ ] Update `project-overview.md` and `README.md`

**Action**:  

- Document:
  - FFmpeg dependency
  - Streaming mode
  - CLI usage
  - Merging strategy
  - Environment vars
- Include examples and known limitations

---

## 📁 Related Files

- `audio_io.py`: [Link](https://raw.githubusercontent.com/senstella/parakeet-mlx/refs/heads/master/parakeet_mlx/audio.py)
- `chunker.py`, `merge.py`
- `cli_ty.py`, `parakeet.py`, `adapt.py`
- `constant.py`, `.env.example`
- `test_merge.py`, `test_formatting.py`, `test_cli.py`
- `project-overview.md`, `README.md`

---

## 🚀 Future Enhancements

- [ ] Investigate NeMo local attention or inference optimizations  
- [ ] Add speaker diarization support  
- [ ] Add YAML/JSON-based config support for reproducible pipelines  

---

> *Document based on internal audit and analysis of the following repositories:*  
>
> - [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
> - [parakeet_nemo_asr_rocm](https://github.com/beecave-homelab/parakeet_nemo_asr_rocm)
