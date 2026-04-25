# rpt-runtime

Runtime and CLI layer for the RPT packed ternary Gemma track.

This repository is intentionally separate from the main research repository. Its job is to make the runtime, CLI, installation flow, and model bundle UX simpler for end users.

## Scope

This repository contains:

- the runtime loaders
- the Triton/native/materialized inference entrypoints
- a small CLI wrapper
- bundle format documentation
- installation and usage instructions

This repository does not contain the full research history or notebooks. Those remain in the main repository:

- [RPT](https://github.com/cesarfavero/RPT)

## Current Supported Target

Current supported environment:

- Linux
- NVIDIA GPU
- CUDA
- Triton
- Colab-compatible stacks

Not a current target:

- Apple Silicon production runtime
- Ollama runtime
- CPU-only consumer packaging

## Runtime Status

Current validated story from the main project:

- original HF Gemma 2B local suite: `0.90`
- improved dense checkpoint `student_pruned`: `0.92`
- packed recovered runtime: `1.00`
- Triton integrated runtime: `1.00`

## Bundle Layout

Expected bundle root:

```text
MODEL_DIR/
  student_pruned/
  adapter_recovered_l78/
  packed/
    manifest.json
    mlp_ternary_2bit.bin
  runtime/
  production_manifest.json
  prewarm_profile.json
  PRODUCTION_USAGE.txt
```

## CLI

This repository ships a small wrapper command:

```bash
./bin/rpt run --model-dir /path/to/MODEL_DIR --prompt "Qual a capital da França?"
```

Equivalent direct Triton command:

```bash
python runtime/run_triton_inference.py --model-dir /path/to/MODEL_DIR --prompt "Qual a capital da França?"
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
```

If needed:

```bash
export HF_TOKEN=your_token_here
```

## Commands

### Run Triton path

```bash
./bin/rpt run --model-dir /path/to/MODEL_DIR --prompt "Quanto é 12 x 8?"
```

### Run native structural path

```bash
./bin/rpt native --model-dir /path/to/MODEL_DIR --prompt "Quanto é 12 x 8?"
```

### Run materialized fallback path

```bash
./bin/rpt hybrid --model-dir /path/to/MODEL_DIR --prompt "Quanto é 12 x 8?"
```

## Roadmap

Near-term roadmap:

1. stable install command
2. bundle auto-download by version
3. release packaging
4. Homebrew formula
5. curl installer
6. Mac/MPS and Ollama-like target exploration

## Relationship To The Main Repository

The main repository remains the source of truth for:

- research notebooks
- benchmark history
- methodology
- result JSONs
- bundle generation

This runtime repository is the product-facing execution layer.
