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
./bin/rpt install --zip /path/to/production_triton_packed_recovered_l78.zip
./bin/rpt run --prompt "Qual a capital da França?"
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

### Install a bundle

```bash
./bin/rpt install --zip /path/to/production_triton_packed_recovered_l78.zip
```

### Run Triton path

```bash
./bin/rpt run --prompt "Quanto é 12 x 8?"
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
2. bundle auto-download by version or remote URL
3. release packaging
4. Homebrew formula
5. curl installer
6. Mac/MPS and Ollama-like target exploration

## Brand Direction

Initial runtime-facing brand direction lives in:

- [`docs/brand/BRAND_DIRECTION.md`](/Users/cesarfavero/Documents/MyPru/rpt-runtime/docs/brand/BRAND_DIRECTION.md)

This is the current source of truth for the visual identity direction while the product layer is being formed.

## Relationship To The Main Repository

The main repository remains the source of truth for:

- research notebooks
- benchmark history
- methodology
- result JSONs
- bundle generation

This runtime repository is the product-facing execution layer.

## Bundle Hosting Note

GitHub Releases is not ideal for the current production bundle because the zip is too large for a clean release-asset workflow. The practical next host should be Hugging Face or another large-asset store, with `rpt-runtime` resolving the bundle URL from metadata.
