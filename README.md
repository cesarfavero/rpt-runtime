# rpt-runtime

Runtime and CLI layer for the RPT packed ternary Gemma track.

This repository is intentionally separate from the main research repository. Its job is to make the runtime, CLI, installation flow, and model bundle UX simpler for end users.

Project site:

- https://rpt.dog

Related public repositories:

- [RPT](https://github.com/cesarfavero/RPT): research, notebooks, validation, result artifacts, and reproducibility notes.
- [rpt-model-bundles](https://github.com/cesarfavero/rpt-model-bundles): bundle manifests, checksums, release metadata, and bundle distribution notes.

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

The historical 100-question suite was used during recovery training and
checkpoint selection. It is development/regression data. Regrading the saved
answers with the strict evaluator retains 100/100 for both packed paths; this
does not establish held-out generalization or acceleration. The historical
External500 report measures an earlier dense/adapter path.

The current validation work adds mandatory packed-file integrity, structural
checks, atomic installation, observed per-layer backend counters and explicit
prewarm/repeated-request measurements. Portable tests exercise these contracts.
Real GPU execution and final-model external evaluation remain separate gates.
See the [evidence review](https://github.com/cesarfavero/RPT/blob/main/docs/reports/EVIDENCE_STATUS_2026-11-02.md)
and [reproduction guide](https://github.com/cesarfavero/RPT/blob/main/REPRODUCIBILITY.md).

Triton decodes packed codes and performs floating-point dot products. The
compatibility label `is_native_ternary_runtime` refers to observed packed-path
execution. It does not mean an integer-only operator or that zero weights are
skipped. `packed_triton_execution_verified` and the per-layer execution counters
are the explicit backend evidence. Performance must be measured against a
matched dense baseline.

## Bundle Layout

Expected bundle root:

```text
MODEL_DIR/
  student_pruned/
  adapter_recovered_l78/
  packed/
    manifest.json
    mlp_ternary_2bit.bin
  production_manifest.json
  prewarm_profile.json
  PRODUCTION_USAGE.txt
```

Important:

- the current Python path still depends on runtime source files during execution;
- the target compiled product path must remove `runtime/*.py` from the public bundle contract;
- see [`docs/BUNDLE_LAYOUT.md`](docs/BUNDLE_LAYOUT.md).

Empty base/adapter directories are rejected. Installation verifies the expected
78 packed tensors and the presence of nonempty model configuration, tokenizer,
adapter configuration and weights, including all referenced shards. This is a
minimum layout check, not proof that Transformers can load every file.

The historical packed-file checksum identifies only the packed binary. A
complete bundle identity also includes dense weights, adapter, tokenizer and
runtime. Use `--require-full-checksums` when the bundle provides complete file
checksums, and `--sha256 EXPECTED_ZIP_SHA256` to verify an externally declared
archive identity. A checksum supplied by the archive itself cannot prove its
publisher's identity.

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

The current CLI behavior is the reference implementation.
The intended public compiled command contract is frozen in:

- [`docs/PUBLIC_CLI_CONTRACT.md`](docs/PUBLIC_CLI_CONTRACT.md)

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

Measure repeated requests without reloading the model:

```bash
python runtime/run_triton_inference.py --model-dir /path/to/MODEL_DIR \
  --prompt "Explain why seasons occur." --max-new-tokens 128 \
  --prewarm --repeat 5 --json
```

The report separates loading, prewarm, first request and subsequent requests.
Aggregate throughput divides total actual output tokens by total synchronized
generation time. The mean of request-level rates is reported separately.
Whole-request latency is not TTFT or TPOT, and the first request is not a
guaranteed cold compilation-cache measurement. Missing metrics stay unmeasured.

### Portable validation

```bash
python -m unittest discover -s tests -v
```

The standard-library tests validate integrity, installation, execution counters
and metric calculations. Tensor tests also need PyTorch; the CI installs a pinned
CPU build. Tests that require Triton or multiple GPUs are explicitly skipped
when those resources are unavailable. Actual GPU kernel and model evaluation
are run through the [RPT validation protocol](https://github.com/cesarfavero/RPT/blob/main/REPRODUCIBILITY.md).

SCNet BW-1/DTK compatibility is unverified. Keep its vendor-provided framework
and run the backend probe before attempting the full bundle. A CUDA-compatible
Python API alone does not identify NVIDIA hardware or prove kernel support.

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

- [`docs/brand/BRAND_DIRECTION.md`](docs/brand/BRAND_DIRECTION.md)

This is the current source of truth for the visual identity direction while the product layer is being formed.

## Relationship To The Main Repository

The main repository remains the source of truth for:

- research notebooks
- benchmark history
- methodology
- result JSONs
- bundle generation

This runtime repository is the product-facing execution layer.

## Issues and Replication Reports

Open issues in this repository for:

- CLI install or execution errors;
- runtime loader problems;
- environment support reports;
- documentation gaps specific to runtime usage;
- reproduction notes for supported Linux/NVIDIA/Colab targets.

For methodology, notebooks, benchmark interpretation, or result artifacts, use
the main [RPT](https://github.com/cesarfavero/RPT) repository instead.

## Bundle Hosting Note

GitHub Releases is not ideal for the current production bundle because the zip is too large for a clean release-asset workflow. The practical next host should be Hugging Face or another large-asset store, with `rpt-runtime` resolving the bundle URL from metadata.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

Third-party model weights, datasets, checkpoints, and trademarks remain subject
to their own upstream licenses and terms. Gemma-family assets remain subject to
the applicable Gemma Terms of Use.

