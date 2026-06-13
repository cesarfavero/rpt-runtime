# Compiled Runtime Roadmap

This roadmap is for the current `rpt-runtime` repository as it exists today.

It does not redesign the product from scratch.
It starts from the current CLI/runtime shape and defines the minimum path to a closed, compiled distribution.

Implementation decision:

- transition strategy and long-term target are documented in `docs/COMPILED_IMPLEMENTATION_STRATEGY.md`

## Current State

The repository already has:

- a user-facing command name: `rpt`
- install/list/use/run/doctor CLI behavior
- bundle registration under `~/.rpt`
- runtime loaders
- Triton/native/hybrid execution entrypoints

The repository is still source-visible because:

- `bin/rpt` is a shell shim that executes `python runtime/cli.py`
- the CLI delegates execution to Python entrypoints
- bundle validation currently requires `runtime/run_triton_inference.py` inside the model bundle
- the install flow assumes a Python runtime plus source-visible runtime files

## Goal

Keep the current product behavior, but remove source-visible public distribution.

Target product behavior:

- `rpt install ...`
- `rpt ls`
- `rpt use ...`
- `rpt run ...`
- `rpt doctor`

Target product property:

- user receives a compiled CLI/runtime surface, not a Python source bundle

## Required Architecture Shift

### Today

```text
shell wrapper -> python cli.py -> python runtime scripts -> bundle contains runtime/*.py
```

### Target

```text
compiled rpt binary -> runtime execution layer -> model bundle without public runtime source
```

## Workstreams

### 1. Freeze the CLI contract

Keep these commands stable:

- `rpt install`
- `rpt run`
- `rpt native`
- `rpt hybrid`
- `rpt ls`
- `rpt use`
- `rpt doctor`

Before any port:

- document exact flags and outputs
- define which commands remain public in v1
- decide whether `native` and `hybrid` stay user-facing or become debug/internal

Recommendation:

- public v1 keeps `install`, `run`, `ls`, `use`, `doctor`
- `native` and `hybrid` become internal/debug unless product really needs them

### 2. Remove runtime code from the model bundle contract

This is the most important change.

Current bundle validation requires:

- `runtime/run_triton_inference.py`

That couples the bundle to source-visible runtime delivery.

Target bundle contract should contain only model artifacts and metadata:

- `student_pruned/`
- `adapter_recovered_l78/`
- `packed/manifest.json`
- `packed/mlp_ternary_2bit.bin`
- `production_manifest.json`
- `prewarm_profile.json`
- notices / terms

The compiled runtime should provide the execution layer itself.

### 3. Port the CLI from Python wrapper to compiled implementation

Minimum responsibilities to port:

- app home/config management
- model registration
- bundle install from zip/dir/url
- bundle validation
- default-model selection
- subprocess/runtime invocation replacement
- doctor command

Recommendation:

- use Rust for the compiled CLI/runtime surface
- keep Python runtime as internal reference during transition

### 4. Define how the compiled runtime calls inference

This is the biggest technical decision.

Options:

1. embed Python privately
   Fast transition, weak closure story

2. compiled launcher around private packaged Python
   Better than current state, still not ideal

3. port runtime execution path to native compiled implementation
   Best closure story, highest cost

Recommendation:

- use the chosen transition strategy from `docs/COMPILED_IMPLEMENTATION_STRATEGY.md`
- target the long-term native path after the distribution surface is closed

### 5. Bundle resolution and remote install

The current CLI already knows how to install from:

- local zip
- local dir
- direct URL
- GitHub release URL pattern

For productization:

- remove assumptions tied to public GitHub release assets
- resolve bundle metadata from a private release manifest
- support authenticated download without exposing source runtime

### 6. Doctor and environment gating

`rpt doctor` should become a real gatekeeper for supported environments:

- detect CUDA availability
- detect compatible GPU
- detect driver/runtime mismatches
- detect missing dependencies
- reject unsupported targets clearly

This is especially important if the runtime remains Linux/NVIDIA-first.

## Suggested Sequence

### Phase A

- freeze CLI contract
- remove runtime files from the bundle contract on paper
- define public vs internal commands

### Phase B

- implement compiled CLI skeleton
- port config/install/list/use/doctor first
- keep run path behind current internal runtime while transitioning

### Phase C

- move execution responsibility out of bundle-delivered Python files
- make bundle validation independent from `runtime/*.py`

### Phase D

- switch public distribution to compiled `rpt`
- keep Python runtime private/internal only
- re-enable public install only after this point

## Immediate Next Actions

- [ ] replace bundle validation rules so bundles no longer require `runtime/run_triton_inference.py`
- [ ] define the v1 public command set
- [ ] choose compiled implementation language and runtime strategy
- [ ] port config/install/list/use/doctor before porting `run`
- [ ] define private bundle manifest format for authenticated download

## Non-Goals For This Roadmap

- retraining the model
- changing benchmark methodology
- Apple Silicon support
- Ollama integration

Those can come later. The immediate problem is closing the runtime distribution surface.
