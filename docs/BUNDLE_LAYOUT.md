# Bundle Layout

This document defines two things:

1. the current source-visible bundle layout still used by the Python runtime;
2. the target public bundle contract for the compiled `rpt` distribution.

## Current Layout

The current Python runtime expects a model bundle directory with this structure:

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

Problem:

- this layout keeps the public bundle coupled to `runtime/*.py`
- that is acceptable for internal engineering, but not for the closed compiled product surface

## Target Public Bundle Contract

The public model bundle should contain only model artifacts and metadata:

```text
MODEL_DIR/
  student_pruned/
  adapter_recovered_l78/
  packed/
    manifest.json
    mlp_ternary_2bit.bin
  production_manifest.json
  prewarm_profile.json
  NOTICE
  TERMS
```

The execution layer must come from the compiled `rpt` runtime, not from bundle-delivered Python files.

## Required Public Bundle Files

- `student_pruned/`
- `adapter_recovered_l78/`
- `packed/manifest.json`
- `packed/mlp_ternary_2bit.bin`
- `production_manifest.json`
- `prewarm_profile.json`
- legal/notice files required for distribution

## Files That Must Leave The Public Contract

- `runtime/run_triton_inference.py`
- `runtime/run_native_inference.py`
- `runtime/run_inference.py`
- `runtime/loader.py`
- `runtime/native_loader.py`
- `runtime/triton_loader.py`
- runtime-local dependency files that exist only to make the Python path executable

## Validation Rule Change Required

The install validator should enforce only model artifacts and metadata.

Current required validation surface:

- `student_pruned/`
- `adapter_recovered_l78/`
- `packed/manifest.json`
- `packed/mlp_ternary_2bit.bin`
- `production_manifest.json`
- `prewarm_profile.json`

The compiled runtime should validate only the model artifact and metadata surface.

## Source Of Bundles

The production bundle continues to be generated from the main RPT repository.

This runtime repository does not need to commit the production bundle itself.
