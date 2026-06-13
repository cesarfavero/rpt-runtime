# Public CLI Contract

This document freezes the intended public CLI surface for the first compiled `rpt` release.

It describes the product contract, not the current Python implementation details.

## Principles

- keep the command surface small
- preserve the current user mental model
- avoid exposing internal runtime modes by default
- keep machine-readable outputs where they already make sense

## Public v1 Commands

### `rpt install`

Purpose:

- register a model bundle locally

Supported inputs:

- local zip
- local extracted bundle directory
- authenticated remote bundle URL
- release manifest resolution in a later step

Stable behavior:

- validates bundle layout
- installs into the local RPT model store
- registers the model in local config
- prints machine-readable install result

Required flags and inputs for v1:

- `--zip`
- `--dir`
- `--url`
- `--name`

### `rpt run`

Purpose:

- run the selected or specified model through the default production runtime path

Stable behavior:

- resolves the default installed model if `--model-dir` is omitted
- accepts a prompt
- supports `--max-new-tokens`
- supports machine-readable JSON output

Required flags and inputs for v1:

- `--prompt`
- `--model-dir` optional
- `--max-new-tokens`
- `--json`

### `rpt ls`

Purpose:

- list installed models

Stable behavior:

- returns registered models
- indicates default selection
- indicates whether the path still exists

### `rpt use`

Purpose:

- set the default installed model

Stable behavior:

- updates local config only

### `rpt doctor`

Purpose:

- inspect whether the local machine can run the selected runtime target

Stable behavior:

- reports runtime environment facts
- reports support or obvious incompatibility

## Internal / Non-Public By Default

These exist today but should not be treated as guaranteed public surface for the first compiled release:

- `rpt native`
- `rpt hybrid`

Reason:

- they expose implementation modes, not product-level behavior
- they are useful for engineering and fallback work, but not required for the first closed distribution

They may remain available behind internal/debug flags, but should not define the public product contract.

## Local State Contract

The public CLI may maintain local state under:

- `~/.rpt/`

Expected contents:

- local model registry
- installed model metadata
- default model selection

The exact file format may change, but the user-facing behavior must remain stable.

## Public Error Contract

At minimum, the compiled CLI must fail clearly for:

- missing model
- invalid bundle layout
- unsupported runtime environment
- incompatible GPU/runtime stack
- missing required artifact files

## Non-Goals For Public v1

- Apple Silicon support
- CPU-first mass-market packaging
- Ollama compatibility
- exposing every internal runtime mode as a public command

## Relationship To Current Implementation

The current Python CLI is the behavioral reference.

The compiled implementation may change internals, but it should preserve the product contract described here.
