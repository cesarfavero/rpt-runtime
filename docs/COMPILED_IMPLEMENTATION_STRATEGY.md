# Compiled Implementation Strategy

This document chooses how `rpt-runtime` should move from the current Python/source-visible delivery to a closed compiled product surface.

It is based on the current repository state, not on a greenfield redesign.

## Current Technical Reality

The current product behavior already exists, but it is implemented as:

```text
bin/rpt -> python runtime/cli.py -> python runtime entrypoints
```

The current run path depends on:

- Python
- PyTorch
- Triton
- PEFT
- Transformers
- runtime source files

The current bundle contract is also coupled to runtime source visibility because validation still expects:

- `runtime/run_triton_inference.py`

## Decision Frame

We need to choose the shortest path to:

- preserve existing `rpt` product behavior
- stop distributing public runtime source
- avoid freezing the team in a long native port before anything shippable exists

## Options

### Option A. Compiled launcher that still requires local Python

Shape:

- small compiled binary
- binary shells out to Python runtime already installed on the system

Pros:

- fastest to build
- tiny migration effort

Cons:

- weak closure story
- still depends on user Python environment
- still exposes runtime source unless packaged separately
- not a real product-grade distribution

Verdict:

- reject as public product path

### Option B. Compiled wrapper around a privately packaged Python runtime

Shape:

- compiled `rpt` binary becomes the public entrypoint
- Python runtime and dependencies are shipped as internal packaged assets
- runtime source is no longer delivered as loose public files
- bundle contract is cleaned so model bundles stop carrying `runtime/*.py`

Pros:

- preserves almost all current runtime logic
- fast enough to unblock product closure
- much better than current source-visible distribution
- lets the team keep using the proven Triton/PyTorch path while the product surface becomes closed

Cons:

- still not a truly native runtime
- package size will remain large
- internal Python stack still has to be managed carefully
- some reverse-engineering resistance comes from packaging, not from removing Python entirely

Verdict:

- best transition strategy

### Option C. Full native compiled runtime

Shape:

- compiled CLI
- compiled execution layer
- no packaged Python runtime in the shipped product path

Pros:

- strongest closure story
- cleanest product architecture
- best long-term install and portability story

Cons:

- highest engineering cost
- highest risk of slowing product momentum
- requires replacing or wrapping substantial Triton/PyTorch behavior

Verdict:

- correct long-term target
- wrong first move if it blocks closure of the current distribution problem

## Chosen Strategy

### Transition strategy: Option B

Use a compiled `rpt` binary as the public surface and package the current Python runtime privately behind it.

This is the shortest path that:

- preserves current working behavior
- removes loose source-visible delivery
- lets public install return only after the runtime is no longer exposed as public Python files

### Long-term target: Option C

After closure is restored and the public surface is controlled, continue reducing dependence on the packaged Python runtime until the execution path becomes truly native.

## Why This Is The Correct Sequence

The real bottleneck right now is not runtime capability.

The real bottleneck is:

- the current public delivery exposes source-visible runtime files
- the site/install flow had to be disabled because of that

Option B fixes that product problem soonest without discarding the runtime that already works.

## Implementation Plan

### Step 1. Clean the bundle contract

Before any packaging work:

- stop treating `runtime/*.py` as part of the public model bundle
- move bundle validation to model artifacts and metadata only

This change is mandatory for every strategy.

### Step 2. Freeze the public CLI contract

Public v1 commands:

- `install`
- `run`
- `ls`
- `use`
- `doctor`

Internal/debug by default:

- `native`
- `hybrid`

### Step 3. Build a compiled `rpt` front door

Responsibilities:

- command parsing
- config/state management
- bundle registration
- environment checks
- invocation of the internal packaged runtime

Recommended implementation language:

- Rust

Reason:

- good distribution story
- good filesystem/process/network ergonomics
- strong fit for a single-binary CLI

### Step 4. Package the Python runtime privately

The public user should not receive:

- `bin/rpt` shell shim
- loose `runtime/*.py`
- public install flow that advertises direct source-visible bundle execution

The compiled runtime should unwrap or locate the internal runtime assets itself.

### Step 5. Re-enable public install only after closure is restored

Public install must stay disabled until:

- the runtime is no longer delivered as loose public Python files
- the model bundle contract is clean
- the install path is aligned with the compiled surface

## Immediate Engineering Decision

Recommendation:

- do **not** start by porting Triton/PyTorch inference code to native execution
- start by cleaning the bundle contract and building a compiled `rpt` surface around the current proven runtime

That is the highest-leverage move.

## Exit Criteria For This Strategy Stage

This stage is complete when:

- `rpt` is a compiled public entrypoint
- the public bundle contract no longer contains `runtime/*.py`
- install does not depend on public Python runtime files
- `go.rpt.dog/install.sh` can be safely re-enabled without exposing runtime source
