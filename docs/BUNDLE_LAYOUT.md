# Bundle Layout

The runtime expects a model bundle directory with this structure:

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

The current production bundle is generated from the main RPT repository and is not committed to this repository.
