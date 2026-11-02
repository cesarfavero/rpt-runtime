from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from runtime.artifacts import CHECKSUM_FORMAT, EXPECTED_TENSOR_COUNT, PACKED_FORMAT, sha256_file


def packed_manifest(numel: int = 4) -> dict:
    return {"format": PACKED_FORMAT,
            "codebook": {"0": "0", "1": "+scale", "2": "-scale", "3": "reserved"},
            "summary": {"num_tensors": 1, "packed_total_bytes": (numel + 3) // 4},
            "tensors": [{"name": "model.layers.0.mlp.gate_proj.weight", "shape": [1, numel],
                         "numel": numel, "scale": 0.5, "offset": 0,
                         "packed_bytes": (numel + 3) // 4, "pad_codes": (-numel) % 4}]}


def add_checksums(root: Path) -> None:
    checksums = {path.relative_to(root).as_posix(): sha256_file(path)
                 for path in root.rglob("*") if path.is_file() and path.name != "artifact_checksums.json"}
    (root / "artifact_checksums.json").write_text(json.dumps({"format": CHECKSUM_FORMAT, "files": checksums}))


def tiny_safetensors(path: Path, name: str) -> None:
    """A real, one-element container for layout tests, not a runnable model."""
    header = json.dumps({name: {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.5))


def bundle_packed_manifest() -> dict:
    manifest = packed_manifest()
    manifest["tensors"] = [
        {**packed_manifest()["tensors"][0], "name": f"model.layers.{layer}.mlp.{projection}.weight", "offset": index}
        for index, (layer, projection) in enumerate(
            (layer, projection) for layer in range(26) for projection in ("gate_proj", "up_proj", "down_proj"))
    ]
    manifest["summary"] = {"num_tensors": EXPECTED_TENSOR_COUNT, "packed_total_bytes": EXPECTED_TENSOR_COUNT}
    return manifest


def make_bundle(root: Path, marker: str = "first", full_checksums: bool = False) -> Path:
    for name in ("student_pruned", "adapter_recovered_l78", "packed"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "student_pruned/config.json").write_text(json.dumps({"model_type": "gemma2", "num_hidden_layers": 26}))
    (root / "student_pruned/tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    (root / "student_pruned/tokenizer.json").write_text(json.dumps({"model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}}))
    (root / "adapter_recovered_l78/adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1, "task_type": "CAUSAL_LM"}))
    tiny_safetensors(root / "student_pruned/model.safetensors", "model.embed_tokens.weight")
    tiny_safetensors(root / "adapter_recovered_l78/adapter_model.safetensors", "base_model.model.layers.0.mlp.gate_proj.lora_A.weight")
    (root / "marker.txt").write_text(marker)
    blob = b'\x24' * EXPECTED_TENSOR_COUNT
    (root / "packed/mlp_ternary_2bit.bin").write_bytes(blob)
    (root / "packed/manifest.json").write_text(json.dumps(bundle_packed_manifest()))
    (root / "production_manifest.json").write_text(json.dumps({
        "expected_packed_sha256": hashlib.sha256(blob).hexdigest(), "expected_packed_mlp_tensors": EXPECTED_TENSOR_COUNT}))
    (root / "prewarm_profile.json").write_text(json.dumps({"prewarm_rows": [{"question": "2 + 2?"}]}))
    if full_checksums:
        add_checksums(root)
    return root
