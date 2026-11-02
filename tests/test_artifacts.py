import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.artifacts import (
    ArtifactError, EXPECTED_TENSOR_COUNT, LEGACY_PACKED_SHA256,
    resolve_expected_packed_sha256, validate_bundle, validate_packed_manifest,
)
from tests.helpers import add_checksums, make_bundle, packed_manifest, tiny_safetensors


class PackedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / 'weights.bin'
        self.binary.write_bytes(b'\x24')

    def test_valid_packed_and_hash(self):
        report = validate_packed_manifest(packed_manifest(), self.binary, expected_tensor_count=1,
                                          expected_sha256=hashlib.sha256(b'\x24').hexdigest())
        self.assertEqual(report['packed_bytes'], 1)
        self.assertTrue(report['codes_validated'])

    def test_rejects_shape_count_offset_scale_and_padding_metadata(self):
        invalid = [('shape', [2, 4]), ('numel', True), ('packed_bytes', 2), ('offset', 1),
                   ('scale', float('nan')), ('scale', -1), ('scale', True), ('pad_codes', 1)]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                manifest = packed_manifest()
                manifest['tensors'][0][key] = value
                with self.assertRaises(ArtifactError):
                    validate_packed_manifest(manifest, self.binary)

    def test_rejects_reserved_codes_in_all_positions(self):
        for shift in (0, 2, 4, 6):
            self.binary.write_bytes(bytes([3 << shift]))
            with self.subTest(shift=shift), self.assertRaises(ArtifactError):
                validate_packed_manifest(packed_manifest(), self.binary)

    def test_rejects_nonzero_padding(self):
        self.binary.write_bytes(b'\x64')
        with self.assertRaisesRegex(ArtifactError, 'padding'):
            validate_packed_manifest(packed_manifest(3), self.binary)
        self.binary.write_bytes(b'\x24')
        validate_packed_manifest(packed_manifest(3), self.binary)

    def test_rejects_truncation_and_trailing_bytes(self):
        for blob in (b'', b'\x24\x24'):
            self.binary.write_bytes(blob)
            with self.assertRaises(ArtifactError):
                validate_packed_manifest(packed_manifest(), self.binary)

    def test_rejects_wrong_hash_and_count(self):
        with self.assertRaisesRegex(ArtifactError, 'SHA256 mismatch'):
            validate_packed_manifest(packed_manifest(), self.binary, expected_sha256='0' * 64)
        with self.assertRaisesRegex(ArtifactError, 'Expected 78'):
            validate_packed_manifest(packed_manifest(), self.binary, expected_tensor_count=78)

    def test_rejects_duplicate_names_and_overlapping_ranges(self):
        manifest = packed_manifest()
        self.binary.write_bytes(b'\x24\x24')
        manifest['tensors'].append(copy.deepcopy(manifest['tensors'][0]))
        with self.assertRaisesRegex(ArtifactError, 'duplicate'):
            validate_packed_manifest(manifest, self.binary)
        manifest['tensors'][1]['name'] = 'model.layers.0.mlp.up_proj.weight'
        with self.assertRaisesRegex(ArtifactError, 'overlapping'):
            validate_packed_manifest(manifest, self.binary)


class BundleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = make_bundle(Path(self.temporary.name) / 'bundle')

    def test_legacy_layout_still_verifies_packed(self):
        report = validate_bundle(self.root)
        self.assertFalse(report['complete_checksums'])
        self.assertEqual(report['packed']['num_tensors'], EXPECTED_TENSOR_COUNT)
        self.assertEqual(report['model_layout']['validation'], 'minimum_layout_only')
        (self.root / 'packed/mlp_ternary_2bit.bin').write_bytes(b'\x00' * EXPECTED_TENSOR_COUNT)
        with self.assertRaisesRegex(ArtifactError, 'SHA256 mismatch'):
            validate_bundle(self.root)

    def test_null_or_malformed_digest_cannot_disable_verification(self):
        manifest_path = self.root / 'production_manifest.json'
        production = json.loads(manifest_path.read_text())
        (self.root / 'packed/mlp_ternary_2bit.bin').write_bytes(b'\x00' * EXPECTED_TENSOR_COUNT)
        for bad_hash in (None, '', False, 123, 'abc', 'g' * 64):
            with self.subTest(digest=bad_hash):
                production['expected_packed_sha256'] = bad_hash
                manifest_path.write_text(json.dumps(production))
                with self.assertRaisesRegex(ArtifactError, 'SHA256'):
                    validate_bundle(self.root)

    def test_absent_digest_uses_historical_digest_not_unchecked_mode(self):
        self.assertEqual(resolve_expected_packed_sha256({}), LEGACY_PACKED_SHA256)
        manifest_path = self.root / 'production_manifest.json'
        production = json.loads(manifest_path.read_text())
        del production['expected_packed_sha256']
        manifest_path.write_text(json.dumps(production))
        with self.assertRaisesRegex(ArtifactError, 'SHA256 mismatch'):
            validate_bundle(self.root)

    def test_override_is_validated_and_agrees_with_present_manifest(self):
        good = 'a' * 64
        self.assertEqual(resolve_expected_packed_sha256({}, override=good.upper()), good)
        self.assertEqual(resolve_expected_packed_sha256({'expected_packed_sha256': good}, override=good), good)
        for bad in ('', False, 'bad'):
            with self.subTest(override=bad), self.assertRaises(ArtifactError):
                resolve_expected_packed_sha256({}, override=bad)
        with self.assertRaises(ArtifactError):
            resolve_expected_packed_sha256({'expected_packed_sha256': None}, override=good)
        with self.assertRaisesRegex(ArtifactError, 'disagrees'):
            resolve_expected_packed_sha256({'expected_packed_sha256': 'b' * 64}, override=good)

    def test_manifest_cannot_change_runtime_tensor_count(self):
        manifest_path = self.root / 'production_manifest.json'
        production = json.loads(manifest_path.read_text())
        for count in (None, True, 1, 77, 79, '78'):
            with self.subTest(count=count):
                production['expected_packed_mlp_tensors'] = count
                manifest_path.write_text(json.dumps(production))
                with self.assertRaises(ArtifactError):
                    validate_bundle(self.root)

    def test_actual_tensor_table_must_still_have_78_entries(self):
        path = self.root / 'packed/manifest.json'
        path.write_text(json.dumps(packed_manifest()))
        (self.root / 'packed/mlp_ternary_2bit.bin').write_bytes(b'\x24')
        with self.assertRaisesRegex(ArtifactError, 'Expected 78 tensors'):
            validate_bundle(self.root)

    def test_required_model_files_cannot_be_missing_or_empty(self):
        for relative in ('student_pruned/config.json', 'student_pruned/tokenizer_config.json',
                         'student_pruned/tokenizer.json', 'student_pruned/model.safetensors',
                         'adapter_recovered_l78/adapter_config.json',
                         'adapter_recovered_l78/adapter_model.safetensors'):
            path = self.root / relative
            original = path.read_bytes()
            for mode in ('missing', 'empty'):
                with self.subTest(path=relative, mode=mode):
                    path.unlink() if mode == 'missing' else path.write_bytes(b'')
                    with self.assertRaises(ArtifactError):
                        validate_bundle(self.root)
                    path.write_bytes(original)

    def test_empty_configs_and_tokenizer_payload_are_rejected(self):
        for relative in ('student_pruned/config.json', 'student_pruned/tokenizer.json',
                         'adapter_recovered_l78/adapter_config.json'):
            path = self.root / relative
            original = path.read_bytes()
            path.write_text('{}')
            with self.subTest(path=relative), self.assertRaises(ArtifactError):
                validate_bundle(self.root)
            path.write_bytes(original)

    def test_valid_sharded_model_and_sentencepiece_layout(self):
        student = self.root / 'student_pruned'
        (student / 'model.safetensors').unlink()
        tiny_safetensors(student / 'model-00001-of-00002.safetensors', 'first.weight')
        tiny_safetensors(student / 'model-00002-of-00002.safetensors', 'second.weight')
        (student / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'first.weight': 'model-00001-of-00002.safetensors',
            'second.weight': 'model-00002-of-00002.safetensors',
        }}))
        (student / 'tokenizer.json').unlink()
        (student / 'tokenizer.model').write_bytes(b'nonempty-sentencepiece-layout-fixture')
        report = validate_bundle(self.root)
        self.assertEqual(len(report['model_layout']['student_weights']['files']), 2)
        self.assertEqual(report['model_layout']['tokenizer_files'], ['tokenizer.model'])

    def test_shard_index_rejects_missing_empty_and_external_references(self):
        student = self.root / 'student_pruned'
        (student / 'model.safetensors').unlink()
        index = student / 'model.safetensors.index.json'
        (student / 'empty.safetensors').write_bytes(b'')
        (self.root / 'outside.safetensors').write_bytes(b'outside')
        for mapping in ({}, {'weight': 'missing.safetensors'}, {'weight': 'empty.safetensors'},
                        {'weight': '../outside.safetensors'}, {'weight': str(self.root / 'outside.safetensors')},
                        {'weight': 12}, {'': 'empty.safetensors'}):
            index.write_text(json.dumps({'weight_map': mapping}))
            with self.subTest(mapping=mapping), self.assertRaises(ArtifactError):
                validate_bundle(self.root)

    def test_complete_checksums_cannot_certify_empty_model_directories(self):
        for directory in (self.root / 'student_pruned', self.root / 'adapter_recovered_l78'):
            for path in directory.iterdir():
                path.unlink()
        add_checksums(self.root)
        with self.assertRaises(ArtifactError):
            validate_bundle(self.root, require_full_checksums=True)

    def test_complete_checksum_coverage_and_corruption(self):
        add_checksums(self.root)
        report = validate_bundle(self.root, require_full_checksums=True)
        self.assertTrue(report['complete_checksums'])
        (self.root / 'marker.txt').write_text('changed')
        with self.assertRaisesRegex(ArtifactError, 'Checksum mismatch'):
            validate_bundle(self.root)

    def test_unlisted_files_are_not_silently_trusted(self):
        add_checksums(self.root)
        (self.root / 'additional.bin').write_bytes(b'x')
        with self.assertRaisesRegex(ArtifactError, 'coverage'):
            validate_bundle(self.root)

    def test_strict_checksums_and_internal_symlinks(self):
        with self.assertRaisesRegex(ArtifactError, 'required'):
            validate_bundle(self.root, require_full_checksums=True)
        (self.root / 'linked').symlink_to(self.root / 'marker.txt')
        with self.assertRaisesRegex(ArtifactError, 'symbolic links'):
            validate_bundle(self.root)
