import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from runtime import cli
from runtime.artifacts import ArtifactError, sha256_file
from tests.helpers import add_checksums, make_bundle


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {'RPT_HOME': str(self.root / 'store')})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.first = make_bundle(self.root / 'first', marker='first', full_checksums=True)
        self.second = make_bundle(self.root / 'second', marker='second', full_checksums=True)

    def archive(self, source: Path) -> Path:
        output = self.root / 'bundle.zip'
        with zipfile.ZipFile(output, 'w') as archive:
            for path in source.rglob('*'):
                if path.is_file():
                    archive.write(path, f'bundle/{path.relative_to(source).as_posix()}')
        return output

    def test_replaces_symlink_without_touching_source(self):
        target = cli.install_from_dir(self.first, 'demo')
        self.assertTrue(target.is_symlink())
        cli.install_from_dir(self.second, 'demo')
        self.assertEqual((target / 'marker.txt').read_text(), 'second')
        self.assertEqual((self.first / 'marker.txt').read_text(), 'first')
        self.assertEqual(cli.load_config()['models']['demo']['path'], str(target))

    def test_bad_archive_preserves_install_and_registry(self):
        target = cli.install_from_dir(self.first, 'demo', copy=True)
        registry = cli.config_path().read_bytes()
        bad = self.root / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as archive:
            archive.writestr('bundle/production_manifest.json', '{}')
        with self.assertRaises(ArtifactError):
            cli.install_from_zip(bad, 'demo')
        self.assertEqual((target / 'marker.txt').read_text(), 'first')
        self.assertEqual(cli.config_path().read_bytes(), registry)

    def test_unloadable_candidate_with_complete_checksums_preserves_install(self):
        target = cli.install_from_dir(self.first, 'demo', copy=True)
        registry = cli.config_path().read_bytes()
        (self.second / 'student_pruned/model.safetensors').unlink()
        add_checksums(self.second)
        for archived in (False, True):
            with self.subTest(archived=archived), self.assertRaises(ArtifactError):
                if archived:
                    cli.install_from_zip(self.archive(self.second), 'demo', require_full_checksums=True)
                else:
                    cli.install_from_dir(self.second, 'demo', copy=True, require_full_checksums=True)
            self.assertEqual((target / 'marker.txt').read_text(), 'first')
            self.assertEqual(cli.config_path().read_bytes(), registry)

    def test_null_digest_cannot_publish_changed_packed_bytes(self):
        target = cli.install_from_dir(self.first, 'demo', copy=True)
        registry = cli.config_path().read_bytes()
        path = self.second / 'production_manifest.json'
        production = json.loads(path.read_text())
        production['expected_packed_sha256'] = None
        path.write_text(json.dumps(production))
        packed = self.second / 'packed/mlp_ternary_2bit.bin'
        packed.write_bytes(b'\x00' * packed.stat().st_size)
        add_checksums(self.second)
        with self.assertRaisesRegex(ArtifactError, 'SHA256'):
            cli.install_from_dir(self.second, 'demo', copy=True, require_full_checksums=True)
        self.assertEqual((target / 'marker.txt').read_text(), 'first')
        self.assertEqual(cli.config_path().read_bytes(), registry)

    def test_registry_failure_rolls_back_install(self):
        target = cli.install_from_dir(self.first, 'demo', copy=True)
        registry = cli.config_path().read_bytes()
        with patch.object(cli, 'register_model', side_effect=OSError('simulated registry write failure')):
            with self.assertRaises(OSError):
                cli.install_from_zip(self.archive(self.second), 'demo')
        self.assertEqual((target / 'marker.txt').read_text(), 'first')
        self.assertEqual(cli.config_path().read_bytes(), registry)

    def test_publish_failure_restores_old_directory(self):
        target = cli.install_from_dir(self.first, 'demo', copy=True)
        original_replace = os.replace

        def replace(source, destination):
            if Path(source).name == 'bundle' and Path(destination) == target:
                raise OSError('simulated publish failure')
            return original_replace(source, destination)

        with patch.object(cli.os, 'replace', side_effect=replace):
            with self.assertRaisesRegex(OSError, 'publish failure'):
                cli.install_from_zip(self.archive(self.second), 'demo')
        self.assertEqual((target / 'marker.txt').read_text(), 'first')

    def test_failed_rollback_keeps_previous_copy_outside_temporary_stage(self):
        target = cli.install_from_dir(self.first, 'demo', copy=True)
        original_replace = os.replace

        def replace(source, destination):
            if Path(destination) == target:
                raise OSError('simulated destination unavailable')
            return original_replace(source, destination)

        with patch.object(cli.os, 'replace', side_effect=replace):
            with self.assertRaisesRegex(ArtifactError, 'Previous installation is preserved'):
                cli.install_from_zip(self.archive(self.second), 'demo')
        backups = list(cli.bundles_dir().glob('.demo-backup-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'marker.txt').read_text(), 'first')

    def test_config_write_failure_is_atomic(self):
        cli.install_from_dir(self.first, 'demo')
        before = cli.config_path().read_bytes()
        with patch.object(cli.os, 'replace', side_effect=OSError('simulated config publish failure')):
            with self.assertRaises(OSError):
                cli.save_config({'models': {}, 'default_model': None})
        self.assertEqual(cli.config_path().read_bytes(), before)
        self.assertEqual(list(cli.app_home().glob('.config-*.json')), [])

    def test_valid_archive_replacement_with_checksums(self):
        target = cli.install_from_dir(self.first, 'demo')
        archive = self.archive(self.second)
        cli.install_from_zip(archive, 'demo', expected_sha256=sha256_file(archive), require_full_checksums=True)
        self.assertFalse(target.is_symlink())
        self.assertEqual((target / 'marker.txt').read_text(), 'second')
        self.assertEqual((self.first / 'marker.txt').read_text(), 'first')

    def test_archive_checksum_failure_preserves_install(self):
        target = cli.install_from_dir(self.first, 'demo')
        with self.assertRaisesRegex(ArtifactError, 'SHA256 mismatch'):
            cli.install_from_zip(self.archive(self.second), 'demo', expected_sha256='0' * 64)
        self.assertEqual(target.resolve(), self.first)

    def test_path_traversal_and_symlink_archive_members_are_rejected(self):
        for name in ('../../outside', '/absolute', 'C:/absolute', 'bundle\\outside'):
            archive_path = self.root / 'malformed.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr(name, 'bad')
            with self.subTest(name=name), self.assertRaises(ArtifactError):
                cli.install_from_zip(archive_path, 'demo')
        info = zipfile.ZipInfo('bundle/link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive_path, 'w') as archive:
            archive.writestr(info, '/outside')
        with self.assertRaises(ArtifactError):
            cli.install_from_zip(archive_path, 'demo')

    def test_invalid_model_names_and_store_lock(self):
        for name in ('../escape', '/tmp/escape', '.', '', 'a/b'):
            with self.subTest(name=name), self.assertRaises(ArtifactError):
                cli.install_from_dir(self.first, name)
        with cli._installation_lock():
            with self.assertRaisesRegex(ArtifactError, 'in progress'):
                cli.install_from_dir(self.first, 'demo')

    def test_reinstall_same_directory_and_dangling_symlink(self):
        target = cli.install_from_dir(self.first, 'demo')
        cli.install_from_dir(self.first, 'demo')
        target.unlink()
        target.symlink_to(self.root / 'missing')
        cli.install_from_dir(self.second, 'demo')
        self.assertEqual(target.resolve(), self.second)

    def test_download_requires_https_and_checksums(self):
        with self.assertRaisesRegex(ArtifactError, 'HTTPS'):
            cli.download_file('http://example.test/bundle.zip', self.root / 'download')
        response = io.BytesIO(b'bad')
        response.geturl = lambda: 'https://example.test/bundle.zip'
        with patch.object(cli.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(ArtifactError, 'SHA256 mismatch'):
                cli.download_file('https://example.test/bundle.zip', self.root / 'download', expected_sha256='0' * 64)
        self.assertFalse((self.root / 'download').exists())
