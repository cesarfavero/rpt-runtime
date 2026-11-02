from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import urllib.parse
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from .artifacts import ArtifactError, require_sha256, sha256_file, validate_bundle
except ImportError:  # direct script entrypoint
    from artifacts import ArtifactError, require_sha256, sha256_file, validate_bundle

APP_NAME = 'rpt'
DEFAULT_MODEL_NAME = 'gemma2b-ternary'
DEFAULT_RELEASE_REPO = 'cesarfavero/rpt-model-bundles'
DEFAULT_RELEASE_ASSET = 'production_triton_packed_recovered_l78.zip'
BUNDLE_ROOT_NAME = 'production_triton_packed_recovered_l78'


def app_home() -> Path:
    base = Path(os.environ.get('RPT_HOME', Path.home() / '.rpt'))
    base.mkdir(parents=True, exist_ok=True)
    return base


def bundles_dir() -> Path:
    path = app_home() / 'models'
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return app_home() / 'config.json'


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {'models': {}, 'default_model': None}
    return json.loads(path.read_text(encoding='utf-8'))


def save_config(cfg: dict[str, Any]) -> None:
    path = config_path()
    descriptor, name = tempfile.mkstemp(prefix='.config-', suffix='.json', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(cfg, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def register_model(name: str, path: Path, source: str | None = None) -> None:
    cfg = load_config()
    cfg.setdefault('models', {})[name] = {'path': str(path.absolute()), 'source': source}
    if not cfg.get('default_model'):
        cfg['default_model'] = name
    save_config(cfg)


def resolve_model_dir(name_or_path: str | None) -> Path:
    if name_or_path:
        candidate = Path(name_or_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        cfg = load_config()
        models = cfg.get('models', {})
        if name_or_path in models:
            return Path(models[name_or_path]['path']).expanduser().resolve()
        raise SystemExit(f'Model not found: {name_or_path}')

    cfg = load_config()
    default_name = cfg.get('default_model')
    if default_name and default_name in cfg.get('models', {}):
        return Path(cfg['models'][default_name]['path']).expanduser().resolve()
    raise SystemExit('No model selected. Use `rpt install ...` or pass --model-dir.')


def human_bytes(num: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f'{value:.2f} {unit}'
        value /= 1024
    return f'{num} B'


def download_file(url: str, out_path: Path, *, expected_sha256: str | None = None, timeout: float = 60) -> None:
    if urllib.parse.urlparse(url).scheme != 'https':
        raise ArtifactError('Remote bundles must use HTTPS')
    if expected_sha256 is not None:
        expected_sha256 = require_sha256(expected_sha256, 'Archive SHA256')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response, out_path.open('wb') as out:
        if urllib.parse.urlparse(response.geturl()).scheme != 'https':
            raise ArtifactError('Bundle download redirected to a non-HTTPS URL')
        shutil.copyfileobj(response, out)
    if expected_sha256 is not None and sha256_file(out_path) != expected_sha256:
        out_path.unlink(missing_ok=True)
        raise ArtifactError('Downloaded archive SHA256 mismatch')


def validate_bundle_root(root: Path, *, require_full_checksums: bool = False) -> dict[str, Any]:
    return validate_bundle(root, require_full_checksums=require_full_checksums)


def _model_target(model_name: str) -> Path:
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', model_name or '') is None:
        raise ArtifactError('Model name must be a simple name, not a filesystem path')
    return bundles_dir() / model_name


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


@contextmanager
def _installation_lock():
    """Serialize registry/install transactions; never overwrite another writer."""
    lock = app_home() / '.install.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ArtifactError(f'Another installation or registry update is in progress: {lock}') from exc
    try:
        yield
    finally:
        lock.rmdir()


def _publish_install(candidate: Path, target: Path, model_name: str, source: str, stage: Path) -> Path:
    """Publish an already validated candidate, restoring the old model on failure."""
    # Read the registry before changing either installation. Invalid JSON fails here.
    load_config()
    # Keep recovery outside the temporary staging directory. If the filesystem
    # also prevents rollback, cleanup must not delete the only previous copy.
    backup = target.parent / f'.{model_name}-backup-{uuid.uuid4().hex}'
    previous = target.exists() or target.is_symlink()
    published = False
    try:
        if previous:
            os.replace(target, backup)
        os.replace(candidate, target)
        published = True
        register_model(model_name, target, source=source)
    except BaseException as original:
        try:
            if published:
                _remove_path(target)
            if backup.exists() or backup.is_symlink():
                os.replace(backup, target)
        except OSError as rollback:
            raise ArtifactError(f'Installation failed ({original}); rollback failed ({rollback}). '
                                f'Previous installation is preserved at {backup}') from original
        raise
    if backup.exists() or backup.is_symlink():
        _remove_path(backup)
    return target


def install_from_dir(bundle_dir: Path, model_name: str, *, copy: bool = False, require_full_checksums: bool = False) -> Path:
    bundle_dir = bundle_dir.expanduser().resolve()
    target = _model_target(model_name)
    with _installation_lock():
        validate_bundle_root(bundle_dir, require_full_checksums=require_full_checksums)
        if target.exists() and target.samefile(bundle_dir):
            register_model(model_name, target, source=str(bundle_dir))
            return target
        if bundle_dir.is_relative_to(target.resolve()):
            raise ArtifactError('Bundle source must not be inside the installation being replaced')
        with tempfile.TemporaryDirectory(prefix=f'.{model_name}-stage-', dir=target.parent) as temporary:
            stage = Path(temporary)
            candidate = stage / 'candidate'
            if copy:
                shutil.copytree(bundle_dir, candidate)
                validate_bundle_root(candidate, require_full_checksums=require_full_checksums)
            else:
                candidate.symlink_to(bundle_dir, target_is_directory=True)
            return _publish_install(candidate, target, model_name, str(bundle_dir), stage)


def _extract_zip(zip_path: Path, extract_root: Path, *, max_unpacked_bytes: int = 32 * 1024 ** 3) -> None:
    """Extract regular files only, after checking the complete archive table."""
    with zipfile.ZipFile(zip_path) as archive:
        seen: set[str] = set()
        total = 0
        for item in archive.infolist():
            name = item.filename
            pure = PurePosixPath(name)
            mode = item.external_attr >> 16
            if (not name or '\\' in name or ':' in name or pure.is_absolute()
                    or '..' in pure.parts or name in seen
                    or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
                raise ArtifactError(f'Unsupported archive member: {name!r}')
            seen.add(name)
            total += item.file_size
            if total > max_unpacked_bytes:
                raise ArtifactError(f'Archive exceeds unpacked-size limit ({max_unpacked_bytes} bytes)')
        archive.extractall(extract_root)


def install_from_zip(zip_path: Path, model_name: str, *, expected_sha256: str | None = None,
                     require_full_checksums: bool = False) -> Path:
    if not zip_path.exists():
        raise ArtifactError(f'Zip not found: {zip_path}')
    if expected_sha256 is not None and sha256_file(zip_path) != require_sha256(expected_sha256, 'Archive SHA256'):
        raise ArtifactError('Archive SHA256 mismatch')
    target = _model_target(model_name)
    with _installation_lock(), tempfile.TemporaryDirectory(prefix=f'.{model_name}-stage-', dir=target.parent) as temporary:
        stage = Path(temporary)
        extract_root = stage / 'extracted'
        extract_root.mkdir()
        _extract_zip(zip_path, extract_root)
        entries = list(extract_root.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise ArtifactError('Bundle archive must contain exactly one root directory')
        extracted_root = entries[0]
        validate_bundle_root(extracted_root, require_full_checksums=require_full_checksums)
        return _publish_install(extracted_root, target, model_name, str(zip_path), stage)


def install_from_url(url: str, model_name: str, *, expected_sha256: str | None = None,
                     require_full_checksums: bool = False) -> Path:
    with tempfile.TemporaryDirectory(prefix='rpt-download-') as tmpdir:
        zip_path = Path(tmpdir) / 'bundle.zip'
        download_file(url, zip_path, expected_sha256=expected_sha256)
        return install_from_zip(zip_path, model_name, require_full_checksums=require_full_checksums)


def install_from_github_release(model_name: str, repo: str = DEFAULT_RELEASE_REPO, asset: str = DEFAULT_RELEASE_ASSET,
                                **options: Any) -> Path:
    url = f'https://github.com/{repo}/releases/latest/download/{asset}'
    return install_from_url(url, model_name, **options)

def cmd_install(args: argparse.Namespace) -> int:
    model_name = args.name or DEFAULT_MODEL_NAME
    options = {'require_full_checksums': args.require_full_checksums}
    if args.dir:
        if args.sha256:
            raise ArtifactError('--sha256 applies to archives, not directories')
        target = install_from_dir(Path(args.dir), model_name, copy=args.copy, **options)
    elif args.zip:
        target = install_from_zip(Path(args.zip).expanduser(), model_name, expected_sha256=args.sha256, **options)
    elif args.url:
        target = install_from_url(args.url, model_name, expected_sha256=args.sha256, **options)
    else:
        target = install_from_github_release(model_name, repo=args.release_repo, asset=args.release_asset,
                                             expected_sha256=args.sha256, **options)
    print(json.dumps({'installed_model': model_name, 'path': str(target), 'size': human_bytes(sum(p.stat().st_size for p in target.rglob('*') if p.is_file()))}, ensure_ascii=False, indent=2))
    return 0


def run_python(module_script: str, forwarded: list[str]) -> int:
    root = Path(__file__).resolve().parent
    script = root / module_script
    cmd = [sys.executable, str(script), *forwarded]
    return subprocess.call(cmd)


def inject_model_dir(forwarded: list[str], model_dir: Path) -> list[str]:
    if '--model-dir' in forwarded:
        return forwarded
    return ['--model-dir', str(model_dir), *forwarded]


def cmd_run(args: argparse.Namespace) -> int:
    model_dir = resolve_model_dir(args.model_dir)
    forwarded = []
    if args.prompt:
        forwarded.extend(['--prompt', args.prompt])
    if args.max_new_tokens is not None:
        forwarded.extend(['--max-new-tokens', str(args.max_new_tokens)])
    if args.json:
        forwarded.append('--json')
    if args.allow_fallback:
        forwarded.append('--allow-fallback')
    if args.prewarm:
        forwarded.append('--prewarm')
    if args.raw_prompt:
        forwarded.append('--raw-prompt')
    if args.require_full_checksums:
        forwarded.append('--require-full-checksums')
    forwarded.extend(['--repeat', str(args.repeat)])
    forwarded = inject_model_dir(forwarded, model_dir)
    return run_python('run_triton_inference.py', forwarded)


def cmd_native(args: argparse.Namespace) -> int:
    model_dir = resolve_model_dir(args.model_dir)
    forwarded = inject_model_dir(['--prompt', args.prompt, '--max-new-tokens', str(args.max_new_tokens)], model_dir)
    return run_python('run_native_inference.py', forwarded)


def cmd_hybrid(args: argparse.Namespace) -> int:
    model_dir = resolve_model_dir(args.model_dir)
    forwarded = inject_model_dir(['--prompt', args.prompt, '--max-new-tokens', str(args.max_new_tokens)], model_dir)
    return run_python('run_inference.py', forwarded)


def cmd_ls(_: argparse.Namespace) -> int:
    cfg = load_config()
    models = cfg.get('models', {})
    payload = []
    for name, meta in models.items():
        path = Path(meta['path'])
        payload.append({
            'name': name,
            'path': str(path),
            'exists': path.exists(),
            'default': name == cfg.get('default_model'),
            'source': meta.get('source'),
        })
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_use(args: argparse.Namespace) -> int:
    with _installation_lock():
        cfg = load_config()
        if args.name not in cfg.get('models', {}):
            raise SystemExit(f'Model not installed: {args.name}')
        cfg['default_model'] = args.name
        save_config(cfg)
    print(json.dumps({'default_model': args.name}, ensure_ascii=False, indent=2))
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    result = {
        'python': sys.version.split()[0],
        'app_home': str(app_home()),
        'models_dir': str(bundles_dir()),
        'hf_token_present': bool(os.environ.get('HF_TOKEN')),
    }
    try:
        import torch
        result['torch'] = torch.__version__
        result['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result['cuda_device_name'] = torch.cuda.get_device_name(0)
    except Exception as exc:
        result['torch_error'] = str(exc)
    try:
        import triton  # noqa: F401
        result['triton_available'] = True
    except Exception as exc:
        result['triton_available'] = False
        result['triton_error'] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rpt', description='RPT runtime CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    p_install = sub.add_parser('install', help='Install a model bundle')
    p_install.add_argument('--name', help='Local installed model name', default=DEFAULT_MODEL_NAME)
    install_source = p_install.add_mutually_exclusive_group()
    install_source.add_argument('--zip', help='Local bundle zip path')
    install_source.add_argument('--dir', help='Local extracted bundle directory')
    install_source.add_argument('--url', help='Direct HTTPS bundle zip URL')
    p_install.add_argument('--copy', action='store_true', help='Copy local extracted bundle instead of symlink registration')
    p_install.add_argument('--sha256', help='Expected archive SHA256 obtained from trusted release metadata')
    p_install.add_argument('--require-full-checksums', action='store_true', help='Require checksums for every bundle file')
    p_install.add_argument('--release-repo', default=DEFAULT_RELEASE_REPO)
    p_install.add_argument('--release-asset', default=DEFAULT_RELEASE_ASSET)
    p_install.set_defaults(func=cmd_install)

    p_run = sub.add_parser('run', help='Run Triton path')
    p_run.add_argument('--model-dir', help='Installed model name or bundle dir')
    p_run.add_argument('--prompt', required=True)
    p_run.add_argument('--max-new-tokens', type=int, default=64)
    p_run.add_argument('--json', action='store_true')
    p_run.add_argument('--allow-fallback', action='store_true', help='Allow unpacked PyTorch execution; excluded from official Triton results')
    p_run.add_argument('--prewarm', action='store_true', help='Execute bundled warmup prompts in this process before generation')
    p_run.add_argument('--raw-prompt', action='store_true', help='Do not apply the historical short-answer template')
    p_run.add_argument('--repeat', type=int, default=1, help='Repeat in one loaded process and report first/subsequent latency separately')
    p_run.add_argument('--require-full-checksums', action='store_true', help='Require checksums for every bundle file')
    p_run.set_defaults(func=cmd_run)

    p_native = sub.add_parser('native', help='Run native structural path')
    p_native.add_argument('--model-dir', help='Installed model name or bundle dir')
    p_native.add_argument('--prompt', required=True)
    p_native.add_argument('--max-new-tokens', type=int, default=64)
    p_native.set_defaults(func=cmd_native)

    p_hybrid = sub.add_parser('hybrid', help='Run materialized hybrid path')
    p_hybrid.add_argument('--model-dir', help='Installed model name or bundle dir')
    p_hybrid.add_argument('--prompt', required=True)
    p_hybrid.add_argument('--max-new-tokens', type=int, default=64)
    p_hybrid.set_defaults(func=cmd_hybrid)

    p_ls = sub.add_parser('ls', help='List installed models')
    p_ls.set_defaults(func=cmd_ls)

    p_use = sub.add_parser('use', help='Select default model')
    p_use.add_argument('name')
    p_use.set_defaults(func=cmd_use)

    p_doctor = sub.add_parser('doctor', help='Inspect local runtime environment')
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (ArtifactError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'rpt: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())

