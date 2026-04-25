from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

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
    config_path().write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


def register_model(name: str, path: Path, source: str | None = None) -> None:
    cfg = load_config()
    cfg.setdefault('models', {})[name] = {'path': str(path.resolve()), 'source': source}
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


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, out_path.open('wb') as out:
        shutil.copyfileobj(response, out)


def validate_bundle_root(root: Path) -> None:
    required = [
        root / 'student_pruned',
        root / 'adapter_recovered_l78',
        root / 'packed' / 'manifest.json',
        root / 'packed' / 'mlp_ternary_2bit.bin',
        root / 'runtime' / 'run_triton_inference.py',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit('Invalid bundle layout. Missing:\n' + '\n'.join(f'- {m}' for m in missing))


def install_from_dir(bundle_dir: Path, model_name: str, *, copy: bool = False) -> Path:
    bundle_dir = bundle_dir.expanduser().resolve()
    validate_bundle_root(bundle_dir)
    target = bundles_dir() / model_name
    if target.exists():
        if target.samefile(bundle_dir):
            register_model(model_name, target, source=str(bundle_dir))
            return target
        shutil.rmtree(target)
    if copy:
        shutil.copytree(bundle_dir, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(bundle_dir, target, target_is_directory=True)
    register_model(model_name, target, source=str(bundle_dir))
    return target


def install_from_zip(zip_path: Path, model_name: str) -> Path:
    if not zip_path.exists():
        raise SystemExit(f'Zip not found: {zip_path}')
    target = bundles_dir() / model_name
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    extract_root = bundles_dir() / f'.{model_name}.extracting'
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)
        extracted_root = extract_root / BUNDLE_ROOT_NAME
        if not extracted_root.exists():
            roots = [p for p in extract_root.iterdir() if p.is_dir()]
            if len(roots) != 1:
                raise SystemExit('Unexpected bundle zip layout.')
            extracted_root = roots[0]
        validate_bundle_root(extracted_root)
        extracted_root.rename(target)
    finally:
        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)
    register_model(model_name, target, source=str(zip_path))
    return target


def install_from_url(url: str, model_name: str) -> Path:
    with tempfile.TemporaryDirectory(prefix='rpt-download-') as tmpdir:
        zip_path = Path(tmpdir) / 'bundle.zip'
        download_file(url, zip_path)
        return install_from_zip(zip_path, model_name)


def install_from_github_release(model_name: str, repo: str = DEFAULT_RELEASE_REPO, asset: str = DEFAULT_RELEASE_ASSET) -> Path:
    url = f'https://github.com/{repo}/releases/latest/download/{asset}'
    return install_from_url(url, model_name)

def cmd_install(args: argparse.Namespace) -> int:
    model_name = args.name or DEFAULT_MODEL_NAME
    if args.dir:
        target = install_from_dir(Path(args.dir), model_name, copy=args.copy)
    elif args.zip:
        target = install_from_zip(Path(args.zip).expanduser(), model_name)
    elif args.url:
        target = install_from_url(args.url, model_name)
    else:
        target = install_from_github_release(model_name, repo=args.release_repo, asset=args.release_asset)
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
    p_install.add_argument('--zip', help='Local bundle zip path')
    p_install.add_argument('--dir', help='Local extracted bundle directory')
    p_install.add_argument('--copy', action='store_true', help='Copy local extracted bundle instead of symlink registration')
    p_install.add_argument('--url', help='Direct bundle zip URL')
    p_install.add_argument('--release-repo', default=DEFAULT_RELEASE_REPO)
    p_install.add_argument('--release-asset', default=DEFAULT_RELEASE_ASSET)
    p_install.set_defaults(func=cmd_install)

    p_run = sub.add_parser('run', help='Run Triton path')
    p_run.add_argument('--model-dir', help='Installed model name or bundle dir')
    p_run.add_argument('--prompt', required=True)
    p_run.add_argument('--max-new-tokens', type=int, default=64)
    p_run.add_argument('--json', action='store_true')
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
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
