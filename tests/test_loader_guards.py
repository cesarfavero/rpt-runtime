"""Exercise actual loader control flow with stdlib fakes, not GPU arithmetic.

The selected function bodies are compiled from triton_loader.py because this
host need not have torch installed. Numerical checks remain in the optional
hardware suite; these tests check hash propagation and device selection only.
"""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import types
import unittest

from runtime import artifacts
from tests.helpers import make_bundle


def loader_functions(names, namespace):
    path = Path(__file__).parents[1] / 'runtime/triton_loader.py'
    source = ast.parse(path.read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError('Requested loader functions are missing')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


class LoaderHashGuardTests(unittest.TestCase):
    def test_verified_loader_rejects_null_hash_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_bundle(Path(directory) / 'bundle')
            manifest_path = root / 'production_manifest.json'
            production = json.loads(manifest_path.read_text())
            production['expected_packed_sha256'] = None
            manifest_path.write_text(json.dumps(production))
            packed = root / 'packed/mlp_ternary_2bit.bin'
            packed.write_bytes(b'\x00' * packed.stat().st_size)
            namespace = {name: getattr(artifacts, name) for name in (
                'read_json', 'resolve_expected_packed_sha256', 'validate_packed_manifest',
                'verify_bundle_checksums', 'EXPECTED_TENSOR_COUNT')}
            namespace.update(Path=Path, triton=object(),
                             torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
            loader = loader_functions({'resolve_layout', 'load_triton_packed_recovered_l78'}, namespace)
            with self.assertRaisesRegex(artifacts.ArtifactError, 'production.expected_packed_sha256'):
                loader['load_triton_packed_recovered_l78'](root, verify_sha=True, require_triton=True)


class Tensor:
    is_cuda = True
    dtype = 'float32'

    def __init__(self, shape, device='cuda:1'):
        self.shape = shape
        self.device = device

    def reshape(self, *shape):
        return Tensor(tuple(self.shape[0] if size == -1 else size for size in shape), self.device)

    def contiguous(self):
        return self

    def to(self, **kwargs):
        return Tensor(self.shape, kwargs.get('device', self.device))

    def stride(self, dimension):
        return self.shape[1] if dimension == 0 else 1


class DeviceSelectionTests(unittest.TestCase):
    def test_launch_enters_input_device_and_restores_previous_device(self):
        for fail in (False, True):
            with self.subTest(launch_failure=fail):
                current = ['cuda:0']
                calls = []

                @contextmanager
                def device_guard(device):
                    previous = current[0]
                    current[0] = device
                    try:
                        yield
                    finally:
                        current[0] = previous

                class Kernel:
                    def __getitem__(inner, grid):
                        def launch(x, packed, out, *args, **kwargs):
                            self.assertEqual(current[0], x.device)
                            self.assertEqual(current[0], packed.device)
                            self.assertEqual(current[0], out.device)
                            calls.append(current[0])
                            if fail:
                                raise RuntimeError('simulated launch failure')
                        return launch

                torch = types.SimpleNamespace(
                    cuda=types.SimpleNamespace(device=device_guard), float32='float32',
                    empty=lambda shape, device, dtype: Tensor(shape, device))
                namespace = loader_functions({'triton_ternary_linear'}, {
                    'torch': torch, 'triton': types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
                    '_ternary_linear_kernel': Kernel(),
                })
                def run():
                    return namespace['triton_ternary_linear'](Tensor((1, 3)), Tensor((2,)),
                                                              in_features=3, out_features=2, scale=0.5)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'simulated launch failure'):
                        run()
                else:
                    self.assertEqual(run().device, 'cuda:1')
                self.assertEqual(calls, ['cuda:1'])
                self.assertEqual(current[0], 'cuda:0')
