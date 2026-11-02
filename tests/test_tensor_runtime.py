"""Numerical CPU checks; accelerator execution is tested separately on hardware."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch is not installed; numerical runtime checks pending')
class PackedTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from runtime import triton_loader
        cls.torch = torch
        cls.loader = triton_loader

    def layer(self, **overrides):
        values = {'packed_bytes': b'\x64\x08', 'shape': [2, 3], 'numel': 6,
                  'scale': 0.5, 'name': 'synthetic.weight', 'fallback_to_pytorch': True}
        values.update(overrides)
        return self.loader.TritonPackedTernaryLinear(**values)

    def test_cpu_forward_matches_dense_and_records_fallback(self):
        torch = self.torch
        layer = self.layer()
        x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        dense = torch.tensor([[0.0, 0.5, -0.5], [0.5, 0.0, -0.5]], dtype=torch.float32)
        torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, dense), rtol=0, atol=0)
        report = self.loader.runtime_execution_report(layer, expected_layers=1)
        self.assertEqual(report['fallback_count'], 1)
        self.assertEqual(report['backend'], 'pytorch')
        self.assertFalse(report['packed_triton_execution_verified'])
        with self.assertRaises(RuntimeError):
            self.loader.runtime_execution_report(layer, expected_layers=1, require_triton=True)
        self.loader.reset_runtime_execution(layer)
        self.assertEqual(self.loader.runtime_execution_report(layer, expected_layers=1)['backend'], 'unobserved')

    def test_strict_layer_rejects_cpu_without_counting_a_success(self):
        layer = self.layer(fallback_to_pytorch=False)
        with self.assertRaisesRegex(RuntimeError, 'fallback disabled'):
            layer(self.torch.ones(1, 3))
        self.assertEqual(layer.execution_counts, {'triton': 0, 'pytorch': 0})

    def test_bad_packed_length_reserved_codes_and_padding_fail_before_forward(self):
        for blob in (b'\x64', b'\x64\x08\x00', b'\x67\x08', b'\x64\x18'):
            with self.subTest(blob=blob), self.assertRaises(ValueError):
                self.layer(packed_bytes=blob)

    def test_invalid_shape_and_input_width(self):
        with self.assertRaises(ValueError):
            self.layer(shape=[3, 3])
        with self.assertRaisesRegex(RuntimeError, 'input width'):
            self.layer()(self.torch.ones(1, 2))

    def test_zero_scale_and_noncontiguous_input(self):
        torch = self.torch
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        result = self.layer(scale=0.0)(x)
        torch.testing.assert_close(result, torch.zeros(4, 2), rtol=0, atol=0)

    def test_triton_launch_uses_input_device_when_current_device_differs(self):
        torch = self.torch
        if self.loader.triton is None or torch.cuda.device_count() < 2:
            self.skipTest('Two CUDA GPUs and Triton are required; multi-GPU launch remains a hardware check')
        with torch.cuda.device(0):
            x = torch.tensor([[1.0, 2.0, 3.0]], device='cuda:1')
            packed = torch.tensor([0x64, 0x08], dtype=torch.uint8, device='cuda:1')
            dense = torch.tensor([[0.0, 0.5, -0.5], [0.5, 0.0, -0.5]], device='cuda:1')
            actual = self.loader.triton_ternary_linear(x, packed, out_features=2, in_features=3, scale=0.5)
            torch.cuda.synchronize(device=1)
            torch.testing.assert_close(actual, torch.nn.functional.linear(x, dense), rtol=1e-5, atol=1e-6)
            self.assertEqual(actual.device.index, 1)
            self.assertEqual(torch.cuda.current_device(), 0)
