import unittest

from runtime.execution import summarize_execution
from runtime.metrics import summarize_generation


class ExecutionEvidenceTests(unittest.TestCase):
    def test_unexecuted_layers_are_not_claimed_as_triton(self):
        report = summarize_execution([('layer', {'triton': 0, 'pytorch': 0})], expected_layers=1)
        self.assertEqual(report['backend'], 'unobserved')
        self.assertFalse(report['is_native_ternary_runtime'])
        with self.assertRaises(RuntimeError):
            summarize_execution([('layer', {'triton': 0})], expected_layers=1, require_triton=True)

    def test_strict_evidence_requires_all_layers_and_zero_fallback(self):
        valid = [('a', {'triton': 3}), ('b', {'triton': 3})]
        report = summarize_execution(valid, expected_layers=2, require_triton=True)
        self.assertEqual(report['backend_counts']['triton'], 6)
        self.assertTrue(report['packed_triton_execution_verified'])
        self.assertIsNone(report['is_optimized_kernel'])
        for rows in (valid[:1], [('a', {'triton': 3}), ('b', {'pytorch': 1})]):
            with self.assertRaises(RuntimeError):
                summarize_execution(rows, expected_layers=2, require_triton=True)


class ThroughputTests(unittest.TestCase):
    def test_aggregates_tokens_over_time_not_unweighted_rates(self):
        report = summarize_generation([{'generated_tokens': 1, 'generation_elapsed_sec': 1.0},
                                       {'generated_tokens': 100, 'generation_elapsed_sec': 10.0}])
        self.assertAlmostEqual(report['aggregate_tokens_per_sec'], 101 / 11)
        self.assertNotEqual(report['aggregate_tokens_per_sec'], 5.5)
        self.assertFalse(report['ttft_measured'])

    def test_invalid_timings_are_rejected(self):
        for elapsed in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                summarize_generation([{'generated_tokens': 1, 'generation_elapsed_sec': elapsed}])

    def test_empty_phase_is_explicit(self):
        report = summarize_generation([])
        self.assertIsNone(report['aggregate_tokens_per_sec'])
