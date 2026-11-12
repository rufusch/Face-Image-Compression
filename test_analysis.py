"""Numerical checks for integral conservation, diffusion, and rank optimization."""
from io import BytesIO
import base64
import json
import unittest
import numpy as np
from PIL import Image
from backend import area_resize, heat_denoise, optimize_rank, compress, reconstruct_npz


class AnalysisModuleTests(unittest.TestCase):
    def test_integral_block_and_fractional_average(self):
        image = Image.fromarray(np.array([[0, 10], [20, 30]], dtype=np.uint8))
        resized, info = area_resize(image, 1)
        self.assertEqual(resized.getpixel((0, 0)), 15)
        self.assertAlmostEqual(info['weight_sum'], 1)
        self.assertAlmostEqual(info['footprint_area'], 4)
        # Three cells averaged into two footprints: [0, 1.5], [1.5, 3].
        image = Image.fromarray(np.array([[0, 60, 120]], dtype=np.uint8))
        resized, info = area_resize(image, 2)
        np.testing.assert_array_equal(np.asarray(resized), [[20, 100]])
        self.assertAlmostEqual(info['integrated_output_mean'], info['source_mean'])

    def test_integral_preserves_constant_color(self):
        image = Image.new('RGB', (13, 9), (20, 50, 100))
        resized, info = area_resize(image, 7)
        self.assertEqual(resized.size, (7, 4))
        np.testing.assert_array_equal(np.asarray(resized)[0, 0], [20, 50, 100])

    def test_heat_constant_conservation_and_stability(self):
        A = np.full((10, 12, 3), 51, dtype=np.float32)
        result, info = heat_denoise(A, steps=5)
        np.testing.assert_array_equal(A, result)
        self.assertLessEqual(info['mu'], 0.25)
        impulse = np.zeros((9, 9)); impulse[4, 4] = 100
        result, info = heat_denoise(impulse, steps=5)
        self.assertAlmostEqual(float(result.sum()), 100, places=4)
        self.assertLess(result.max(), 100)
        self.assertGreaterEqual(result.min(), 0)
        self.assertLess(info['history'][-1]['variance'], info['history'][0]['variance'])
        with self.assertRaises(ValueError):
            heat_denoise(A, dt=0.3)

    def test_heat_reduces_controlled_noise(self):
        clean = np.full((30, 30), 120.0)
        noisy = clean+np.random.default_rng(42).normal(0, 15, clean.shape)
        denoised, _ = heat_denoise(noisy, 3)
        self.assertLess(np.mean((denoised-clean)**2), np.mean((noisy-clean)**2))

    def test_optimizer_checks_discrete_cost_and_derivative(self):
        s = np.exp(-np.arange(30)/5)*100
        info = optimize_rank([s], 30, 40, .1)
        self.assertEqual(info['recommended_rank'], int(np.argmin(info['costs']))+1)
        self.assertTrue(info['model_available'])
        if 1 < info['critical_point'] < 30:
            x, a, b = info['critical_point'], info['a'], info['b']
            derivative = -a*b*np.exp(-b*x)+.1*4*(30+40+1)/(30*40)
            self.assertAlmostEqual(derivative, 0, places=10)
            self.assertGreater(info['second_derivative'], 0)
        heavier = optimize_rank([s], 30, 40, 1)
        self.assertLessEqual(heavier['recommended_rank'], info['recommended_rank'])
        self.assertEqual(optimize_rank([np.zeros(5)], 5, 5)['recommended_rank'], 1)

    def test_all_modules_pipeline_and_factor_download(self):
        pixels = np.tile(np.arange(60, dtype=np.uint8)*4, (40, 1))
        stream = BytesIO(); Image.fromarray(pixels).save(stream, 'PNG'); stream.seek(0)
        result = compress(stream, max_side=30, noise_std=15, denoise_steps=3, auto_rank=True)
        self.assertEqual((result['width'], result['height']), (30, 20))
        self.assertTrue(result['analysis']['resizing']['enabled'])
        self.assertEqual(result['rank'], result['analysis']['rank_optimization']['recommended_rank'])
        self.assertEqual(result['analysis']['denoising']['steps'], 3)
        json.dumps(result, allow_nan=False)
        entry = result['results'][0]
        restored = reconstruct_npz(BytesIO(base64.b64decode(entry['download'])))
        preview = Image.open(BytesIO(base64.b64decode(entry['preview'])))
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(preview))


if __name__ == '__main__':
    unittest.main()
