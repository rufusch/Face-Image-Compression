from io import BytesIO
import base64
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import backend


class ComparisonTests(unittest.TestCase):
    def test_rank_25_with_residual_recovers_reference(self):
        rng = np.random.default_rng(25)
        for grayscale, shape in ((True, (64, 64)), (False, (64, 64, 3))):
            pixels = rng.integers(0, 256, shape, dtype=np.uint8)
            upload = BytesIO()
            Image.fromarray(pixels).save(upload, 'PNG')
            upload.seek(0)
            result = backend.compare_ranks(upload, grayscale=grayscale)
            entry = next(item for item in result['comparisons'] if item['rank'] == 25)
            factors = base64.b64decode(entry['download'])
            self.assertFalse(np.array_equal(np.asarray(backend.reconstruct_npz(BytesIO(factors))), pixels))
            archive = backend.exact_recovery_archive(factors, Image.fromarray(pixels))
            np.testing.assert_array_equal(np.asarray(backend.reconstruct_npz(BytesIO(archive))), pixels)

    def test_five_ranks_reuse_one_decomposition(self):
        pixels = np.random.default_rng(8).integers(0, 256, (110, 140), dtype=np.uint8)
        file = BytesIO(); Image.fromarray(pixels).save(file, 'PNG'); file.seek(0)
        with patch('backend.svd', wraps=backend.svd) as decomposition:
            result = backend.compare_ranks(file)
            self.assertEqual(decomposition.call_count, 1)
        self.assertEqual([r['rank'] for r in result['comparisons']], [100, 50, 25, 10, 5])
        errors = [r['metrics']['mse'] for r in result['comparisons']]
        self.assertEqual(errors, sorted(errors))
        for item in result['comparisons']:
            image = Image.open(BytesIO(base64.b64decode(item['preview'])))
            self.assertEqual(image.size, (140, 110))
            restored = backend.reconstruct_npz(BytesIO(base64.b64decode(item['download'])))
            np.testing.assert_array_equal(np.asarray(restored), np.asarray(image))

    def test_small_image_displays_actual_rank(self):
        file = BytesIO(); Image.new('RGB', (20, 15), 'gray').save(file, 'PNG'); file.seek(0)
        result = backend.compare_ranks(file, grayscale=False)
        self.assertEqual([r['rank'] for r in result['comparisons']], [15,15,15,10,5])
        self.assertEqual([r['requested_rank'] for r in result['comparisons']], [100,50,25,10,5])


if __name__ == '__main__':
    unittest.main()
