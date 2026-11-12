import base64
from io import BytesIO
import unittest
from unittest.mock import patch, MagicMock
import runpy
import numpy as np
from PIL import Image
from app import app
from backend import compress, reconstruct_npz, decompose_channel


def picture(color=False):
    rng = np.random.default_rng(12)
    a = rng.integers(0, 256, (24, 20, 3) if color else (24, 20), dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(a).save(stream, 'PNG')
    stream.seek(0)
    return stream


class CompressionTests(unittest.TestCase):
    def test_native_resolution_and_optional_resize(self):
        pixels = np.tile(np.arange(1200, dtype=np.uint16) % 256, (36, 1)).astype(np.uint8)
        def image_file():
            stream = BytesIO()
            Image.fromarray(pixels).save(stream, 'PNG')
            stream.seek(0)
            return stream
        native = compress(image_file(), rank=5)
        self.assertEqual((native['width'], native['height']), (1200, 36))
        for result in native['results']:
            with Image.open(BytesIO(base64.b64decode(result['preview']))) as preview:
                self.assertEqual(preview.size, (1200, 36))
        smaller = compress(image_file(), rank=5, max_side=512)
        self.assertEqual(smaller['width'], 512)

    def test_matrix_formulas(self):
        A = np.diag([3.0, 2.0]).astype(np.float32)
        Ak, U, s, Vt, info = decompose_channel(A, 1)
        np.testing.assert_allclose(Ak, np.diag([3.0, 0.0]), atol=1e-6)
        self.assertAlmostEqual(info['discarded_squared_error'], 4)
        self.assertAlmostEqual(info['energy'], 100*9/13)
        self.assertEqual(U.shape, (2, 1))
        self.assertEqual(Vt.shape, (1, 2))

    def test_main_entry_point(self):
        import main
        with patch('sys.argv', ['main.py', '--web']), patch.object(main, 'loopback_available', return_value=True), patch.object(main, 'start_application') as start:
            main.main()
            start.assert_called_once()

    def test_blocked_loopback_opens_desktop(self):
        import main
        with patch.object(main, 'loopback_available', return_value=False), patch('desktop.start_desktop') as desktop:
            main.main()
            desktop.assert_called_once()

    def test_large_image_rejected_before_loading(self):
        with patch('backend.Image.MAX_IMAGE_PIXELS', 100):
            with self.assertRaises(Image.DecompressionBombError):
                compress(picture())

    def test_invalid_settings_and_upload_limit(self):
        client = app.test_client()
        response = client.post('/api/compress', data={'image': (picture(), 'face.png'), 'rank': 'invalid'})
        self.assertEqual(response.status_code, 400)
        with patch.dict(app.config, MAX_CONTENT_LENGTH=100):
            response = client.post('/api/compress', data={'image': (picture(), 'face.png')})
            self.assertEqual(response.status_code, 413)
            self.assertIn('error', response.get_json())

    def test_launcher_falls_back_and_closes(self):
        server = MagicMock(server_port=8765)
        server.serve_forever.side_effect = KeyboardInterrupt
        with patch('werkzeug.serving.make_server', side_effect=[OSError('occupied'), server]) as factory:
            with patch('launch.Timer') as timer:
                from launch import start_application
                start_application()
                self.assertEqual(factory.call_count, 2)
                timer.return_value.start.assert_called_once()
                server.serve_forever.assert_called_once()
                server.server_close.assert_called_once()

    def test_full_rank_reconstruction_and_archive(self):
        for color in (False, True):
            d = compress(picture(color), rank=100, grayscale=not color)
            self.assertEqual(d['rank'], 20)
            self.assertEqual(d['results'][0]['metrics']['mse'], 0)
            archive = BytesIO(base64.b64decode(d['results'][0]['download']))
            restored = reconstruct_npz(archive)
            baseline = Image.open(BytesIO(base64.b64decode(d['original'])))
            np.testing.assert_array_equal(np.asarray(restored), np.asarray(baseline))

    def test_rank_improves_error_and_costs_storage(self):
        low = compress(picture(), rank=2)
        high = compress(picture(), rank=10)
        self.assertGreater(low['results'][0]['metrics']['mse'], high['results'][0]['metrics']['mse'])
        self.assertLess(low['decomposition']['factor_bytes'], high['decomposition']['factor_bytes'])
        self.assertEqual(low['decomposition']['factor_values'], 2*(24+20+1))

    def test_zero_image(self):
        f = BytesIO()
        Image.new('L', (12, 12), 0).save(f, 'PNG')
        f.seek(0)
        d = compress(f)
        self.assertEqual(d['decomposition']['channels'][0]['energy'], 100)
        self.assertEqual(d['results'][0]['metrics']['relative_error'], 0)

    def test_routes_and_encoded_outputs(self):
        client = app.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.post('/api/compress').status_code, 400)
        self.assertEqual(client.post('/api/compress', data={'image': (BytesIO(b'bad'), 'bad.png')}).status_code, 400)
        response = client.post('/api/compress', data={'image': (picture(True), 'face.png'), 'mode':'color', 'rank':'5'})
        self.assertEqual(response.status_code, 200)
        d = response.get_json()
        self.assertEqual([r['name'] for r in d['results']], ['SVD'])
        for result in d['results']:
            payload = base64.b64decode(result['download'])
            self.assertEqual(len(payload), result['bytes'])
            if result['name'] != 'SVD':
                with Image.open(BytesIO(payload)) as image:
                    self.assertEqual(image.size, (20, 24))


if __name__ == '__main__':
    unittest.main()
