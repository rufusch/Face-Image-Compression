"""Local web server. Compression implementation lives in backend.py."""
from io import BytesIO
from inspect import getsource
from functools import lru_cache
import os
from pathlib import Path
from tempfile import gettempdir
from zipfile import BadZipFile
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from PIL import UnidentifiedImageError, Image
from werkzeug.exceptions import RequestEntityTooLarge
from backend import compare_ranks, compress, decompose_channel, exact_recovery_archive, reconstruct_npz

app = Flask(__name__, static_folder='public/static', static_url_path='/static')
app.config['VERCEL_HOSTED'] = os.environ.get('VERCEL') == '1'
HOSTED_PAYLOAD_LIMIT = 4_000_000
HOSTED_IMAGE_SIDE = 1024
app.config['MAX_CONTENT_LENGTH'] = HOSTED_PAYLOAD_LIMIT if app.config['VERCEL_HOSTED'] else 12 * 1024 * 1024


@app.get('/')
def index():
    return render_template(
        'index.html',
        hosted=app.config['VERCEL_HOSTED'],
        svd_source=getsource(decompose_channel) + '\n\n' + getsource(compare_ranks),
        archive_source=getsource(exact_recovery_archive) + '\n\n' + getsource(reconstruct_npz),
    )


@app.post('/api/compress')
def process():
    file = request.files.get('image')
    if not file or not file.filename:
        return jsonify(error='Choose an image first.'), 400
    try:
        max_side = int(request.form['max_side']) if request.form.get('max_side') else None
        if app.config['VERCEL_HOSTED']:
            with Image.open(file.stream) as source:
                requested_side = min(max(source.size), max_side) if max_side is not None else max(source.size)
            file.stream.seek(0)
            if requested_side > HOSTED_IMAGE_SIDE:
                return jsonify(error='Set Maximum side to 1024 pixels or less in Processing options.'), 413
        result = compress(file.stream, rank=int(request.form.get('rank', 40)),
                                max_side=max_side,
                                noise_std=float(request.form.get('noise_std', 0)),
                                denoise_steps=int(request.form.get('denoise_steps', 0)),
                                auto_rank=request.form.get('auto_rank') == 'on',
                                storage_weight=float(request.form.get('storage_weight', 0.1)),
                                comparison_ranks=(100, 50, 25, 10, 5) if request.form.get('gallery') else None,
                                grayscale=request.form.get('mode', 'gray') == 'gray')
        if app.config['VERCEL_HOSTED'] and request.form.get('gallery'):
            result = {key: result[key] for key in ('width', 'height', 'channels', 'original', 'comparisons', 'analysis')}
        return jsonify(result)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return jsonify(error='Image dimensions are too large. Please resize the image first.'), 400
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error='Could not read the image or settings. Try a PNG, JPEG, or WebP image.'), 400


@lru_cache(maxsize=1)
def olivetti_images():
    from sklearn.datasets import fetch_olivetti_faces
    data_home = Path(gettempdir()) / 'facelab-data' if app.config['VERCEL_HOSTED'] else None
    return fetch_olivetti_faces(data_home=str(data_home) if data_home is not None else None).images


def png_response(image, name):
    output = BytesIO()
    image.save(output, 'PNG')
    output.seek(0)
    return send_file(output, mimetype='image/png', download_name=name)


@app.get('/api/samples/<int:number>')
def sample(number):
    if not 1 <= number <= 400:
        return jsonify(error='Choose a sample from 1 to 400.'), 404
    try:
        pixels = np.rint(olivetti_images()[number-1]*255).astype(np.uint8)
        return png_response(Image.fromarray(pixels), f'olivetti-{number:03d}.png')
    except (OSError, ImportError, ValueError):
        return jsonify(error='Sample faces are unavailable. You can upload your own image.'), 503


@app.post('/api/archives')
def save_archive():
    factors, reference = request.files.get('factors'), request.files.get('reference')
    if factors is None or reference is None:
        return jsonify(error='A reconstruction and its reference are required.'), 400
    try:
        with Image.open(reference.stream) as image:
            data = exact_recovery_archive(factors.read(), image)
        return send_file(BytesIO(data), mimetype='application/octet-stream',
                         as_attachment=True, download_name='facelab.npz')
    except (BadZipFile, KeyError, OSError, TypeError, ValueError, Image.DecompressionBombError):
        return jsonify(error='Could not create an archive from these factors.'), 400


@app.post('/api/archives/open')
def open_archive():
    archive = request.files.get('archive')
    if archive is None:
        return jsonify(error='Choose a FaceLab .npz archive.'), 400
    try:
        return png_response(reconstruct_npz(archive.stream), 'reconstructed.png')
    except (BadZipFile, KeyError, OSError, TypeError, ValueError, Image.DecompressionBombError):
        return jsonify(error='This file could not be read as a FaceLab SVD archive.'), 400


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    if app.config['VERCEL_HOSTED']:
        return jsonify(error='The hosted upload limit is 4 MB including form data. Choose a smaller file.'), 413
    return jsonify(error='Upload must be smaller than 12 MB.'), 413


@app.after_request
def check_hosted_payload(response):
    if (app.config['VERCEL_HOSTED'] and request.path.startswith('/api/')
            and response.content_length is not None and response.content_length > HOSTED_PAYLOAD_LIMIT):
        response.close()
        return jsonify(error='This result is too large for the hosted studio. Try a smaller processing size or use the local app.'), 413
    return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
