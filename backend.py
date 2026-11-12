"""FACELAB: MATRIX DECOMPOSITION AND COMPRESSION

MATRIX NOTATION
    A[i,j] = pixel brightness at row i, column j (0 to 255).
    A has m rows and n columns. r = min(m,n). Keep 1 <= k <= r.
    Grayscale: one matrix A. Color: separate matrices A_R, A_G, A_B.

FORMULA 1 -- Reduced SVD
    A = U @ Sigma @ Vt
    U: m x r; Sigma: r x r diagonal; Vt: r x n.
    Sigma = diag(s), where s contains singular values in descending order.
    U.T @ U = I and Vt @ Vt.T = I.

FORMULA 2 -- Eigenvalues and eigenvectors
    (A.T @ A) @ v_i = sigma_i**2 * v_i
    (A @ A.T) @ u_i = sigma_i**2 * u_i
    Thus lambda_i = sigma_i**2; rows of Vt are v_i.T.
    Direct SVD avoids explicitly forming A.T @ A for numerical stability.

FORMULA 3 -- Rank-k compression and reconstruction
    A_k = U_k @ diag(s_k) @ Vt_k
        = sum(sigma_i * outer(u_i,v_i), i=1,...,k)
    U_k: m x k; s_k: k values; Vt_k: k x n.
    This minimizes Frobenius error among matrices of rank at most k,
    before rounding and clipping the reconstructed pixels to [0,255].

FORMULA 4 -- Energy and theoretical error
    energy_percent = 100 * sum(s[:k]**2) / sum(s**2)
    ||A - A_k||_F**2 = sum(s[k:]**2)
    An all-zero matrix uses 100 percent retained energy by convention.

FORMULA 5 -- Storage for C channels
    original_values = C*m*n
    factor_values = C*k*(m+n+1)
    scalar_ratio = original_values / factor_values
    Original uint8 values use 1 byte; float32 factors use 4 bytes each.
    NPZ downloads additionally ZIP-compress the stored factors.

FORMULA 6 -- Quality of decoded pixels B compared to reference A
    MSE = sum((A-B)**2) / number_of_values
    PSNR = 10*log10(255**2/MSE); infinite for an exact match
    relative_error = ||A-B||_F / ||A||_F
    savings_percent = 100*(1-download_bytes/baseline_PNG_bytes)

Rank optimization -- optimize_rank(): E_hat(x)=a exp(-b*x), J=E_hat+lambda*R(x).
    R(x)=4*x*(m+n+1)/(m*n). J'=-ab exp(-bx)+lambda*R'.
    J''=ab^2 exp(-bx)>0. Check the domain, neighbors, and measured ranks.
Area integration -- area_resize(): output pixel = integral_R f(x,y) dxdy / area(R).
    Uses exact source-cell overlap weights for piecewise-constant pixels.
Noise smoothing -- heat_denoise(): u_t=alpha*(u_xx+u_yy).
    Explicit finite differences, zero-flux boundaries, alpha*dt<=1/4.

Read decompose_channel() for the mathematical steps, and compress() for
image preparation, per-channel SVD compression, and storage accounting.
"""
from io import BytesIO
import base64
import numpy as np
from PIL import Image, ImageOps
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit

MAX_SIDE = 512
Image.MAX_IMAGE_PIXELS = 20_000_000


def area_weights(source_length, target_length):
    """Exact overlap integrals for a piecewise-constant pixel field.

    W[j,i] = length(source pixel i intersect output footprint j) / footprint width.
    Each row sums to 1. Two-dimensional integration is separable: B = Wy A Wx.T.
    """
    scale = source_length / target_length
    rows, columns, values = [], [], []
    for j in range(target_length):
        left, right = j*scale, (j+1)*scale
        for i in range(int(np.floor(left)), min(source_length, int(np.ceil(right)))):
            overlap = max(0.0, min(right, i+1)-max(left, i))
            if overlap:
                rows.append(j); columns.append(i); values.append(overlap/scale)
    return csr_matrix((values, (rows, columns)), shape=(target_length, source_length))


def area_resize(image, max_side):
    """I_R = (1/area(R)) double_integral_R f(x,y) dx dy.

    Original resolution is preserved unless a smaller maximum side is selected.
    Integral is exact for constant intensity within each source pixel.
    """
    w, h = image.size
    ratio = min(1.0, int(max_side)/max(w, h))
    if max_side < 1:
        raise ValueError('Resolution must be positive.')
    nw, nh = max(1, int(w*ratio)), max(1, int(h*ratio))
    wx, wy = area_weights(w, nw), area_weights(h, nh)
    A = np.asarray(image, dtype=np.float64)
    planes = [A] if A.ndim == 2 else [A[:, :, c] for c in range(3)]
    resized = [np.asarray(wx.dot(wy.dot(p).T).T) for p in planes]
    B = resized[0] if A.ndim == 2 else np.stack(resized, axis=2)
    # A worked output pixel from the first (or grayscale) channel.
    j, i = nh//2, nw//2
    yw, xw = wy.getrow(j), wx.getrow(i)
    patch = planes[0][np.ix_(yw.indices, xw.indices)]
    weights = yw.data[:, None] * xw.data[None, :]
    explanation = {'enabled': True, 'formula': 'I_R = integral_R f(x,y) dx dy / area(R)',
                   'source_size': [w, h], 'output_size': [nw, nh],
                   'footprint_area': (w/nw)*(h/nh), 'output_pixel_xy': [i, j],
                   'source_rows': yw.indices[:8].tolist(), 'source_columns': xw.indices[:8].tolist(),
                   'source_patch_preview': patch[:8, :8].tolist(),
                   'normalized_weights_preview': weights[:8, :8].tolist(),
                   'weight_sum': float(weights.sum()), 'pixel_before_rounding': float(resized[0][j, i]),
                   'source_mean': float(A.mean()), 'integrated_output_mean': float(B.mean()),
                   'note': 'Patch previews show at most 8 rows/columns; calculation uses the entire footprint.'}
    return Image.fromarray(np.clip(np.rint(B), 0, 255).astype(np.uint8)), explanation


def heat_denoise(A, steps=10, alpha=1.0, dt=0.2):
    """u_t = alpha*(u_xx + u_yy); dx = dy = 1 pixel.

    Explicit update: u_new = u + mu*(north+south+east+west-4*u).
    mu = alpha*dt <= 1/4 is required for this two-dimensional scheme.
    Edge replication imposes zero outward flux (Neumann boundary condition).
    Channels diffuse independently; calculations use float32.
    """
    if not 0 <= int(steps) <= 200 or alpha <= 0 or dt <= 0 or alpha*dt > 0.25:
        raise ValueError('Heat solver needs 0–200 steps and 0 < alpha*dt <= 0.25.')
    u = np.asarray(A, dtype=np.float32).copy()
    history = [{'step': 0, 'mean': float(u.mean()), 'variance': float(u.var())}]
    mu = alpha*dt
    padding = ((1, 1), (1, 1)) + (((0, 0),) if u.ndim == 3 else ())
    for step in range(int(steps)):
        padded = np.pad(u, padding, mode='edge')
        u += mu*(padded[:-2, 1:-1]+padded[2:, 1:-1]+padded[1:-1, :-2]+padded[1:-1, 2:]-4*u)
        history.append({'step': step+1, 'mean': float(u.mean()), 'variance': float(u.var())})
    return u, {'formula': 'u_t = alpha*(u_xx + u_yy)', 'alpha': alpha, 'dt': dt,
               'dx': 1, 'dy': 1, 'mu': mu, 'stability_limit': 0.25,
               'steps': int(steps), 'time': int(steps)*dt, 'boundary': 'zero-flux Neumann',
               'history': history}


def optimize_rank(spectra, m, n, storage_weight=0.1):
    """Derivative-based continuous model plus exact discrete validation.

    E(k) = sum_channels sum_{i>k} sigma_i^2 / sum_channels sum_i sigma_i^2.
    R(k) = 4*k*(m+n+1)/(m*n), float32 factor bytes / uint8 pixel bytes.
    Fit E_hat(x)=a*exp(-b*x); J_hat(x)=E_hat(x)+lambda*R(x).
    J_hat'(x)=-a*b*exp(-b*x)+lambda*4*(m+n+1)/(m*n).
    J_hat''(x)=a*b^2*exp(-b*x)>0 when a,b>0.
    Critical point x*=log(a*b/(lambda*R'))/b, restricted to [1,r].
    Final recommendation minimizes measured J(k) across ALL integer ranks;
    the continuous model is illustrative and can be a poor fit.
    """
    weight = float(storage_weight)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Storage weight must be a positive finite number.')
    r = min(m, n)
    squares = np.sum([np.asarray(s, dtype=float)**2 for s in spectra], axis=0)
    total = float(squares.sum())
    errors = np.maximum(0, (total-np.cumsum(squares))/total) if total else np.zeros(r)
    ranks = np.arange(1, r+1, dtype=float)
    slope = 4*(m+n+1)/(m*n)
    costs = errors+weight*slope*ranks
    result = {'formula': 'J(x) = a exp(-b x) + lambda * 4x(m+n+1)/(mn)',
              'storage_weight': weight, 'ranks': ranks.astype(int).tolist(),
              'errors': errors.tolist(), 'costs': costs.tolist(),
              'storage_ratios': (slope*ranks).tolist(), 'recommended_rank': int(np.argmin(costs)+1),
              'model_available': False, 'critical_point': None,
              'note': 'Costs use theoretical float32 factor bytes, not ZIP file size. Error is measured against the SVD input before rounding.'}
    if r < 3 or errors.max() < 1e-12:
        result['model_note'] = 'No meaningful exponential fit: matrix is zero/very low rank or has too few samples.'
        return result
    try:
        def model(x, a, b):
            return a*np.exp(-b*x)
        params, _ = curve_fit(model, ranks, errors, p0=[min(1.0, errors[0]*1.1), 2/r],
                              bounds=([1e-12, 1e-12], [10, 100]), maxfev=10000)
        a, b = map(float, params)
        fitted = model(ranks, a, b)
        variance = float(np.sum((errors-errors.mean())**2))
        fit_r2 = 1-float(np.sum((errors-fitted)**2))/variance if variance else None
        critical = float(np.log(a*b/(weight*slope))/b)
        constrained = float(np.clip(critical, 1, r))
        candidates = sorted(set([1, r, int(np.floor(constrained)), int(np.ceil(constrained))]))
        result.update(model_available=True, a=a, b=b, fit_r2=fit_r2,
                      fitted_errors=fitted.tolist(), fitted_costs=(fitted+weight*slope*ranks).tolist(),
                      critical_point=critical, constrained_minimum=constrained,
                      second_derivative=float(a*b*b*np.exp(-b*constrained)),
                      candidates=[{'rank': k, 'measured_cost': float(costs[k-1])} for k in candidates],
                      model_note='Fit is weak; use the measured recommendation.' if fit_r2 is None or fit_r2 < 0.8 else 'Fit describes this error curve reasonably well.')
    except (ValueError, RuntimeError, FloatingPointError):
        result['model_note'] = 'Exponential fit failed; exact integer-rank comparison is still available.'
    return result


def encode(image, fmt, **options):
    output = BytesIO()
    image.save(output, format=fmt, **options)
    return output.getvalue()


def b64(data):
    return base64.b64encode(data).decode('ascii')


def metrics(reference, result):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(result.convert(reference.mode), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    norm = float(np.linalg.norm(a.ravel()))
    return {'mse': mse, 'psnr': None if mse == 0 else float(10 * np.log10(255**2 / mse)),
            'relative_error': float(np.linalg.norm((a-b).ravel()) / norm) if norm else (0.0 if mse == 0 else None)}


def decompose_channel(A, k, channel_name='Gray', decomposition=None):
    """Return A_k, U_k, s_k, Vt_k, and explanatory data for one 2D matrix.

    The saved factors can reconstruct A_k without the original photograph.
    """
    A = np.asarray(A, dtype=np.float32)
    if A.ndim != 2 or min(A.shape) < 1 or not np.isfinite(A).all():
        raise ValueError('A must be a nonempty finite 2D matrix.')
    m, n = A.shape
    k = max(1, min(int(k), m, n))

    # STEP 1: A = U @ Sigma @ Vt, where Sigma = diag(s).
    U, s, Vt = decomposition if decomposition is not None else svd(A, full_matrices=False, check_finite=False, lapack_driver='gesdd')

    # STEP 2: Retain the k largest singular values and their vectors.
    U_k = U[:, :k]       # m x k
    s_k = s[:k]          # k diagonal entries of Sigma_k
    Vt_k = Vt[:k, :]     # k x n

    # STEP 3: A_k = U_k @ diag(s_k) @ Vt_k.
    # Broadcasting scales each column of U_k; no diagonal allocation needed.
    A_k = (U_k * s_k) @ Vt_k

    # STEP 4: E(k) = 100 * sum(first k sigma^2) / sum(all sigma^2).
    squared_values = s.astype(np.float64) ** 2
    cumulative_energy = np.cumsum(squared_values)
    total = float(cumulative_energy[-1])
    curve = cumulative_energy / total * 100 if total else np.full(len(s), 100.0)
    details = {
        'channel': channel_name,
        'singular_values': s.tolist(), 'energy_curve': curve.tolist(),
        'energy': float(curve[k-1]),
        'discarded_squared_error': float(np.sum(squared_values[k:])),
        'A_shape': [m, n], 'U_shape': list(U_k.shape),
        'Sigma_shape': [k, k], 'Vt_shape': list(Vt_k.shape),
        'A_preview': A[:5, :5].tolist(),
        'U_preview': U_k[:5, :5].tolist(), 's_preview': s_k[:5].tolist(),
        'Vt_preview': Vt_k[:5, :5].tolist(),
        'reconstruction_preview': A_k[:5, :5].tolist(),
    }
    return A_k, U_k, s_k, Vt_k, details


def compress(upload, rank=40, grayscale=True, max_side=None, noise_std=0,
             denoise_steps=0, auto_rank=False, storage_weight=0.1, comparison_ranks=None):
    """Decode, normalize, compress, and return JSON-safe results. No disk storage."""
    with Image.open(upload) as source:
        if source.width * source.height > Image.MAX_IMAGE_PIXELS:
            raise Image.DecompressionBombError('Image exceeds 20 million pixels.')
        source.load()
        source = ImageOps.exif_transpose(source)
        original_size = source.size
        # Composite transparency over white so transparent pixels do not turn black.
        rgba = source.convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        background.alpha_composite(rgba)
        image = background.convert('L' if grayscale else 'RGB')
    # Preserve native dimensions by default. Resizing is an explicit user choice.
    integral_info = {'enabled': False, 'note': 'Original resolution selected. Choose a smaller processing size to apply area-average resizing.'}
    if max_side is not None:
        limit = int(max_side)
        if limit < 1:
            raise ValueError('Resolution must be a positive number.')
        image, integral_info = area_resize(image, limit)
    w, h = image.size
    k = max(1, min(int(rank), h, w))
    baseline = encode(image, 'PNG')
    clean_image = image
    noise_std = float(noise_std)
    if not np.isfinite(noise_std) or not 0 <= noise_std <= 100:
        raise ValueError('Noise standard deviation must be between 0 and 100.')
    a = np.asarray(image, dtype=np.float32)
    # Reproducible synthetic-noise experiment; the clean upload remains reference.
    noisy = np.clip(a + np.random.default_rng(42).normal(0, noise_std, a.shape), 0, 255).astype(np.float32) if noise_std else a.copy()
    denoised, heat_info = heat_denoise(noisy, steps=int(denoise_steps))
    as_image = lambda pixels: Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
    noisy_image, image = as_image(noisy), as_image(denoised)
    a = np.asarray(image, dtype=np.float32)
    heat_info.update(noise_std=noise_std, noise_seed=42,
                     noisy_metrics=metrics(clean_image, noisy_image),
                     denoised_metrics=metrics(clean_image, image))
    planes = [a] if a.ndim == 2 else [a[:, :, c] for c in range(3)]
    decompositions = [svd(p, full_matrices=False, check_finite=False, lapack_driver='gesdd') for p in planes]
    optimization = optimize_rank([d[1] for d in decompositions], h, w, storage_weight)
    if auto_rank:
        k = optimization['recommended_rank']
    reconstructed, factors, details = [], {}, []
    for c, plane in enumerate(planes):
        channel_name = 'Gray' if len(planes) == 1 else ['Red', 'Green', 'Blue'][c]
        A_k, uk, sk, vtk, explanation = decompose_channel(plane, k, channel_name, decompositions[c])
        reconstructed.append(A_k)
        factors.update({f'U_{c}': uk, f's_{c}': sk, f'Vt_{c}': vtk})
        details.append(explanation)
    approximation = reconstructed[0] if len(planes) == 1 else np.stack(reconstructed, axis=2)
    result = Image.fromarray(np.clip(np.rint(approximation), 0, 255).astype(np.uint8))
    archive = BytesIO()
    np.savez_compressed(archive, **factors, mode=np.array(image.mode))
    channels = len(planes)
    factor_values = channels*k*(h+w+1)
    raw_bytes = w*h*channels
    svd_bytes = factor_values*4  # float32 factors versus uint8 source pixels
    results = []

    def add(name, output, payload, extension, mime, note):
        results.append({'name': name, 'preview': b64(encode(output, 'PNG')),
                        'download': b64(payload), 'extension': extension, 'mime': mime,
                        'bytes': len(payload), 'savings': (1-len(payload)/len(baseline))*100,
                        'metrics': metrics(clean_image, output),
                        'svd_input_metrics': metrics(image, output), 'note': note})

    add('SVD', result, archive.getvalue(), 'npz', 'application/octet-stream',
        f'Rank {k}; downloadable float32 matrix factors. Preview is reconstructed from those factors.')
    comparisons = []
    # Reuse the SAME SVD factors for every rank; no repeated decomposition.
    for requested in comparison_ranks or ():
        actual = max(1, min(int(requested), h, w))
        images, saved = [], {}
        for c, (U, s, Vt) in enumerate(decompositions):
            Uk, sk, Vtk = U[:, :actual], s[:actual], Vt[:actual, :]
            images.append((Uk*sk) @ Vtk)
            saved.update({f'U_{c}': Uk, f's_{c}': sk, f'Vt_{c}': Vtk})
        pixels = images[0] if channels == 1 else np.stack(images, axis=2)
        output = as_image(pixels)
        stored = BytesIO()
        np.savez_compressed(stored, **saved, mode=np.array(image.mode))
        comparisons.append({'name': f'SVD · r = {actual}', 'rank': actual,
                            'requested_rank': int(requested), 'preview': b64(encode(output, 'PNG')),
                            'download': b64(stored.getvalue()), 'bytes': len(stored.getvalue()),
                            'metrics': metrics(clean_image, output), 'extension': 'npz',
                            'mime': 'application/octet-stream',
                            'savings': (1-len(stored.getvalue())/len(baseline))*100,
                            'note': f'Requested r = {requested}; limited to {actual} by image dimensions.' if actual != requested else f'Rank-{actual} reconstruction.'})
    return {'width': w, 'height': h, 'original_size': list(original_size), 'rank': k,
            'channels': channels, 'baseline_bytes': len(baseline), 'raw_bytes': raw_bytes,
            'original': b64(baseline), 'results': results, 'comparisons': comparisons,
            'analysis': {'rank_optimization': optimization, 'resizing': integral_info, 'denoising': heat_info},
            'stages': {'clean': b64(baseline), 'noisy': b64(encode(noisy_image, 'PNG')),
                       'denoised': b64(encode(image, 'PNG'))},
            'decomposition': {'channels': details, 'factor_values': factor_values,
                              'pixel_values': raw_bytes, 'factor_bytes': svd_bytes,
                              'scalar_ratio': raw_bytes/factor_values,
                              'byte_ratio': raw_bytes/svd_bytes}}


def compare_ranks(upload, grayscale=True):
    """Original-resolution comparison at r=100,50,25,10,5 from one SVD per channel."""
    return compress(upload, rank=100, grayscale=grayscale, comparison_ranks=(100, 50, 25, 10, 5))


def exact_recovery_archive(factor_archive, reference):
    """Add a lossless pixel residual to a rank-r factor archive."""
    predicted = reconstruct_npz(BytesIO(factor_archive))
    if predicted.mode != reference.mode or predicted.size != reference.size:
        raise ValueError('The reference and reconstruction must have matching dimensions and mode.')
    residual = np.asarray(reference, dtype=np.int16) - np.asarray(predicted, dtype=np.int16)
    with np.load(BytesIO(factor_archive), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    output = BytesIO()
    np.savez_compressed(output, **arrays, residual=residual)
    return output.getvalue()


def reconstruct_npz(path):
    """Reconstruct a grayscale or RGB FaceLab SVD archive."""
    with np.load(path, allow_pickle=False) as data:
        mode = str(data['mode'])
        if mode not in ('L', 'RGB'):
            raise ValueError('The archive must contain grayscale or RGB SVD factors.')
        channels = 1 if mode == 'L' else 3
        factors = []
        expected_shape = None
        for channel in range(channels):
            U, s, Vt = (data[f'{key}_{channel}'] for key in ('U', 's', 'Vt'))
            if U.ndim != 2 or s.ndim != 1 or Vt.ndim != 2:
                raise ValueError('The archive contains invalid SVD factor dimensions.')
            height, width, rank = U.shape[0], Vt.shape[1], s.size
            if not (0 < rank <= min(height, width) and U.shape[1] == Vt.shape[0] == rank):
                raise ValueError('The archive contains incompatible SVD factors.')
            if height * width > Image.MAX_IMAGE_PIXELS:
                raise ValueError('The restored image would exceed 20 million pixels.')
            if expected_shape is not None and expected_shape != (height, width):
                raise ValueError('The archive channels have different image dimensions.')
            if any(f.dtype.kind not in 'fiu' or not np.isfinite(f).all() for f in (U, s, Vt)):
                raise ValueError('The archive contains invalid SVD factor values.')
            expected_shape = (height, width)
            factors.append((U, s, Vt))
        planes = [(U * s) @ Vt for U, s, Vt in factors]
        pixels = planes[0] if channels == 1 else np.stack(planes, axis=2)
        predicted = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
        if 'residual' not in data:
            return Image.fromarray(predicted)
        residual = data['residual']
        if residual.shape != predicted.shape or residual.dtype.kind != 'i' or residual.dtype.itemsize != 2:
            raise ValueError('The archive contains an invalid pixel residual.')
        restored = predicted.astype(np.int32) + residual
        if restored.min() < 0 or restored.max() > 255:
            raise ValueError('The archive residual produces invalid pixel values.')
        return Image.fromarray(restored.astype(np.uint8))
