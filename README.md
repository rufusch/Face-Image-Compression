# FaceLab — SVD rank comparison

For the web studio, run `python -m flask --app app run --host 127.0.0.1 --port 8765`
and open `http://127.0.0.1:8765/`. The web studio supports Olivetti samples, image
uploads, five-rank comparisons, zoom, PNG export, and exact-recovery SVD archives.

## Deploy to Vercel

Import the Git repository in Vercel and set **Root Directory** to the folder
containing `app.py`, `requirements.txt`, and `vercel.json` (usually
`face-compression`). Use the **Flask** framework preset and leave the build
and output overrides at their defaults. The browser assets are in `public/static`.
The first Olivetti request on a new instance may download the dataset to its
temporary cache. No API keys are required.

The hosted studio accepts request bodies up to 4 MB and processes images with a
maximum side of 1024 pixels. Choose a smaller processing size if the five-rank
response exceeds 4 MB. These app limits allow room under
[Vercel's function payload limit](https://vercel.com/docs/functions/limitations#request-body-size).

Run `python main.py`, or open main.py in VS Code and choose Run Python File.

1. Choose one of 400 Olivetti faces from the dropdown, or upload your own image.
2. Keep Grayscale selected, or turn it off for color.
3. Click **Compare ranks**.

The first sample selection loads the [Olivetti dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html) through scikit-learn. It downloads the dataset if it is missing from the local cache. Uploads are available while the dataset loads. Each sample is 64 × 64, so the rank-100 card displays its actual rank of 64. On Windows, the desktop window opens maximized and scales the previews to fit the gallery.

The screen shows the original and five reconstructions: **r = 100, 50, 25, 10, 5**.
SVD is computed once per image channel and reused at all five ranks.
Image dimensions are preserved. Previews fit the cards; individual Save image
buttons save the full-resolution reconstructed pixels as PNG.
**Save comparison** exports a labeled comparison sheet for presentations.

If an image's smaller dimension is less than a requested rank, the app uses the
maximum available rank and labels the actual and requested values honestly.
For a useful five-rank comparison, choose an image at least 100 pixels on each side.
The backend retains a 20-million-pixel limit; large SVDs need more time and memory.
The displayed file sizes are NPZ SVD-factor sizes, not preview PNG sizes. A lower
rank uses fewer factors; actual encoded sizes depend on the data and encoding.

## Code
- `main.py`: start the application and check dependencies.
- `desktop.py`: simple five-image gallery and save controls.
- `backend.py`: matrix decomposition, reconstruction, and comparison generation.
- `advanced_desktop.py`: previous optional analysis interface, kept separately.
- `app.py`, `launch.py`, `templates/`: optional earlier browser interface.
- `test_*.py`: mathematical, pipeline, and comparison checks.

SVD formula: A = U Sigma Vt; A_r = U[:, :r] diag(s[:r]) Vt[:r, :].
Compression changes the number of matrix components, not the output dimensions.
Resizing, diffusion, and optimization functions remain in backend.py; the simple
comparison screen does not apply them automatically.

Run checks with `python -m unittest -v`.

## Inspect images and mathematics

Use **Save factors** on a reconstruction card to export an archive with that
rank's SVD factors and the lossless pixel difference from the gallery original.
Choose **File → Open SVD Archive** to recover the gallery original and export
it as PNG. Recovery matches the gallery reference in the selected color mode.
The residual adds storage. The save confirmation shows the total archive size.
Older factor-only archives open at their saved rank.

Click any of the six images to open its detail viewer. Use +/− or the mouse
wheel to zoom, drag the image to pan, or use scrollbars. **100%** shows actual
pixels; **Fit to window** fits the preview without modifying image data. Esc
closes the viewer. Zoomed display allocation is bounded; exports retain the
original reconstruction dimensions.

**SVD formulas & code** opens equations, matrix dimensions for the loaded image,
and actual source code read from backend.py. It includes full SVD, rank-r
reconstruction, retained energy, approximation error, storage, and eigenvalues.

Each card has an **Expand ↗** button as well as a clickable thumbnail. The viewer
opens in front of the gallery. Use ranks 10 and 5 to see stronger detail loss.
