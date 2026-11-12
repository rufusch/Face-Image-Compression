"""Simple SVD gallery: upload once, compare five ranks, save the result."""
import base64
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
from backend import compare_ranks
import inspect
import backend

BG, WHITE, INK, MUTED, ACCENT = '#f3f6f7', '#ffffff', '#18323a', '#6b7b83', '#087f74'


class Studio:
    def __init__(self, root):
        self.root = root
        self.input_image = None
        self.image_name = None
        self.data = None
        self.future = None
        self.sample_future = None
        self.faces = None
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.gray = tk.BooleanVar(value=True)
        self.math_window = None
        self.archive_future = None
        self.status = tk.StringVar(value='Choose a picture to compare five levels of SVD compression.')
        self.cards = []
        root.title('FaceLab | SVD comparison · ranks 100 / 50 / 25 / 10 / 5')
        root.geometry(f'{min(1800, root.winfo_screenwidth()-40)}x{min(1100, root.winfo_screenheight()-100)}')
        root.minsize(960, 720)
        if root.tk.call('tk', 'windowingsystem') == 'win32':
            root.state('zoomed')
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TProgressbar', background=ACCENT, troughcolor='#e1e9e9')
        menu = tk.Menu(root)
        self.file_menu = tk.Menu(menu, tearoff=False)
        self.file_menu.add_command(label='Open SVD Archive…', command=self.open_archive)
        menu.add_cascade(label='File', menu=self.file_menu)
        root.configure(menu=menu)
        header = tk.Frame(root, bg=WHITE, padx=24, pady=18)
        header.pack(fill='x')
        tk.Label(header, text='FaceLab', font=('Segoe UI', 23, 'bold'), bg=WHITE, fg=INK).pack(side='left')
        tk.Label(header, text='One image. Five ranks.', font=('Segoe UI', 12), bg=WHITE, fg=MUTED).pack(side='left', padx=22)
        self.save_button = self.button(header, 'Save comparison', self.save_sheet)
        self.save_button.pack(side='right')
        self.button(header, 'SVD formulas & code', self.show_math).pack(side='right', padx=10)
        self.compress_panel = tk.Frame(root, bg=BG)
        self.compress_panel.pack(fill='both', expand=True)
        toolbar = tk.Frame(self.compress_panel, bg=BG, padx=24, pady=16)
        toolbar.pack(fill='x')
        self.face_picker = ttk.Combobox(toolbar, state='readonly', width=22,
                                       values=[f'Olivetti face {i:03d}' for i in range(1, 401)])
        self.face_picker.set('Choose a sample face')
        self.face_picker.pack(side='left', padx=(0, 10))
        self.face_picker.bind('<<ComboboxSelected>>', self.select_face)
        self.open_button = self.button(toolbar, '+  Upload picture', self.choose)
        self.open_button.pack(side='left')
        self.run_button = self.button(toolbar, 'Compare ranks  →', self.begin)
        self.run_button.pack(side='left', padx=10)
        tk.Checkbutton(toolbar, text='Grayscale', variable=self.gray, bg=BG, fg=INK, activebackground=BG, font=('Segoe UI', 11)).pack(side='left', padx=10)
        self.file_label = tk.Label(toolbar, text='Original resolution is preserved', font=('Segoe UI', 10), bg=BG, fg=MUTED)
        self.file_label.pack(side='right')
        self.grid = tk.Frame(self.compress_panel, bg=BG)
        self.grid.pack(fill='both', expand=True, padx=18)
        for i in range(3):
            self.grid.columnconfigure(i, weight=1, uniform='cards')
        for i in range(2):
            self.grid.rowconfigure(i, weight=1, uniform='rows')
        for i, title in enumerate(['Original', 'r = 100', 'r = 50', 'r = 25', 'r = 10', 'r = 5']):
            frame = tk.Frame(self.grid, bg=WHITE, padx=12, pady=10)
            frame.grid(row=i//3, column=i%3, sticky='nsew', padx=6, pady=6)
            top = tk.Frame(frame, bg=WHITE)
            top.pack(fill='x')
            heading = tk.Label(top, text=title, font=('Segoe UI', 13, 'bold'), bg=WHITE, fg=INK)
            heading.pack(side='left')
            tk.Button(top, text='Expand ↗', command=lambda index=i: self.expand_image(index), bg=WHITE, fg=ACCENT, relief='flat', cursor='hand2').pack(side='right')
            heading.bind('<Button-1>', lambda event, index=i: self.expand_image(index))
            frame.bind('<Button-1>', lambda event, index=i: self.expand_image(index))
            if i:
                tk.Button(top, text='Save image', command=lambda j=i-1: self.save_image(j), bg=WHITE, fg=ACCENT, relief='flat', cursor='hand2').pack(side='right')
                tk.Button(top, text='Save factors', command=lambda j=i-1: self.save_factors(j), bg=WHITE, fg=ACCENT, relief='flat', cursor='hand2').pack(side='right')
            image = tk.Label(frame, text='Your picture appears here', bg='#f6f8f9', fg=MUTED, font=('Segoe UI', 11), cursor='hand2')
            image.pack(fill='both', expand=True, pady=8)
            image.bind('<Button-1>', lambda event, index=i: self.expand_image(index))
            caption = tk.Label(frame, text='—', bg=WHITE, fg=MUTED, font=('Segoe UI', 9))
            caption.pack(anchor='w')
            caption.bind('<Button-1>', lambda event, index=i: self.expand_image(index))
            self.cards.append((heading, image, caption))
        self.progress = ttk.Progressbar(self.compress_panel, mode='indeterminate')
        self.progress.pack(fill='x', padx=24, pady=(8, 0))
        tk.Label(self.compress_panel, textvariable=self.status, bg=BG, fg=MUTED, font=('Segoe UI', 10), anchor='w', padx=24, pady=12).pack(fill='x')
        root.bind('<Configure>', self.resize)
        self.resize_job = None
        root.protocol('WM_DELETE_WINDOW', self.close)

    def button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=ACCENT, fg=WHITE, activebackground='#08675f', activeforeground=WHITE, font=('Segoe UI', 10, 'bold'), relief='flat', padx=16, pady=11, cursor='hand2')

    def open_archive(self):
        if self.archive_future is not None:
            return
        path = filedialog.askopenfilename(title='Open SVD Archive',
                                          filetypes=[('SVD factor archive', '*.npz')])
        if not path:
            return
        self.file_menu.entryconfigure(0, state='disabled')
        self.status.set('Opening SVD archive…')
        self.archive_future = self.pool.submit(backend.reconstruct_npz, path)
        self.root.after(120, self.poll_archive, Path(path).stem)

    def poll_archive(self, name):
        if not self.archive_future.done():
            self.root.after(120, self.poll_archive, name)
            return
        try:
            image = self.archive_future.result()
            viewer = ImageViewer(self.root, image, f'SVD archive · {name}')
            ttk.Button(viewer.toolbar, text='Save PNG',
                       command=lambda: self.save_archive_image(image, name)).pack(side='left', padx=3)
            ttk.Button(viewer.toolbar, text='Formulas & code',
                       command=lambda: self.show_math(archive=True)).pack(side='left', padx=3)
            self.status.set(f'Opened {name}.npz')
        except Exception as error:
            self.status.set('Could not open the SVD archive.')
            messagebox.showerror('Cannot open SVD archive', str(error))
        finally:
            self.archive_future = None
            self.file_menu.entryconfigure(0, state='normal')

    def save_archive_image(self, image, name):
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                            initialfile=f'{name}-reconstructed.png',
                                            filetypes=[('PNG image', '*.png')])
        if path:
            try:
                image.save(path, format='PNG')
                self.status.set(f'Saved {Path(path).name}')
            except OSError as error:
                messagebox.showerror('Save failed', str(error))

    def choose(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif'), ('All files', '*.*')])
        if not path:
            return
        try:
            with Image.open(path) as image:
                self.source = ImageOps.exif_transpose(image).convert('RGB')
            self.face_picker.set('Choose a sample face')
            self.set_image(self.source, path, Path(path).name)
            self.run_button.configure(state='normal')
        except Exception as error:
            messagebox.showerror('Cannot open image', str(error))

    def set_image(self, image, source, name):
        self.source = image
        self.input_image, self.image_name, self.data = source, name, None
        for _, widget, caption in self.cards:
            widget.configure(image='', text='Click Compare ranks')
            widget.photo = None
            caption.configure(text='—')
        self.file_label.configure(text=f'{name}  ·  {image.width} × {image.height} px')
        self.show(self.cards[0][1], image)
        self.status.set('Ready. Click Compare ranks to generate all five reconstructions.')

    def select_face(self, event=None):
        index = self.face_picker.current()
        if index < 0:
            return
        if self.faces is None:
            self.face_picker.configure(state='disabled')
            self.run_button.configure(state='disabled')
            self.status.set('Loading the Olivetti dataset. Picture upload is available while it loads.')
            self.sample_future = self.pool.submit(self.load_faces)
            self.root.after(120, self.poll_faces)
            return
        image = Image.fromarray(np.rint(self.faces[index] * 255).astype(np.uint8))
        png = BytesIO()
        image.save(png, 'PNG')
        self.set_image(image, png.getvalue(), self.face_picker.get())

    @staticmethod
    def load_faces():
        from sklearn.datasets import fetch_olivetti_faces
        return fetch_olivetti_faces().images

    def poll_faces(self):
        if not self.sample_future.done():
            self.root.after(120, self.poll_faces)
            return
        busy = self.future is not None and not self.future.done()
        if not busy:
            self.face_picker.configure(state='readonly')
            self.run_button.configure(state='normal')
        try:
            self.faces = self.sample_future.result()
            self.select_face()
        except Exception as error:
            if self.face_picker.current() >= 0:
                self.face_picker.set('Choose a sample face')
                self.status.set('Sample faces unavailable. Select a face to retry or upload a picture.')
                messagebox.showerror('Cannot load Olivetti faces', f'First use requires access to the dataset download.\n\n{error}')
        finally:
            self.sample_future = None

    def begin(self):
        if self.input_image is None:
            self.choose()
        if self.input_image is None:
            return
        self.open_button.configure(state='disabled')
        self.run_button.configure(state='disabled')
        self.face_picker.configure(state='disabled')
        self.progress.start(12)
        self.status.set('Computing SVD once and reconstructing ranks 100, 50, 25, 10, 5… Large images may take longer.')
        self.future = self.pool.submit(self.worker, self.input_image, self.gray.get())
        self.root.after(120, self.poll)

    @staticmethod
    def worker(source, grayscale):
        with (BytesIO(source) if isinstance(source, bytes) else open(source, 'rb')) as file:
            return compare_ranks(file, grayscale=grayscale)

    def poll(self):
        if not self.future.done():
            self.root.after(120, self.poll)
            return
        self.progress.stop()
        self.open_button.configure(state='normal')
        self.run_button.configure(state='normal')
        if self.sample_future is None:
            self.face_picker.configure(state='readonly')
        try:
            self.data = self.future.result()
            self.refresh()
            self.status.set('Click any image to expand and zoom. Open SVD formulas & code to see the calculations.')
        except Exception as error:
            self.status.set('Could not process the image.')
            messagebox.showerror('Compression failed', str(error))

    def show(self, widget, image):
        image = ImageOps.contain(image, (max(100, widget.winfo_width()-8), max(80, widget.winfo_height()-8)), Image.Resampling.LANCZOS)
        widget.photo = ImageTk.PhotoImage(image)
        widget.configure(image=widget.photo, text='')

    def refresh(self):
        if not self.data:
            if hasattr(self, 'source'):
                self.show(self.cards[0][1], self.source)
            return
        entries = [{'preview': self.data['original']}] + self.data['comparisons']
        for index, (entry, (heading, widget, caption)) in enumerate(zip(entries, self.cards)):
            self.show(widget, Image.open(BytesIO(base64.b64decode(entry['preview']))))
            if index:
                rank = entry['rank']
                heading.configure(text=f'r = {rank}' + (f" (requested {entry['requested_rank']})" if rank != entry['requested_rank'] else ''))
                score = entry['metrics']['psnr']
                caption.configure(text=f"{'Exact match' if score is None else f'PSNR {score:.1f} dB'}  ·  SVD factors {entry['bytes']/1024:.1f} KB")
            else:
                caption.configure(text=f"{self.data['width']} × {self.data['height']} px  ·  Reference image")

    def resize(self, event):
        if event.widget == self.root:
            if self.resize_job:
                self.root.after_cancel(self.resize_job)
            self.resize_job = self.root.after(180, self.refresh)

    def expand_image(self, index):
        if self.data:
            entry = {'preview': self.data['original']} if index == 0 else self.data['comparisons'][index-1]
            image = Image.open(BytesIO(base64.b64decode(entry['preview']))).copy()
            title = 'Original' if index == 0 else f"r = {entry['rank']}"
        elif index == 0 and hasattr(self, 'source'):
            image, title = self.source.copy(), 'Original'
        else:
            return None
        viewer = ImageViewer(self.root, image, title)
        self.viewer = viewer
        return viewer

    def show_math(self, archive=False):
        if self.math_window is not None and self.math_window.winfo_exists():
            self.math_window.destroy()
        win = tk.Toplevel(self.root)
        self.math_window = win
        mode = 'SVD archive' if archive else 'Compress'
        win.title(f'{mode} | SVD formulas & code')
        win.geometry('1000x760')
        tabs = ttk.Notebook(win)
        tabs.pack(fill='both', expand=True, padx=12, pady=12)
        def page(title, content):
            frame = ttk.Frame(tabs)
            tabs.add(frame, text=title)
            text = tk.Text(frame, wrap='none', font=('Consolas', 11), bg='#142830', fg='#e0efea', padx=20, pady=20)
            vertical = ttk.Scrollbar(frame, command=text.yview)
            horizontal = ttk.Scrollbar(frame, orient='horizontal', command=text.xview)
            vertical.pack(side='right', fill='y')
            horizontal.pack(side='bottom', fill='x')
            text.pack(fill='both', expand=True)
            text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
            text.insert('end', content)
            text.configure(state='disabled')

        if archive:
            formulas = '''RECONSTRUCT FROM SAVED SVD FACTORS
Read U_r, s_r and Vt_r from the selected .npz archive.

RECONSTRUCT EACH CHANNEL
U_r has shape m × r, s_r has r entries, and Vt_r has shape r × n.
A_r = U_r @ diag(s_r) @ Vt_r
    = (U_r * s_r) @ Vt_r

RETURN TO IMAGE PIXELS
pixels = clip(round(A_r), 0, 255)
For RGB, stack the restored red, green, and blue channels.
The output has the same dimensions as the saved reconstruction.

EXACT PIXEL RECOVERY
reference = rounded_rank_r_pixels + saved_residual
New archives store the signed difference between the reference and
the rank-r pixels. Older factor-only archives open at their saved rank.
'''
            page('Formulas', formulas)
            page('Backend code', '# backend.py\n\n' + inspect.getsource(backend.reconstruct_npz))
            return win
        formulas = '''IMAGE AS A MATRIX
A[i,j] is pixel brightness (0–255). A has m rows and n columns.
Grayscale uses one matrix; color uses separate R, G and B matrices.

SINGULAR VALUE DECOMPOSITION
A = U Σ Vᵀ
p = min(m,n)
U: m × p     Σ: p × p diagonal     Vᵀ: p × n
Σ = diag(σ₁, σ₂, ..., σₚ), with σ₁ ≥ σ₂ ≥ ... ≥ 0.

KEEP ONLY r COMPONENTS
Aᵣ = Uᵣ Σᵣ Vᵣᵀ = sum(i=1..r) σᵢ uᵢ vᵢᵀ
Uᵣ: m × r     Σᵣ: r × r     Vᵣᵀ: r × n
The gallery uses r = 100, 50, 25, 10, 5.
For small images, r cannot exceed min(m,n).

RECONSTRUCTION IN PYTHON
A_r = (U[:, :r] * s[:r]) @ Vt[:r, :]
Multiplication by s[:r] scales each column of U; it equals U_r @ diag(s_r).
Round and clip A_r to [0,255] only when converting it back to pixels.

ERROR AND RETAINED ENERGY (before rounding/clipping)
||A - Aᵣ||²_F = sum(σᵢ² for i > r)
Retained energy = 100 × sum(first r σᵢ²) / sum(all σᵢ²)
Aᵣ minimizes Frobenius error among matrices with rank at most r.

STORAGE (per channel)
Original: m × n pixel values.
SVD factors: mr + r + rn = r(m+n+1) values.
Original uint8 pixels use 1 byte each; float32 factors use 4 bytes each.
The displayed factor file size also includes NPZ/ZIP encoding.
Smaller rank does not always mean a smaller file than PNG or JPEG.

EIGENVALUE CONNECTION
AᵀA vᵢ = σᵢ² vᵢ. Thus λᵢ = σᵢ².
Direct SVD avoids explicitly forming AᵀA.
'''
        if self.data:
            m, n = self.data['height'], self.data['width']
            formulas += f'\nYOUR IMAGE\nA: {m} × {n}; channels: {self.data["channels"]}\n'
            for item in self.data['comparisons']:
                r = item['rank']
                formulas += f'r={r}: U_r {m}×{r}, Σ_r {r}×{r}, V_rᵀ {r}×{n}\n'
        page('Formulas', formulas)
        # Read source dynamically so this panel stays consistent with backend.py.
        source = '# backend.py — actual running implementation\n\n'
        source += inspect.getsource(backend.decompose_channel)
        source += '\n\n'+inspect.getsource(backend.compare_ranks)
        source += '\n\n'+inspect.getsource(backend.compress)
        page('Backend code', source)
        return win

    def save_image(self, index):
        if not self.data:
            return
        entry = self.data['comparisons'][index]
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=f"{Path(self.image_name).stem}-r{entry['rank']}.png")
        if path:
            try:
                Path(path).write_bytes(base64.b64decode(entry['preview']))
                self.status.set('Saved full-resolution reconstructed image.')
            except OSError as error:
                messagebox.showerror('Save failed', str(error))

    def save_factors(self, index):
        if not self.data:
            return
        entry = self.data['comparisons'][index]
        path = filedialog.asksaveasfilename(defaultextension='.npz',
                                            initialfile=f"{Path(self.image_name).stem}-r{entry['rank']}.npz",
                                            filetypes=[('SVD factor archive', '*.npz')])
        if path:
            try:
                factors = base64.b64decode(entry['download'])
                with Image.open(BytesIO(base64.b64decode(self.data['original']))) as reference:
                    archive = backend.exact_recovery_archive(factors, reference)
                Path(path).write_bytes(archive)
                self.status.set(f'Saved {Path(path).name} with exact pixel recovery ({len(archive)/1024:.1f} KB).')
            except (OSError, ValueError) as error:
                messagebox.showerror('Save failed', str(error))

    def save_sheet(self):
        if not self.data:
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile='svd-comparison.png')
        if not path:
            return
        try:
            sheet = Image.new('RGB', (1500, 1040), BG)
            draw = ImageDraw.Draw(sheet)
            try:
                font = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 28)
                small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 20)
            except OSError:
                font = small = ImageFont.load_default()
            entries = [('Original', self.data['original'])] + [(f"r = {r['rank']}", r['preview']) for r in self.data['comparisons']]
            for i, (title, data) in enumerate(entries):
                x, y = (i%3)*500, (i//3)*520
                image = Image.open(BytesIO(base64.b64decode(data))).convert('RGB')
                image = ImageOps.contain(image, (460, 435), Image.Resampling.LANCZOS)
                sheet.paste(image, (x+20+(460-image.width)//2, y+65+(435-image.height)//2))
                draw.text((x+24, y+18), title, fill=INK, font=font)
            sheet.save(path)
            self.status.set('Comparison sheet saved. Individual Save image buttons export original-resolution reconstructions.')
        except Exception as error:
            messagebox.showerror('Save failed', str(error))

    def close(self):
        if self.future and not self.future.done():
            if not messagebox.askyesno('Still processing', 'Close the window? The current calculation will finish in the background.'):
                return
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


class ImageViewer:
    """Independent zoom/pan view; the source and saved pixels stay unchanged."""
    def __init__(self, parent, image, title):
        self.image = image.copy()
        self.scale = 1.0
        self.fit_enabled = True
        self.win = tk.Toplevel(parent)
        self.win.title(f'{title} | {image.width} × {image.height} px')
        self.win.geometry('1050x780')
        self.win.transient(parent)
        self.win.lift()
        self.win.focus_force()
        toolbar = tk.Frame(self.win, bg=WHITE, padx=12, pady=10)
        self.toolbar = toolbar
        toolbar.pack(fill='x')
        for label, command in [('−', lambda: self.zoom(1/1.25)), ('+', lambda: self.zoom(1.25)), ('100% · actual pixels', lambda: self.render(1)), ('Fit to window', self.fit)]:
            tk.Button(toolbar, text=label, command=command, bg=WHITE, fg=ACCENT, relief='flat', padx=12, pady=6, cursor='hand2').pack(side='left', padx=3)
        self.info = tk.Label(toolbar, bg=WHITE, fg=MUTED)
        self.info.pack(side='right')
        body = tk.Frame(self.win)
        body.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(body, bg='#1c2e35', highlightthickness=0)
        xbar = ttk.Scrollbar(body, orient='horizontal', command=self.canvas.xview)
        ybar = ttk.Scrollbar(body, orient='vertical', command=self.canvas.yview)
        xbar.pack(side='bottom', fill='x'); ybar.pack(side='right', fill='y')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.bind('<Configure>', self.fit_on_resize)
        self.canvas.bind('<ButtonPress-1>', lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind('<B1-Motion>', lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind('<MouseWheel>', lambda e: self.zoom(1.25 if e.delta > 0 else 1/1.25))
        self.win.bind('<Escape>', lambda e: self.win.destroy())
        tk.Label(self.win, text='Drag to pan · Mouse wheel or +/− to zoom · 100% shows actual pixels · Esc to close', bg=WHITE, fg=MUTED, pady=8).pack(fill='x')
        self.win.after_idle(self.fit)

    def fit(self):
        self.fit_enabled = True
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width > 1 and height > 1:
            self.render(min(width/self.image.width, height/self.image.height))
            self.fit_enabled = True
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)

    def fit_on_resize(self, event):
        if self.fit_enabled:
            self.fit()

    def zoom(self, factor):
        self.render(self.scale*factor)

    def render(self, scale):
        self.fit_enabled = False
        # Bound the display allocation; this does not alter the source image.
        limit = min(8.0, (32_000_000/(self.image.width*self.image.height))**0.5)
        self.scale = max(min(0.05, limit), min(scale, limit))
        size = (max(1, round(self.image.width*self.scale)), max(1, round(self.image.height*self.scale)))
        resized = self.image.resize(size, Image.Resampling.NEAREST if self.scale >= 1 else Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete('all')
        width = max(size[0], self.canvas.winfo_width())
        height = max(size[1], self.canvas.winfo_height())
        self.canvas.create_image((width-size[0])//2, (height-size[1])//2, image=self.photo, anchor='nw')
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.info.configure(text=f'{self.scale*100:.0f}%  ·  {self.image.width} × {self.image.height} px')


def start_desktop():
    root = tk.Tk()
    Studio(root)
    root.mainloop()
