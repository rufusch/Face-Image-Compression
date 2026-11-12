"""Modern local image-compression studio. All processing uses backend.py."""
import base64
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
from backend import compress

BG = '#f3f5f7'
WHITE = '#ffffff'
INK = '#172a33'
MUTED = '#6b7b85'
ACCENT = '#087f74'
LINE = '#e1e7eb'


class Studio:
    def __init__(self, root):
        self.root = root
        root.title('FaceLab Studio | Image compression')
        root.geometry('1380x900')
        root.minsize(1120, 780)
        root.configure(bg=BG)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.path = None
        self.data = None
        self.future = None
        self.photos = []
        self.method = 'SVD'
        self.closed = False
        self.mode = tk.StringVar(value='Color')
        self.resolution = tk.StringVar(value='Original resolution')
        self.rank = tk.IntVar(value=40)
        self.auto_rank = tk.BooleanVar(value=False)
        self.storage_weight = tk.DoubleVar(value=0.1)
        self.noise_std = tk.DoubleVar(value=0)
        self.denoise_steps = tk.IntVar(value=0)
        self.status = tk.StringVar(value='Ready when you are. Add an image to get started.')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TCombobox', padding=8, fieldbackground=WHITE, foreground=INK)
        style.configure('TSpinbox', padding=7, fieldbackground=WHITE, foreground=INK)
        style.configure('Studio.Horizontal.TProgressbar', background=ACCENT, troughcolor=LINE, borderwidth=0)
        self.build()
        root.protocol('WM_DELETE_WINDOW', self.close)

    def label(self, parent, text='', size=11, color=INK, bold=False, **kwargs):
        return tk.Label(parent, text=text, font=('Segoe UI', size, 'bold' if bold else 'normal'), fg=color, bg=parent.cget('bg'), **kwargs)

    def button(self, parent, text, command, primary=False, **kwargs):
        return tk.Button(parent, text=text, command=command, font=('Segoe UI', 10, 'bold'),
                         bg=ACCENT if primary else '#edf3f3', fg=WHITE if primary else ACCENT,
                         activebackground='#09685f' if primary else '#dce9e7',
                         activeforeground=WHITE if primary else ACCENT, relief='flat', bd=0,
                         cursor='hand2', padx=15, pady=11, **kwargs)

    def build(self):
        header = tk.Frame(self.root, bg=WHITE, padx=28, pady=16)
        header.pack(fill='x')
        self.label(header, '◒  FaceLab', 21, bold=True).pack(side='left')
        self.label(header, 'IMAGE COMPRESSION STUDIO', 10, MUTED).pack(side='left', padx=24)
        self.label(header, '●  LOCAL & PRIVATE', 9, ACCENT, True).pack(side='right')
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=24, pady=20)
        sidebar_shell = tk.Frame(body, bg=WHITE, width=290)
        sidebar_shell.pack(side='left', fill='y', padx=(0, 20))
        sidebar_shell.pack_propagate(False)
        sidebar_canvas = tk.Canvas(sidebar_shell, bg=WHITE, highlightthickness=0, width=274)
        sidebar_scroll = ttk.Scrollbar(sidebar_shell, command=sidebar_canvas.yview)
        sidebar_scroll.pack(side='right', fill='y')
        sidebar_canvas.pack(side='left', fill='both', expand=True)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar = tk.Frame(sidebar_canvas, bg=WHITE, padx=16, pady=20)
        sidebar_canvas.create_window(0, 0, anchor='nw', window=sidebar, width=274)
        sidebar.bind('<Configure>', lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox('all')))
        self.label(sidebar, 'YOUR WORKSPACE', 9, ACCENT, True).pack(anchor='w')
        self.label(sidebar, 'Make room for more.', 17, bold=True).pack(anchor='w', pady=(8, 4))
        self.label(sidebar, 'Any photograph. The power of SVD.', 10, MUTED, wraplength=240, justify='left').pack(anchor='w')
        self.upload = self.button(sidebar, '+  Add an image', self.choose, True)
        self.upload.pack(fill='x', pady=(22, 8))
        self.file_label = self.label(sidebar, 'PNG, JPEG, WebP, BMP, TIFF, GIF', 9, MUTED, wraplength=240, justify='left')
        self.file_label.pack(anchor='w')
        for title, variable, values in [
            ('IMAGE MODE', self.mode, ['Color', 'Grayscale']),
            ('PROCESSING RESOLUTION', self.resolution, ['Original resolution', 'Fit within 2048 px', 'Fit within 1024 px', 'Fit within 512 px'])]:
            self.label(sidebar, title, 9, MUTED, True).pack(anchor='w', pady=(22, 7))
            ttk.Combobox(sidebar, textvariable=variable, values=values, state='readonly').pack(fill='x')
        self.label(sidebar, 'Original dimensions are preserved by default. Preview fitting does not resize exports.', 9, MUTED, wraplength=238, justify='left').pack(anchor='w', pady=(8, 0))
        for title, variable, low, high, note in [
            ('SVD COMPONENTS', self.rank, 1, 100000, 'Higher rank keeps more image detail.')]:
            self.label(sidebar, title, 9, MUTED, True).pack(anchor='w', pady=(18, 6))
            widget = ttk.Spinbox(sidebar, from_=low, to=high, textvariable=variable, width=10)
            widget.pack(fill='x')
            if title == 'SVD COMPONENTS':
                self.rank_input = widget
            self.label(sidebar, note, 9, MUTED, wraplength=238, justify='left').pack(anchor='w', pady=(4, 0))
        tk.Checkbutton(sidebar, text='Recommend rank automatically', variable=self.auto_rank,
                       bg=WHITE, fg=INK, selectcolor=WHITE, activebackground=WHITE,
                       font=('Segoe UI', 9), anchor='w').pack(fill='x', pady=(15, 5))
        self.button(sidebar, 'Image preparation & optimization', self.processing_settings).pack(fill='x', pady=4)
        self.button(sidebar, 'Load balanced preset', self.demo_settings).pack(fill='x', pady=4)
        self.run = self.button(sidebar, 'Compress image  →', self.begin, True)
        self.run.pack(fill='x', pady=(24, 8))
        self.progress = ttk.Progressbar(sidebar, mode='indeterminate', style='Studio.Horizontal.TProgressbar')
        self.progress.pack(fill='x')
        self.label(sidebar, 'Large original-resolution SVDs can take time. Processing runs in the background.', 9, MUTED, wraplength=238, justify='left').pack(anchor='w', pady=10)

        main = tk.Frame(body, bg=BG)
        main.pack(side='left', fill='both', expand=True)
        title = tk.Frame(main, bg=BG)
        title.pack(fill='x')
        self.label(title, 'Less space. Same story.', 26, bold=True).pack(anchor='w')
        self.subtitle = self.label(title, 'Explore image quality, file size, and the mathematics in between.', 11, MUTED)
        self.subtitle.pack(anchor='w', pady=(5, 18))
        toolbar = tk.Frame(main, bg=BG)
        toolbar.pack(fill='x', pady=(0, 14))
        self.tabs = {}
        for name in ['SVD']:
            button = self.button(toolbar, name, lambda n=name: self.select(n))
            button.pack(side='left', padx=(0, 7))
            self.tabs[name] = button
        self.matrix_button = self.button(toolbar, 'Matrices & formulas ↗', self.matrices)
        self.matrix_button.pack(side='right')
        self.button(toolbar, 'Analysis', self.analysis_panel).pack(side='right', padx=8)
        images = tk.Frame(main, bg=BG)
        images.pack(fill='both', expand=True)
        images.columnconfigure((0, 1), weight=1, uniform='images')
        images.rowconfigure(0, weight=1)
        self.original_card, self.original_view, self.original_meta = self.image_card(images, '01  /  ORIGINAL', 0)
        self.result_card, self.result_view, self.result_meta = self.image_card(images, '02  /  COMPRESSED', 1)
        self.original_view.bind('<Button-1>', lambda e: self.full_view(False))
        self.result_view.bind('<Button-1>', lambda e: self.full_view(True))
        self.stats = tk.Frame(main, bg=BG)
        self.stats.pack(fill='x', pady=16)
        self.stat_labels = []
        for heading in ['FILE SIZE', 'SPACE SAVED', 'IMAGE QUALITY', 'RETAINED ENERGY']:
            frame = tk.Frame(self.stats, bg=WHITE, padx=16, pady=14)
            frame.pack(side='left', fill='both', expand=True, padx=(0, 8))
            self.label(frame, heading, 8, MUTED, True).pack(anchor='w')
            value = self.label(frame, '—', 19, bold=True)
            value.pack(anchor='w', pady=(6, 0))
            self.stat_labels.append(value)
        bottom = tk.Frame(main, bg=WHITE, padx=18, pady=12)
        bottom.pack(fill='x')
        self.save = self.button(bottom, 'Save compressed file', self.save_file, True)
        self.save.pack(side='right')
        self.save_preview = self.button(bottom, 'Save image PNG', lambda: self.save_file(True))
        self.save_preview.pack(side='right', padx=8)
        self.note = self.label(bottom, 'Your results will appear here.', 9, MUTED, wraplength=360, justify='left')
        self.note.pack(side='left', fill='x', expand=True)
        self.label(main, '', 9).pack()
        tk.Label(self.root, textvariable=self.status, bg=WHITE, fg=MUTED, font=('Segoe UI', 10), anchor='w', padx=26, pady=10).pack(fill='x', side='bottom')
        self.select('SVD')
        self.root.bind('<Configure>', self.schedule_resize)
        self.resize_job = None

    def image_card(self, parent, heading, column):
        frame = tk.Frame(parent, bg=WHITE, padx=15, pady=14)
        frame.grid(row=0, column=column, sticky='nsew', padx=(0, 8) if column == 0 else (8, 0))
        self.label(frame, heading, 9, MUTED, True).pack(anchor='w')
        view = tk.Label(frame, text='Add a picture\nto start exploring' if column == 0 else 'Your compressed image\nwill appear here', bg='#f5f8fa', fg=MUTED, font=('Segoe UI', 13), cursor='hand2')
        view.pack(fill='both', expand=True, pady=12)
        meta = self.label(frame, 'Click an image to inspect at full resolution', 9, MUTED, wraplength=350)
        meta.pack(anchor='w')
        return frame, view, meta

    def schedule_resize(self, event):
        if event.widget != self.root:
            return
        if getattr(self, 'resize_job', None):
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(160, self.refresh_images)

    def show_image(self, widget, image):
        fitted = image.copy()
        fitted.thumbnail((max(100, widget.winfo_width()-12), max(100, widget.winfo_height()-12)), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(fitted)
        widget.configure(image=photo, text='')
        widget.photo = photo

    def refresh_images(self):
        if self.closed:
            return
        if self.data:
            self.show_image(self.original_view, Image.open(BytesIO(base64.b64decode(self.data['original']))))
            self.show_image(self.result_view, Image.open(BytesIO(base64.b64decode(self.current()['preview']))))
        elif hasattr(self, 'source_preview'):
            self.show_image(self.original_view, self.source_preview)

    def choose(self):
        path = filedialog.askopenfilename(title='Choose any photograph or image', filetypes=[('Images', '*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif *.ico *.ppm *.pgm'), ('All files', '*.*')])
        if not path:
            return
        try:
            with Image.open(path) as image:
                oriented = ImageOps.exif_transpose(image)
                self.source_preview = oriented.convert('RGB')
                w, h = oriented.size
            self.path = path
            self.data = None
            self.rank_input.configure(to=min(w, h))
            self.file_label.configure(text=f'{Path(path).name}\n{w:,} × {h:,} pixels · {Path(path).stat().st_size/1024/1024:.2f} MB')
            self.original_meta.configure(text=f'{w:,} × {h:,} px · Source image')
            self.result_view.configure(image='', text='Ready to compress\nChoose your settings and press Compress')
            self.result_view.photo = None
            self.status.set('Image loaded. Original resolution is selected by default.')
            self.refresh_images()
        except Exception as error:
            messagebox.showerror('Could not open image', str(error))

    def current(self):
        return next(r for r in self.data['results'] if r['name'] == self.method)

    def select(self, name):
        self.method = name
        for title, button in self.tabs.items():
            button.configure(bg=ACCENT if title == name else WHITE, fg=WHITE if title == name else MUTED)
        if not self.data:
            return
        r = self.current()
        self.refresh_images()
        self.original_meta.configure(text=f"{self.data['width']:,} × {self.data['height']:,} px · Baseline PNG {self.data['baseline_bytes']/1024:.1f} KB")
        self.result_meta.configure(text=f"{self.data['width']:,} × {self.data['height']:,} px · {name} · Click to view 1:1")
        psnr = r['metrics']['psnr']
        energy = sum(c['energy'] for c in self.data['decomposition']['channels'])/self.data['channels']
        values = [f"{r['bytes']/1024:.1f} KB", f"{r['savings']:.1f}%", 'Exact' if psnr is None else f'{psnr:.1f} dB', f'{energy:.1f}%' if name == 'SVD' else '—']
        for label, value in zip(self.stat_labels, values):
            label.configure(text=value)
        self.note.configure(text=r['note']+' Savings compare against the baseline PNG.')
        self.save_preview.configure(state='normal' if name == 'SVD' else 'disabled')

    def begin(self):
        if not self.path:
            self.choose()
        if not self.path:
            return
        try:
            rank = self.rank.get()
            if rank < 1:
                raise ValueError('Rank must be a positive integer.')
            max_side = None if self.resolution.get() == 'Original resolution' else int(self.resolution.get().split()[2])
        except (ValueError, tk.TclError) as error:
            messagebox.showerror('Check your settings', str(error))
            return
        self.run.configure(state='disabled')
        self.upload.configure(state='disabled')
        self.progress.start(12)
        self.status.set('Compressing at selected resolution… The interface stays available while the matrices are computed.')
        try:
            options = dict(rank=rank, grayscale=self.mode.get() == 'Grayscale', max_side=max_side,
                           auto_rank=self.auto_rank.get(), storage_weight=self.storage_weight.get(),
                           noise_std=self.noise_std.get(), denoise_steps=self.denoise_steps.get())
            if not 0 <= options['noise_std'] <= 100 or not 0 <= options['denoise_steps'] <= 200 or options['storage_weight'] <= 0:
                raise ValueError('Noise: 0–100; heat steps: 0–200; storage weight: greater than zero.')
        except (ValueError, tk.TclError) as error:
            self.run.configure(state='normal')
            self.upload.configure(state='normal')
            self.progress.stop()
            messagebox.showerror('Processing settings', str(error))
            return
        self.future = self.pool.submit(self.worker, self.path, options)
        self.root.after(120, self.poll)

    @staticmethod
    def worker(path, options):
        with open(path, 'rb') as file:
            return compress(file, **options)

    def poll(self):
        if self.closed:
            return
        if not self.future.done():
            self.root.after(120, self.poll)
            return
        self.progress.stop()
        self.run.configure(state='normal')
        self.upload.configure(state='normal')
        try:
            self.data = self.future.result()
            self.rank.set(self.data['rank'])
            self.select(self.method)
            self.status.set(f"Done · {self.data['width']:,} × {self.data['height']:,} pixels · Inspect your SVD reconstruction or save your results.")
        except Exception as error:
            self.status.set('Could not complete compression. Adjust settings or choose another image.')
            messagebox.showerror('Compression failed', str(error))

    def save_file(self, preview=False):
        if not self.data:
            return
        r = self.current()
        extension = 'png' if preview else r['extension']
        path = filedialog.asksaveasfilename(defaultextension='.'+extension, initialfile=f'{Path(self.path).stem}-{self.method.lower().replace(" ", "-")}.{extension}')
        if path:
            try:
                Path(path).write_bytes(base64.b64decode(r['preview' if preview else 'download']))
                self.status.set(f'Saved {Path(path).name}')
            except OSError as error:
                messagebox.showerror('Could not save', str(error))

    def full_view(self, compressed):
        if not self.data and (compressed or not hasattr(self, 'source_preview')):
            return
        if self.data:
            image = Image.open(BytesIO(base64.b64decode(self.current()['preview'] if compressed else self.data['original'])))
        else:
            image = self.source_preview
        win = tk.Toplevel(self.root)
        win.title(f'Full resolution · {image.width} × {image.height} pixels')
        win.geometry('1000x750')
        canvas = tk.Canvas(win, bg='#172a33', highlightthickness=0)
        x = ttk.Scrollbar(win, orient='horizontal', command=canvas.xview)
        y = ttk.Scrollbar(win, orient='vertical', command=canvas.yview)
        x.pack(side='bottom', fill='x')
        y.pack(side='right', fill='y')
        canvas.pack(fill='both', expand=True)
        canvas.configure(xscrollcommand=x.set, yscrollcommand=y.set)
        canvas.photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=canvas.photo, anchor='nw')
        canvas.configure(scrollregion=(0, 0, image.width, image.height))

    def matrices(self):
        import backend
        win = tk.Toplevel(self.root)
        win.title('The mathematics | SVD decomposition')
        win.geometry('950x720')
        text = tk.Text(win, bg='#142830', fg='#d7eee8', insertbackground=WHITE, font=('Consolas', 11), padx=24, pady=24, wrap='word')
        scroll = ttk.Scrollbar(win, command=text.yview)
        scroll.pack(side='right', fill='y')
        text.pack(fill='both', expand=True)
        text.configure(yscrollcommand=scroll.set)
        text.insert('end', backend.__doc__)
        if self.data:
            for c in self.data['decomposition']['channels']:
                text.insert('end', '\n\n'+json.dumps({k:v for k,v in c.items() if k not in ('singular_values', 'energy_curve')}, indent=2))
        text.configure(state='disabled')

    def demo_settings(self):
        self.resolution.set('Fit within 512 px')
        self.mode.set('Grayscale')
        self.auto_rank.set(True)
        self.storage_weight.set(0.1)
        self.noise_std.set(15)
        self.denoise_steps.set(3)
        self.status.set('Balanced preset loaded: area resizing, noise σ=15, 3 smoothing steps, automatic rank. Compress, then open Analysis.')

    def processing_settings(self):
        win = tk.Toplevel(self.root)
        win.title('Image preparation & optimization')
        win.geometry('540x510')
        win.configure(bg=WHITE)
        frame = tk.Frame(win, bg=WHITE, padx=24, pady=24)
        frame.pack(fill='both', expand=True)
        self.label(frame, 'Image preparation', 20, bold=True).pack(anchor='w')
        for heading, variable, low, high, step, explanation in [
            ('Storage penalty λ', self.storage_weight, 0.001, 10, 0.01, 'Larger λ favors fewer SVD components. Cost uses float32 storage and normalized squared error.'),
            ('Added noise standard deviation', self.noise_std, 0, 100, 1, '0 preserves the upload. Positive values add reproducible artificial noise to test denoising.'),
            ('Smoothing steps', self.denoise_steps, 0, 200, 1, '0 disables denoising. Each step uses α=1, Δt=0.2, and zero-flux boundaries. Too many steps blur details.')]:
            self.label(frame, heading, 11, bold=True).pack(anchor='w', pady=(18, 5))
            ttk.Spinbox(frame, textvariable=variable, from_=low, to=high, increment=step).pack(fill='x')
            self.label(frame, explanation, 9, MUTED, wraplength=475, justify='left').pack(anchor='w', pady=5)
        self.label(frame, 'For area-average resizing, choose a smaller Processing Resolution in the sidebar. Original resolution skips resizing.', 10, ACCENT, wraplength=475, justify='left').pack(anchor='w', pady=12)
        self.button(frame, 'Done', win.destroy, True).pack(anchor='e')

    def analysis_panel(self):
        if not self.data:
            messagebox.showinfo('Analysis', 'Compress an image first. Use Load balanced preset to activate all three modules.')
            return
        modules = self.data['analysis']
        win = tk.Toplevel(self.root)
        win.title('FaceLab | Image analysis')
        win.geometry('1120x820')
        notebook = ttk.Notebook(win)
        notebook.pack(fill='both', expand=True, padx=12, pady=12)
        def page(title):
            frame = tk.Frame(notebook, bg=WHITE, padx=18, pady=16)
            notebook.add(frame, text=title)
            return frame
        def paragraph(frame, text):
            self.label(frame, text, 10, MUTED, wraplength=1030, justify='left').pack(anchor='w', pady=6)
        def report(frame, text):
            box = tk.Text(frame, height=12, font=('Consolas', 10), bg='#eff5f4', fg=INK, relief='flat', padx=12, pady=12, wrap='word')
            scroll = ttk.Scrollbar(frame, command=box.yview)
            scroll.pack(side='right', fill='y')
            box.configure(yscrollcommand=scroll.set)
            box.pack(fill='both', expand=True)
            box.insert('end', text)
            box.configure(state='disabled')
        # Rank optimization: measured and modeled costs, derivative and actual discrete check.
        frame = page('Rank optimization')
        optimization = modules['rank_optimization']
        self.label(frame, f"Recommended rank: {optimization['recommended_rank']}  |  Applied rank: {self.data['rank']}", 19, bold=True).pack(anchor='w')
        paragraph(frame, 'Error E(k) is the squared discarded singular values / total energy. R(k)=4k(m+n+1)/(mn). Minimize J(k)=E(k)+λR(k). The recommendation checks every integer rank; the derivative applies only to the continuous fitted model.')
        self.plot(frame, [('Measured cost', optimization['costs'], ACCENT)] +
                  ([('Fitted cost', optimization['fitted_costs'], '#d18735')] if optimization['model_available'] else []), optimization['recommended_rank'])
        shown = {k:v for k,v in optimization.items() if k not in ('ranks', 'errors', 'costs', 'storage_ratios', 'fitted_errors', 'fitted_costs')}
        formulas = "E_hat(x)=a exp(-bx)\nJ_hat'(x)=-ab exp(-bx)+λ·4(m+n+1)/(mn)\nJ_hat''(x)=ab² exp(-bx)>0\nx*=ln[ab/(λ·4(m+n+1)/(mn))]/b\nCheck the domain endpoints and neighboring integer ranks.\n\n"
        report(frame, formulas+json.dumps(shown, indent=2))
        # Area integration: a worked fractional pixel-footprint integral.
        frame = page('Area-average resizing')
        self.label(frame, 'Every output pixel is an area integral', 19, bold=True).pack(anchor='w')
        paragraph(frame, 'I_R = (1 / area(R)) ∬_R f(x,y) dx dy. Source pixels are piecewise-constant cells. The backend computes exact overlap weights, including fractional footprints, then applies B = W_y A W_xᵀ. Display values are rounded only after integration.')
        report(frame, json.dumps(modules['resizing'], indent=2))
        # Diffusion: visual progression and measured comparison against the clean image.
        frame = page('Noise & smoothing')
        self.label(frame, 'Follow the image through diffusion', 19, bold=True).pack(anchor='w')
        paragraph(frame, 'u_t=α(u_xx+u_yy). Explicit update: u_new=u+μ(N+S+E+W−4u), μ=αΔt=0.2≤0.25. Boundaries have zero outward flux. All scores below compare with the clean resized reference, including the final SVD result.')
        gallery = tk.Frame(frame, bg=WHITE)
        gallery.pack(fill='x', pady=8)
        gallery.photos = []
        diffusion = modules['denoising']
        for name, encoded, measured in [
            ('Clean reference', self.data['stages']['clean'], None),
            ('Noisy input', self.data['stages']['noisy'], diffusion['noisy_metrics']),
            ('After diffusion', self.data['stages']['denoised'], diffusion['denoised_metrics']),
            ('SVD reconstruction', self.current()['preview'], self.current()['metrics'])]:
            column = tk.Frame(gallery, bg=WHITE)
            column.pack(side='left', fill='both', expand=True)
            image = Image.open(BytesIO(base64.b64decode(encoded)))
            image.thumbnail((235, 200))
            photo = ImageTk.PhotoImage(image)
            gallery.photos.append(photo)
            tk.Label(column, image=photo, bg=WHITE).pack()
            self.label(column, name, 10, bold=True).pack()
            if measured:
                self.label(column, f"MSE {measured['mse']:.3f}", 9, MUTED).pack()
        paragraph(frame, 'Noise is synthetic and reproducible (seed 42). Denoising can improve a noisy image, but excessive diffusion removes real detail. An uploaded noisy photograph has no known clean ground truth; the upload is the reference for this controlled experiment.')
        self.plot(frame, [('Spatial variance', [item['variance'] for item in diffusion['history']], ACCENT)])
        report(frame, json.dumps({k:v for k,v in diffusion.items() if k != 'history'}, indent=2))
        self.button(win, 'Export analysis JSON', self.export_analysis_report, True).pack(anchor='e', padx=12, pady=(0, 12))

    def plot(self, frame, series, marker=None):
        canvas = tk.Canvas(frame, height=205, bg='#f5f8fa', highlightthickness=0)
        canvas.pack(fill='x', pady=8)
        def draw(event):
            canvas.delete('all')
            width = max(200, event.width)
            left, top, right, bottom = 65, 30, width-25, 165
            ymax = max(max(values) for _, values, _ in series) or 1
            count = max(len(values) for _, values, _ in series)
            for fraction in (0, .5, 1):
                y = bottom-fraction*(bottom-top)
                canvas.create_line(left, y, right, y, fill=LINE)
                canvas.create_text(left-6, y, text=f'{fraction*ymax:.3g}', anchor='e', fill=MUTED)
            for index, (name, values, color) in enumerate(series):
                points = []
                for j, value in enumerate(values):
                    points.extend((left+j/max(1, count-1)*(right-left), bottom-value/ymax*(bottom-top)))
                if len(points) >= 4:
                    canvas.create_line(*points, fill=color, width=2)
                else:
                    canvas.create_oval(left-3, points[1]-3, left+3, points[1]+3, fill=color, outline=color)
                canvas.create_text(left+index*200, 14, text=name, anchor='w', fill=color)
            canvas.create_text(left, 188, text='Rank 1' if marker else 'Step 0', anchor='w', fill=MUTED)
            canvas.create_text(right, 188, text=str(count if marker else count-1), anchor='e', fill=MUTED)
            if marker:
                x = left+(marker-1)/max(1, count-1)*(right-left)
                canvas.create_line(x, top, x, bottom, fill=INK, dash=(4, 3))
        canvas.bind('<Configure>', draw)

    def export_analysis_report(self):
        path = filedialog.asksaveasfilename(defaultextension='.json', initialfile='facelab-analysis.json')
        if path:
            try:
                Path(path).write_text(json.dumps(self.data['analysis'], indent=2, allow_nan=False), encoding='utf-8')
                self.status.set('Analysis exported.')
            except (OSError, ValueError) as error:
                messagebox.showerror('Export failed', str(error))

    def close(self):
        self.closed = True
        if self.future and not self.future.done():
            if not messagebox.askyesno('Processing in progress', 'Close the window? The current calculation will finish in the background before the process exits.'):
                self.closed = False
                return
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def start_desktop():
    root = tk.Tk()
    Studio(root)
    root.mainloop()
