import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy import linalg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import warnings, time, threading, csv
from tkinter import filedialog
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
#  DATASET  (mutable — replaced when user uploads CSV)
# ─────────────────────────────────────────────────────────────────
DEFAULT_STOCKS = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
DEFAULT_PRICES = np.array([
    [182.0, 140.0, 375.0, 178.0, 245.0],
    [185.0, 142.0, 378.0, 180.0, 250.0],
    [183.0, 141.0, 376.0, 179.0, 247.0],
    [188.0, 145.0, 382.0, 183.0, 260.0],
    [190.0, 147.0, 385.0, 185.0, 265.0],
    [186.0, 143.0, 379.0, 181.0, 252.0],
    [192.0, 149.0, 388.0, 187.0, 270.0],
    [195.0, 151.0, 391.0, 189.0, 275.0],
    [193.0, 150.0, 389.0, 188.0, 272.0],
    [197.0, 153.0, 394.0, 191.0, 280.0],
], dtype=float)

STOCKS     = DEFAULT_STOCKS[:]
RAW_PRICES = DEFAULT_PRICES.copy()
DAYS       = [f'D{i:02d}' for i in range(1, len(RAW_PRICES)+1)]
SCOLS      = ['#FFD700','#00C853','#FF8C00','#00BFFF','#E040FB',
              '#FF6B9D','#A8FF3E','#FF9500','#00E5FF','#FF5252']

def load_csv(filepath):
    """Parse CSV: first row = header (stock names), remaining rows = daily prices.
    Returns (stocks_list, prices_ndarray) or raises ValueError with message."""
    import csv
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 3:
        raise ValueError('CSV must have a header row + at least 2 data rows.')
    header = [c.strip() for c in rows[0]]
    # first column might be a date/day label — detect if non-numeric
    try:
        float(rows[1][0])
        stock_names = header          # all columns are stocks
        data_rows   = rows[1:]
        col_start   = 0
    except (ValueError, IndexError):
        stock_names = header[1:]      # skip first label column
        data_rows   = rows[1:]
        col_start   = 1
    if not stock_names:
        raise ValueError('No stock columns found in CSV.')
    prices = []
    for i, row in enumerate(data_rows):
        try:
            vals = [float(row[j]) for j in range(col_start, col_start+len(stock_names))]
        except (ValueError, IndexError) as e:
            raise ValueError(f'Row {i+2}: could not parse numeric values — {e}')
        prices.append(vals)
    return stock_names, np.array(prices, dtype=float)

# ─────────────────────────────────────────────────────────────────
#  PIPELINE MATH
# ─────────────────────────────────────────────────────────────────
def s1_normalize(A):
    mean = A.mean(axis=0)
    return A - mean, mean

def s2_lu(A):
    M = A.T @ A
    P, L, U = linalg.lu(M)
    return P, L, U, M

def s3_rank_nullity(A):
    rank = np.linalg.matrix_rank(A)
    return rank, A.shape[1] - rank

def s4_basis(A):
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = np.sum(s > 1e-10)
    return Vt[:rank], rank

def s5_gram_schmidt(basis):
    V = basis.copy().T
    Q = np.zeros_like(V, dtype=float)
    for i in range(V.shape[1]):
        v = V[:, i].copy()
        for j in range(i):
            v -= np.dot(Q[:, j], V[:, i]) * Q[:, j]
        n = np.linalg.norm(v)
        if n > 1e-10:
            Q[:, i] = v / n
    return Q

def s6_projection(A, Q):
    return A @ Q @ Q.T

def s7_least_squares(A):
    days = np.arange(1, A.shape[0]+1).reshape(-1, 1)
    X = np.hstack([np.ones_like(days), days])
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ A
    return coeffs, (np.array([[1, 11]]) @ coeffs).flatten()

def s8_eigenanalysis(A):
    cov = (A.T @ A) / (A.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx], cov

def s9_svd_compress(A, k=2):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    comp = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return comp, (np.sum(s[:k]**2) / np.sum(s**2)) * 100, s

def s10_enhance(A, compressed, mean, _pred):
    return compressed + mean

# ─────────────────────────────────────────────────────────────────
#  BLOOMBERG TERMINAL PALETTE
# ─────────────────────────────────────────────────────────────────
BG         = "#0a0a0a"       # near-black terminal bg
PANEL      = "#111111"       # slightly lighter panels
PANEL2     = "#161616"       # card bg
BORDER     = "#2a2a2a"       # subtle dividers
BORDER2    = "#333333"

GOLD       = "#FFD700"       # primary accent — gold
GOLD_DIM   = "#B8960C"       # muted gold
GOLD_DARK  = "#3a2e00"       # gold tint bg

GREEN      = "#00C853"       # up / positive
GREEN_DIM  = "#00701e"
RED        = "#FF3D3D"       # down / negative
AMBER      = "#FFA500"       # warning / mid

TEXT       = "#E8E8E8"       # primary text
TEXT2      = "#999999"       # secondary text
TEXT3      = "#555555"       # disabled text

STAGES = [
    ("S1",  "NORMALIZE",     "Matrix Repr. · Mean Centering"),
    ("S2",  "LU DECOMP",     "LU Factorization of AᵀA"),
    ("S3",  "RANK/NULLITY",  "Independence Analysis"),
    ("S4",  "BASIS SELECT",  "Basis Extraction"),
    ("S5",  "GRAM-SCHMIDT",  "Orthogonalization"),
    ("S6",  "PROJECTION",    "Subspace Projection · Denoise"),
    ("S7",  "LEAST SQUARES", "Prediction  x̂=(AᵀA)⁻¹Aᵀb"),
    ("S8",  "EIGENANALYSIS", "Dominant Trend Discovery"),
    ("S9",  "SVD COMPRESS",  "Singular Value Decomposition"),
    ("S10", "ENHANCE",       "Final Market Model Output"),
]

TICKER_DATA = {
    'AAPL': ('+1.23%', GREEN), 'GOOG': ('-0.45%', RED),
    'MSFT': ('+0.87%', GREEN), 'AMZN': ('+2.10%', GREEN),
    'TSLA': ('-1.33%', RED),   'SPX':  ('+0.55%', GREEN),
    'VIX':  ('14.32',  AMBER), 'DXY':  ('+0.12%', GREEN),
}

# ─────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────
class BloombergPipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LA-QUANT  ·  Linear Algebra Market Pipeline")
        self.root.configure(bg=BG)
        self.root.geometry("1480x900")
        self.root.minsize(1200, 720)

        self.current_stage = tk.IntVar(value=0)
        self._ticker_offset = 0
        self.results = {}
        self._run_all_stages()
        self._build_ui()
        self._start_ticker()

    # ── computations ──────────────────────────────────────────────
    def _run_all_stages(self):
        A = RAW_PRICES.copy()  # uses current global dataset
        centered, mean   = s1_normalize(A)
        P, L, U, M       = s2_lu(centered)
        rank, nullity    = s3_rank_nullity(centered)
        basis, r         = s4_basis(centered)
        Q                = s5_gram_schmidt(basis)
        projected        = s6_projection(centered, Q)
        coeffs, pred     = s7_least_squares(centered)
        evals, evecs, cov = s8_eigenanalysis(centered)
        k_svd = max(1, min(2, len(STOCKS)-1))
        compressed, var_ret, svals = s9_svd_compress(centered, k=k_svd)
        enhanced         = s10_enhance(A, compressed, mean, pred)
        self.results = dict(
            raw=A, mean=mean, s1=centered,
            s2_P=P, s2_L=L, s2_U=U, s2_M=M,
            s3_rank=rank, s3_null=nullity,
            s4_basis=basis, s4_r=r,
            s5_Q=Q,
            s6_proj=projected,
            s7_c=coeffs, s7_pred=pred,
            s8_ev=evals, s8_vec=evecs, s8_cov=cov,
            s9_comp=compressed, s9_var=var_ret, s9_sv=svals,
            s10_enh=enhanced,
        )

    # ── top-level layout ──────────────────────────────────────────
    def _build_ui(self):
        self._build_topbar()
        self._build_ticker_bar()
        tk.Frame(self.root, bg=GOLD, height=1).pack(fill='x')
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True)
        # Build order: statusbar -> workspace -> sidebar
        # sidebar._select_stage(0) needs status_lbl and stage_tag to exist first
        self._build_statusbar()
        self._build_workspace(body)
        self._build_sidebar(body)

    # ── TOP BAR ───────────────────────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=44)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        # Logo block
        logo_block = tk.Frame(bar, bg=GOLD_DARK, padx=14)
        logo_block.pack(side='left', fill='y')
        tk.Label(logo_block, text="LA", font=("Courier", 16, "bold"),
                 fg=GOLD, bg=GOLD_DARK).pack(side='left')
        tk.Label(logo_block, text="-QUANT", font=("Courier", 10),
                 fg=GOLD_DIM, bg=GOLD_DARK).pack(side='left', pady=2)

        tk.Frame(bar, bg=GOLD, width=2).pack(side='left', fill='y', pady=6)

        tk.Label(bar, text="  LINEAR ALGEBRA MARKET PIPELINE",
                 font=("Courier", 11, "bold"), fg=TEXT, bg=PANEL).pack(side='left', padx=10)

        tk.Label(bar, text="UE24MA241B  ·  PES UNIVERSITY  ·  DEPT. OF CSE",
                 font=("Courier", 9), fg=TEXT3, bg=PANEL).pack(side='right', padx=16)

        # Upload CSV button
        upload_btn = tk.Label(bar, text="  ⬆  LOAD CSV  ",
                              font=("Courier", 9, "bold"),
                              fg=BG, bg=GOLD, cursor='hand2', padx=4)
        upload_btn.pack(side='right', padx=(0, 8), pady=7)
        upload_btn.bind('<Button-1>', lambda e: self._upload_csv())
        upload_btn.bind('<Enter>', lambda e: upload_btn.config(bg=GREEN))
        upload_btn.bind('<Leave>', lambda e: upload_btn.config(bg=GOLD))

        tk.Frame(bar, bg=BORDER, width=1).pack(side='right', fill='y', pady=8)

        # dataset label
        self.dataset_lbl = tk.Label(bar, text='  ◈ DEMO DATA  ',
                                    font=('Courier', 8), fg=AMBER, bg=PANEL)
        self.dataset_lbl.pack(side='right', padx=4)

        tk.Frame(bar, bg=BORDER, width=1).pack(side='right', fill='y', pady=8)

        # live clock
        self.clock_lbl = tk.Label(bar, text="", font=("Courier", 10),
                                  fg=GOLD, bg=PANEL)
        self.clock_lbl.pack(side='right', padx=4)
        self._tick_clock()

    def _tick_clock(self):
        import datetime
        self.clock_lbl.config(text=datetime.datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # ── CSV UPLOAD ────────────────────────────────────────────────
    def _upload_csv(self):
        filepath = filedialog.askopenfilename(
            title='Load Stock Price CSV',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not filepath:
            return
        try:
            stocks, prices = load_csv(filepath)
        except ValueError as e:
            messagebox.showerror('CSV Error', str(e))
            return

        # validate minimum size
        if prices.shape[0] < 3:
            messagebox.showerror('CSV Error', 'Need at least 3 rows (days) of data.')
            return
        if prices.shape[1] < 2:
            messagebox.showerror('CSV Error', 'Need at least 2 stock columns.')
            return

        # cap stocks at 10 for colour palette
        if len(stocks) > 10:
            stocks  = stocks[:10]
            prices  = prices[:, :10]

        # update globals
        global STOCKS, RAW_PRICES, DAYS
        STOCKS     = stocks
        RAW_PRICES = prices
        DAYS       = [f'D{i:02d}' for i in range(1, prices.shape[0]+1)]

        # update sidebar stats and dataset label
        import os
        fname = os.path.basename(filepath)
        self.dataset_lbl.config(
            text=f'  ◈ {fname[:18]}  ',
            fg=GREEN
        )

        # re-run pipeline and refresh current stage
        self._run_all_stages()
        self._refresh_sidebar_stats()
        self._render_stage(self.current_stage.get())
        self._set_status(
            f'CSV LOADED  ·  {prices.shape[0]} days × {len(stocks)} stocks  ·  {fname}'
        )
        msg = (
            f'Stocks : {chr(183).join(stocks)}\n'
            f'Days   : {prices.shape[0]}\n'
            f'Shape  : {prices.shape[0]} x {len(stocks)}\n\n'
            'Pipeline re-run on new data.'
        )
        messagebox.showinfo('Dataset Loaded', f'{fname}\n\n' + msg)

    def _refresh_sidebar_stats(self):
        """Update the stats panel labels in sidebar after data change."""
        r = self.results
        for lbl, val in zip(self._stat_vals,
                            [f'{RAW_PRICES.shape[0]} × {len(STOCKS)}',
                             str(r['s3_rank']),
                             str(r['s3_null']),
                             f"{r['s9_var']:.1f}%"]):
            lbl.config(text=val)

    # ── TICKER BAR ────────────────────────────────────────────────
    def _build_ticker_bar(self):
        self.ticker_bar = tk.Frame(self.root, bg="#0d0d0d", height=22)
        self.ticker_bar.pack(fill='x')
        self.ticker_bar.pack_propagate(False)
        self.ticker_canvas = tk.Canvas(self.ticker_bar, bg="#0d0d0d",
                                       height=22, highlightthickness=0)
        self.ticker_canvas.pack(fill='both', expand=True)
        self._ticker_items = []

    def _start_ticker(self):
        self.root.update_idletasks()
        self._build_ticker_content()
        self._scroll_ticker()

    def _build_ticker_content(self):
        self.ticker_canvas.delete('all')
        self._ticker_items = []
        x = 10
        for sym, (val, col) in TICKER_DATA.items():
            t1 = self.ticker_canvas.create_text(
                x, 11, text=sym, font=("Courier", 8, "bold"),
                fill=GOLD, anchor='w')
            bb = self.ticker_canvas.bbox(t1)
            x = bb[2] + 4
            t2 = self.ticker_canvas.create_text(
                x, 11, text=val, font=("Courier", 8),
                fill=col, anchor='w')
            bb2 = self.ticker_canvas.bbox(t2)
            x = bb2[2] + 24
            self._ticker_items.extend([t1, t2])
        self._ticker_width = x

    def _scroll_ticker(self):
        self._ticker_offset += 1
        w = self.ticker_canvas.winfo_width() or 1400
        for item in self._ticker_items:
            self.ticker_canvas.move(item, -1, 0)
            bb = self.ticker_canvas.bbox(item)
            if bb and bb[2] < 0:
                self.ticker_canvas.move(item, w + self._ticker_width, 0)
        self.root.after(30, self._scroll_ticker)

    # ── SIDEBAR ───────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=PANEL, width=230)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)

        # header
        hdr = tk.Frame(side, bg=GOLD_DARK, pady=8)
        hdr.pack(fill='x')
        tk.Label(hdr, text="▸ PIPELINE STAGES",
                 font=("Courier", 9, "bold"), fg=GOLD,
                 bg=GOLD_DARK).pack(anchor='w', padx=12)

        tk.Frame(side, bg=GOLD, height=1).pack(fill='x')

        self.stage_frames = []
        for i, (code, name, desc) in enumerate(STAGES):
            row = tk.Frame(side, bg=PANEL, cursor='hand2')
            row.pack(fill='x')

            # left accent bar
            accent = tk.Frame(row, width=3, bg=PANEL)
            accent.pack(side='left', fill='y')

            inner = tk.Frame(row, bg=PANEL, pady=7, padx=10)
            inner.pack(side='left', fill='x', expand=True)

            code_lbl = tk.Label(inner, text=code,
                                font=("Courier", 7, "bold"),
                                fg=GOLD_DIM, bg=PANEL, anchor='w')
            code_lbl.pack(fill='x')

            name_lbl = tk.Label(inner, text=name,
                                font=("Courier", 10, "bold"),
                                fg=TEXT, bg=PANEL, anchor='w')
            name_lbl.pack(fill='x')

            desc_lbl = tk.Label(inner, text=desc,
                                font=("Courier", 7),
                                fg=TEXT3, bg=PANEL, anchor='w')
            desc_lbl.pack(fill='x')

            tk.Frame(side, bg=BORDER, height=1).pack(fill='x')

            widgets = (row, accent, inner, code_lbl, name_lbl, desc_lbl)
            self.stage_frames.append(widgets)

            def _bind(w, idx=i):
                w.bind("<Button-1>", lambda e, x=idx: self._select_stage(x))
                for child in w.winfo_children():
                    child.bind("<Button-1>", lambda e, x=idx: self._select_stage(x))
            _bind(row)
            _bind(inner)

            def _hover_in(e, w=row, a=accent, il=inner, c=code_lbl, n=name_lbl, d=desc_lbl, idx=i):
                if self.current_stage.get() != idx:
                    for ww in [w, il]: ww.config(bg=PANEL2)
                    for ww in [c, n, d]: ww.config(bg=PANEL2)
            def _hover_out(e, w=row, a=accent, il=inner, c=code_lbl, n=name_lbl, d=desc_lbl, idx=i):
                if self.current_stage.get() != idx:
                    for ww in [w, il]: ww.config(bg=PANEL)
                    for ww in [c, n, d]: ww.config(bg=PANEL)

            row.bind("<Enter>", _hover_in)
            row.bind("<Leave>", _hover_out)

        # RUN ALL button
        tk.Frame(side, bg=BORDER, height=1).pack(fill='x', pady=(10, 0))
        run_btn = tk.Label(side,
                           text="  ▶▶  EXECUTE FULL PIPELINE",
                           font=("Courier", 9, "bold"),
                           fg=BG, bg=GOLD, pady=10, cursor='hand2')
        run_btn.pack(fill='x', padx=12, pady=10)
        run_btn.bind("<Button-1>", lambda e: self._run_all_and_show())
        run_btn.bind("<Enter>", lambda e: run_btn.config(bg=GREEN))
        run_btn.bind("<Leave>", lambda e: run_btn.config(bg=GOLD))

        # stats panel
        tk.Frame(side, bg=BORDER, height=1).pack(fill='x')
        stats = tk.Frame(side, bg=PANEL, pady=8, padx=12)
        stats.pack(fill='x')
        r = self.results
        self._stat_vals = []   # keep refs for live update
        for label, val, col in [
            ("MATRIX", f"{RAW_PRICES.shape[0]} × {len(STOCKS)}", GOLD),
            ("RANK", str(r['s3_rank']), GREEN),
            ("NULLITY", str(r['s3_null']), AMBER),
            ("SVD VAR%", f"{r['s9_var']:.1f}%", GREEN),
        ]:
            row2 = tk.Frame(stats, bg=PANEL)
            row2.pack(fill='x', pady=1)
            tk.Label(row2, text=label, font=("Courier", 7),
                     fg=TEXT3, bg=PANEL, width=9, anchor='w').pack(side='left')
            vl = tk.Label(row2, text=val, font=("Courier", 9, "bold"),
                          fg=col, bg=PANEL)
            vl.pack(side='left')
            self._stat_vals.append(vl)

        self._select_stage(0)

    # ── WORKSPACE ─────────────────────────────────────────────────
    def _build_workspace(self, parent):
        ws = tk.Frame(parent, bg=BG)
        ws.pack(side='right', fill='both', expand=True, padx=0)

        # stage title bar
        self.title_bar = tk.Frame(ws, bg=PANEL2, height=38)
        self.title_bar.pack(fill='x')
        self.title_bar.pack_propagate(False)

        self.stage_tag = tk.Label(self.title_bar, text="",
                                  font=("Courier", 8, "bold"),
                                  fg=BG, bg=GOLD, padx=8)
        self.stage_tag.pack(side='left', padx=(12, 8), pady=7)

        self.stage_title = tk.Label(self.title_bar, text="",
                                    font=("Courier", 13, "bold"),
                                    fg=TEXT, bg=PANEL2)
        self.stage_title.pack(side='left')

        self.stage_desc = tk.Label(self.title_bar, text="",
                                   font=("Courier", 9),
                                   fg=TEXT3, bg=PANEL2)
        self.stage_desc.pack(side='left', padx=14)

        # content row
        content = tk.Frame(ws, bg=BG)
        content.pack(fill='both', expand=True, padx=10, pady=8)

        # left: data panel
        left = tk.Frame(content, bg=PANEL2, width=290,
                        highlightbackground=BORDER2, highlightthickness=1)
        left.pack(side='left', fill='y', padx=(0, 8))
        left.pack_propagate(False)

        dp_hdr = tk.Frame(left, bg=GOLD_DARK, pady=5)
        dp_hdr.pack(fill='x')
        tk.Label(dp_hdr, text="◈  DATA OUTPUT",
                 font=("Courier", 8, "bold"),
                 fg=GOLD, bg=GOLD_DARK).pack(anchor='w', padx=10)

        tk.Frame(left, bg=GOLD, height=1).pack(fill='x')

        self.data_text = tk.Text(left, bg=PANEL2, fg=TEXT,
                                 font=("Courier", 9),
                                 relief='flat', wrap='word',
                                 padx=12, pady=10,
                                 insertbackground=GOLD,
                                 selectbackground=GOLD_DARK,
                                 spacing1=2, spacing3=2)
        self.data_text.pack(fill='both', expand=True)

        # tag styles
        self.data_text.tag_config('gold',  foreground=GOLD,  font=("Courier", 9, "bold"))
        self.data_text.tag_config('green', foreground=GREEN, font=("Courier", 9))
        self.data_text.tag_config('red',   foreground=RED,   font=("Courier", 9))
        self.data_text.tag_config('dim',   foreground=TEXT3, font=("Courier", 8))
        self.data_text.tag_config('head',  foreground=GOLD,  font=("Courier", 10, "bold"))
        self.data_text.tag_config('amber', foreground=AMBER, font=("Courier", 9))

        # right: chart panel
        self.chart_outer = tk.Frame(content, bg=PANEL2,
                                    highlightbackground=BORDER2, highlightthickness=1)
        self.chart_outer.pack(side='left', fill='both', expand=True)

        self.canvas_widget = None

    # ── STATUS BAR ────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg="#0d0d0d", height=20)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        tk.Frame(bar, bg=GOLD, width=2).pack(side='left', fill='y')
        self.status_lbl = tk.Label(bar,
                                   text="  SYSTEM READY  ·  ALL STAGES LOADED  ·  10×5 MATRIX",
                                   font=("Courier", 7), fg=TEXT3, bg="#0d0d0d")
        self.status_lbl.pack(side='left', padx=8)
        tk.Label(bar, text="PES UNIVERSITY  ·  UE24MA241B  ·  LINEAR ALGEBRA & APPLICATIONS  ",
                 font=("Courier", 7), fg=TEXT3, bg="#0d0d0d").pack(side='right')

    def _set_status(self, msg):
        self.status_lbl.config(text=f"  {msg}")

    # ── STAGE SELECTION ───────────────────────────────────────────
    def _select_stage(self, idx):
        prev = self.current_stage.get()
        self.current_stage.set(idx)

        # reset prev
        row, accent, inner, code_lbl, name_lbl, desc_lbl = self.stage_frames[prev]
        for w in [row, inner]: w.config(bg=PANEL)
        for w in [code_lbl, name_lbl, desc_lbl]: w.config(bg=PANEL)
        accent.config(bg=PANEL)
        name_lbl.config(fg=TEXT)
        code_lbl.config(fg=GOLD_DIM)

        # highlight current
        row, accent, inner, code_lbl, name_lbl, desc_lbl = self.stage_frames[idx]
        for w in [row, inner]: w.config(bg=GOLD_DARK)
        for w in [code_lbl, name_lbl, desc_lbl]: w.config(bg=GOLD_DARK)
        accent.config(bg=GOLD)
        name_lbl.config(fg=GOLD)
        code_lbl.config(fg=GOLD_DIM)

        code, name, desc = STAGES[idx]
        self.stage_tag.config(text=f" {code} ")
        self.stage_title.config(text=name)
        self.stage_desc.config(text=f"·  {desc}")

        self._set_status(f"LOADING STAGE {code} — {name}")
        self._render_stage(idx)
        self._set_status(f"STAGE {code} ACTIVE  ·  {desc}")

    def _run_all_and_show(self):
        self._run_all_stages()
        self._select_stage(9)
        messagebox.showinfo("Pipeline Complete",
                            "✓  All 10 stages executed successfully.\n\nShowing final enhanced output (S10).")

    # ── RENDER DISPATCHER ─────────────────────────────────────────
    def _render_stage(self, idx):
        self.data_text.config(state='normal')
        self.data_text.delete('1.0', 'end')
        # Destroy ALL children of chart_outer (handles both normal canvas
        # widgets AND the extra frames/sliders that S7 and S9 pack in)
        for widget in self.chart_outer.winfo_children():
            widget.destroy()
        self.canvas_widget = None
        # Re-add the header bar that lives inside chart_outer
        ch_hdr = tk.Frame(self.chart_outer, bg=GOLD_DARK, pady=5)
        ch_hdr.pack(fill='x')
        tk.Label(ch_hdr, text="◈  VISUALIZATION",
                 font=("Courier", 8, "bold"),
                 fg=GOLD, bg=GOLD_DARK).pack(anchor='w', padx=10)
        tk.Frame(self.chart_outer, bg=GOLD, height=1).pack(fill='x')
        [self._r0, self._r1, self._r2, self._r3, self._r4,
         self._r5, self._r6, self._r7, self._r8, self._r9][idx]()
        self.data_text.config(state='disabled')

    def _w(self, text, tag=None):
        if tag:
            self.data_text.insert('end', text + '\n', tag)
        else:
            self.data_text.insert('end', text + '\n')

    def _sep(self):
        self._w("─" * 32, 'dim')

    def _fig(self):
        fig = Figure(facecolor=PANEL2)
        return fig

    def _show(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.chart_outer)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas_widget = canvas

    def _ax_style(self, ax, title=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT3, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(BORDER2)
        if title:
            ax.set_title(title, color=GOLD_DIM, fontsize=8, pad=5,
                         fontfamily='monospace')

    # ── S1 ────────────────────────────────────────────────────────
    def _r0(self):
        r = self.results
        self._w("▸ MATRIX REPRESENTATION", 'head'); self._sep()
        self._w("Rows  = Trading Days (10)", 'dim')
        self._w("Cols  = Stocks (5)", 'dim')
        self._w("Entry = Closing Price\n", 'dim')
        self._w("COL MEANS REMOVED:", 'gold')
        for s, m in zip(STOCKS, r['mean']):
            self._w(f"  {s:<5}  μ = {m:>7.2f}", 'green')
        self._sep()
        self._w("CENTERED  [Day 1]:", 'gold')
        for s, v in zip(STOCKS, r['s1'][0]):
            tag = 'green' if v >= 0 else 'red'
            self._w(f"  {s:<5}  {v:>+8.3f}", tag)
        self._sep()
        self._w(f"Shape:  {r['s1'].shape}", 'amber')

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('TOP: raw price lines per stock', 'dim')
        self._w('  shows absolute price levels', 'dim')
        self._w('BOTTOM: mean-centered prices', 'dim')
        self._w('  zero line = average price', 'dim')
        self._w('  +/- = above/below average', 'dim')
        self._w('  divergence = unique movement', 'green')

        fig = self._fig()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        fig.subplots_adjust(hspace=0.5, left=0.1, right=0.97, top=0.93, bottom=0.08)
        nd = r['raw'].shape[0]
        for i, s in enumerate(STOCKS):
            ax1.plot(range(1, nd+1), r['raw'][:, i], color=SCOLS[i % len(SCOLS)],
                     lw=1.8, marker='o', ms=3, label=s)
            ax2.plot(range(1, nd+1), r['s1'][:, i], color=SCOLS[i % len(SCOLS)],
                     lw=1.8, marker='o', ms=3)
        for ax, t in [(ax1, "RAW PRICES  ($)"), (ax2, "MEAN-CENTERED  (Δ$)")]:
            self._ax_style(ax, t)
            ax.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT,
                      framealpha=0.8, ncol=5, loc='upper left')
        ax2.axhline(0, color=BORDER2, lw=0.7, ls='--')
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)

    # ── S2 ────────────────────────────────────────────────────────
    def _r1(self):
        r = self.results
        self._w("▸ LU DECOMPOSITION", 'head'); self._sep()
        self._w("AᵀA  =  P · L · U\n", 'gold')
        self._w(f"AᵀA  shape: {r['s2_M'].shape}", 'dim')
        self._w("L  (Lower triangular):", 'gold')
        for row in r['s2_L']:
            self._w("  " + "  ".join(f"{v:5.2f}" for v in row), 'green')
        self._sep()
        self._w("U  (Upper triangular):", 'gold')
        for row in r['s2_U']:
            self._w("  " + "  ".join(f"{v:6.0f}" for v in row), 'amber')
        self._sep()
        self._w("PURPOSE:", 'gold')
        self._w("Solves Ax=b in O(n²)", 'dim')
        self._w("instead of O(n³)", 'dim')

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('3 heatmaps = P, L, U matrices', 'dim')
        self._w('Colour = magnitude of entry', 'dim')
        self._w('L: lower triangle non-zero', 'dim')
        self._w('  (row operations captured)', 'green')
        self._w('U: upper triangle non-zero', 'dim')
        self._w('  (reduced form of AᵀA)', 'green')
        self._w('P: permutation (row swaps)', 'dim')

        fig = self._fig()
        axes = [fig.add_subplot(1, 3, k+1) for k in range(3)]
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, wspace=0.35)
        mats = [('P', r['s2_P']), ('L', r['s2_L']), ('U', r['s2_U'])]
        cmaps = ['YlOrBr', 'YlGn', 'PuBu']
        for ax, (nm, mat), cm in zip(axes, mats, cmaps):
            im = ax.imshow(mat, cmap=cm, aspect='auto')
            self._ax_style(ax, nm)
            n = len(STOCKS)
            ax.set_xticks(range(n)); ax.set_xticklabels(STOCKS, fontsize=max(4,6-max(0,n-5)), color=TEXT3)
            ax.set_yticks(range(n)); ax.set_yticklabels(STOCKS, fontsize=max(4,6-max(0,n-5)), color=TEXT3)
            for ii in range(n):
                for jj in range(n):
                    ax.text(jj, ii, f"{mat[ii,jj]:.1f}", ha='center', va='center',
                            fontsize=5.5, color='black')
        fig.suptitle("LU DECOMPOSITION  ·  AᵀA = P·L·U",
                     color=GOLD_DIM, fontsize=8, fontfamily='monospace')
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)

    # ── S3 ────────────────────────────────────────────────────────
    def _r2(self):
        r = self.results
        rank, null = r['s3_rank'], r['s3_null']
        self._w("▸ RANK & NULLITY", 'head'); self._sep()
        self._w(f"Matrix  :  10 × 5", 'dim')
        self._w(f"Rank    :  {rank}   ← indep. directions", 'green')
        self._w(f"Nullity :  {null}   ← redundant dims\n", 'amber')
        self._sep()
        self._w("RANK-NULLITY THEOREM:", 'gold')
        self._w(f"  rank + nullity = n", 'dim')
        self._w(f"  {rank}  +  {null}  =  {len(STOCKS)}", 'green')
        self._sep()
        self._w("MARKET INTERPRETATION:", 'gold')
        self._w(f"  {rank} of {len(STOCKS)} stocks carry", 'dim')
        self._w(f"  INDEPENDENT information.", 'green')
        if null:
            self._w(f"  {null} stock(s) are linearly", 'dim')
            self._w(f"  dependent on the others.", 'amber')

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('Bar = each singular value', 'dim')
        self._w('GREEN bars = rank space', 'green')
        self._w('  (independent directions)', 'dim')
        self._w('RED bars = null space', 'red')
        self._w('  (redundant/zero dimensions)', 'dim')
        self._w('Tall bar = more variance', 'dim')
        self._w('Near-zero bar ≈ dependent', 'amber')

        fig = self._fig()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.12)
        _, sv, _ = np.linalg.svd(r['s1'])
        cols = [GREEN if i < rank else RED for i in range(len(sv))]
        bars = ax.bar(range(1, len(sv)+1), sv, color=cols,
                      edgecolor=BORDER, linewidth=0.8, width=0.5)
        for bar, v in zip(bars, sv):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{v:.1f}", ha='center', color=TEXT3, fontsize=7,
                    fontfamily='monospace')
        self._ax_style(ax, "SINGULAR VALUES  ·  Non-zero = Rank")
        ax.set_xlabel("Component", color=TEXT3, fontsize=7, fontfamily='monospace')
        ax.set_ylabel("Singular Value", color=TEXT3, fontsize=7, fontfamily='monospace')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=GREEN, label=f'Rank space ({rank})'),
                            Patch(color=RED,   label=f'Null space ({null})')],
                  fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)

    # ── S4 ────────────────────────────────────────────────────────
    def _r3(self):
        r = self.results
        basis = r['s4_basis']
        self._w("▸ BASIS SELECTION", 'head'); self._sep()
        self._w("Min. set of linearly indep.", 'dim')
        self._w("vectors spanning price space.\n", 'dim')
        self._w(f"Basis shape : {basis.shape}", 'amber')
        self._w(f"  {basis.shape[0]} vectors × {basis.shape[1]} stocks\n", 'dim')
        self._sep()
        self._w("BASIS VECTORS:", 'gold')
        for i, row in enumerate(basis):
            self._w(f"  B{i+1}: [" + "  ".join(f"{v:+.3f}" for v in row) + "]", 'green')
        self._sep()
        self._w("No redundancy — each vector", 'dim')
        self._w("is linearly independent.", 'green')

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('Heatmap rows = basis vectors', 'dim')
        self._w('Columns = stocks', 'dim')
        self._w('Warm colour = large positive', 'dim')
        self._w('Cool colour = large negative', 'dim')
        self._w('Each row spans one independent', 'dim')
        self._w('market direction', 'green')
        self._w('Similar rows = redundancy', 'amber')

        fig = self._fig()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.14)
        im = ax.imshow(basis, cmap='YlOrBr', aspect='auto')
        self._ax_style(ax, "BASIS VECTORS  ·  Rows = Basis Directions")
        ax.set_xticks(range(len(STOCKS))); ax.set_xticklabels(STOCKS, color=TEXT, fontsize=max(6,9-max(0,len(STOCKS)-5)))
        ax.set_yticks(range(basis.shape[0]))
        ax.set_yticklabels([f"B{i+1}" for i in range(basis.shape[0])],
                           color=TEXT, fontsize=9)
        for i in range(basis.shape[0]):
            for j in range(basis.shape[1]):
                ax.text(j, i, f"{basis[i,j]:.2f}", ha='center', va='center',
                        fontsize=8, color='black', fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)

    # ── S5 ────────────────────────────────────────────────────────
    def _r4(self):
        r = self.results
        Q = r['s5_Q']
        QtQ = Q.T @ Q
        self._w("▸ GRAM-SCHMIDT", 'head'); self._sep()
        self._w("Convert basis → orthogonal.", 'dim')
        self._w("Each vector ⊥ all others.\n", 'dim')
        self._w(f"Q shape  : {Q.shape}", 'amber')
        self._sep()
        self._w("ORTHOGONALITY  (QᵀQ ≈ I):", 'gold')
        for row in QtQ:
            self._w("  [" + " ".join(f"{v:5.2f}" for v in row) + "]", 'green')
        self._sep()
        self._w("VERIFICATION:", 'gold')
        err = np.max(np.abs(QtQ - np.eye(QtQ.shape[0])))
        tag = 'green' if err < 1e-6 else 'amber'
        self._w(f"  Max deviation: {err:.2e}", tag)
        self._w("  ✓ Orthogonality confirmed" if err < 1e-6
                else "  ⚠ Small numerical error", tag)
        self._sep()
        self._w("Each column = one pure", 'dim')
        self._w("independent market trend.", 'green')

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('LEFT: Q matrix heatmap', 'dim')
        self._w('  columns = orthogonal vectors', 'dim')
        self._w('  each = a pure trend direction', 'green')
        self._w('RIGHT: QᵀQ should = Identity', 'dim')
        self._w('  diagonal=1 (unit length)', 'green')
        self._w('  off-diagonal≈0 (orthogonal)', 'green')
        self._w('  any off-diag ≠ 0 = error', 'amber')

        fig = self._fig()
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                               left=0.08, right=0.97, top=0.88, bottom=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax1.imshow(Q, cmap='YlGn', aspect='auto')
        self._ax_style(ax1, "Q  (Orthogonal Basis)")
        ax1.set_xticks(range(Q.shape[1]))
        ax1.set_xticklabels([f"v{j+1}" for j in range(Q.shape[1])], color=TEXT, fontsize=max(6,8-max(0,len(STOCKS)-5)))
        ax1.set_yticks(range(Q.shape[0])); ax1.set_yticklabels(STOCKS[:Q.shape[0]], color=TEXT, fontsize=max(6,8-max(0,len(STOCKS)-5)))
        im2 = ax2.imshow(QtQ, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=1.1)
        self._ax_style(ax2, "QᵀQ  ≈  Identity")
        ax2.set_xticks(range(QtQ.shape[0])); ax2.set_yticks(range(QtQ.shape[0]))
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)

    # ── S6 ────────────────────────────────────────────────────────
    def _r5(self):
        r = self.results
        orig = r['s1']; proj = r['s6_proj']
        resid = orig - proj
        self._w("▸ PROJECTION", 'head'); self._sep()
        self._w("Project data onto orthogonal", 'dim')
        self._w("subspace → removes noise.\n", 'dim')
        self._w("FORMULA:", 'gold')
        self._w("  P = Q(QᵀQ)⁻¹Qᵀ · A", 'green')
        self._sep()
        self._w("RESIDUAL (noise):", 'gold')
        self._w(f"  Max  : {np.abs(resid).max():.6f}", 'green')
        self._w(f"  Mean : {np.abs(resid).mean():.6f}", 'green')
        self._w(f"  Norm : {np.linalg.norm(resid):.6f}", 'amber')
        self._sep()
        self._w("PROJECTION CAPTURES:", 'gold')
        self._w("  • True underlying trend", 'green')
        self._w("  • Discards fluctuations", 'green')
        self._w("  • Fills missing values", 'green')

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('DASHED = original noisy data', 'dim')
        self._w('SOLID = projected clean data', 'green')
        self._w('Gap between = noise removed', 'dim')
        self._w('Solid lines are smoother', 'green')
        self._w('Both lines overlap if data', 'dim')
        self._w('already lies in subspace', 'amber')

        fig = self._fig()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.1)
        nd = orig.shape[0]
        for i, s in enumerate(STOCKS):
            ax.plot(range(1, nd+1), orig[:, i], color=SCOLS[i % len(SCOLS)],
                    ls='--', lw=1.2, alpha=0.45)
            ax.plot(range(1, nd+1), proj[:, i], color=SCOLS[i % len(SCOLS)],
                    lw=2.2, label=s)
        ax.axhline(0, color=BORDER2, lw=0.6, ls=':')
        self._ax_style(ax, "ORIGINAL (dashed)  vs  PROJECTED (solid)")
        ax.set_xlabel("Day", color=TEXT3, fontsize=7, fontfamily='monospace')
        ax.set_ylabel("Centered Price", color=TEXT3, fontsize=7, fontfamily='monospace')
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, ncol=5)
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)

    # ── S7 ────────────────────────────────────────────────────────
    def _r6(self):
        r = self.results
        nd = r['raw'].shape[0]
        self._s7_day = tk.IntVar(value=nd + 1)

        pred_real = r['s7_pred'] + r['mean']
        self._w("▸ LEAST SQUARES", 'head'); self._sep()
        self._w("x̂ = (AᵀA)⁻¹ · Aᵀ · b\n", 'gold')
        self._w("Fit linear trend per stock.\n", 'dim')
        self._w("GRAPH EXPLANATION:", 'gold')
        self._w("• Solid lines = actual prices", 'dim')
        self._w("• Dashed lines = fitted trend", 'dim')
        self._w("• ◆ markers = prediction", 'green')
        self._w("• Vertical line = present/future", 'dim')
        self._w("• Slope shows bullish/bearish", 'dim')
        self._sep()
        self._w("DRAG SLIDER → PICK DAY", 'gold')
        self._s7_info_start = self.data_text.index('end')
        self._update_s7_info(nd + 1)

        # ── chart frame with slider ──
        outer = tk.Frame(self.chart_outer, bg=PANEL2)
        outer.pack(fill='both', expand=True)

        # slider bar
        ctrl = tk.Frame(outer, bg=PANEL2, pady=6)
        ctrl.pack(fill='x', padx=10)
        tk.Label(ctrl, text="PREDICT DAY:", font=("Courier", 8, "bold"),
                 fg=GOLD, bg=PANEL2).pack(side='left', padx=(0,8))
        self._s7_day_lbl = tk.Label(ctrl, text=f"Day {nd+1}",
                                     font=("Courier", 10, "bold"),
                                     fg=GREEN, bg=PANEL2, width=8)
        self._s7_day_lbl.pack(side='left')
        slider = tk.Scale(ctrl, from_=nd+1, to=nd+30,
                          orient='horizontal', variable=self._s7_day,
                          bg=PANEL2, fg=GOLD, troughcolor=GOLD_DARK,
                          highlightthickness=0, relief='flat',
                          font=("Courier", 7), showvalue=0,
                          command=self._on_s7_slider)
        slider.pack(side='left', fill='x', expand=True, padx=8)
        tk.Label(ctrl, text=f"max D{nd+30}",
                 font=("Courier", 7), fg=TEXT3, bg=PANEL2).pack(side='left')

        tk.Frame(outer, bg=BORDER, height=1).pack(fill='x', padx=6)

        # chart canvas area
        self._s7_chart_frame = tk.Frame(outer, bg=PANEL2)
        self._s7_chart_frame.pack(fill='both', expand=True)
        self._draw_s7_chart(nd + 1)

    def _on_s7_slider(self, val):
        day = int(float(val))
        self._s7_day_lbl.config(text=f"Day {day}")
        self._draw_s7_chart(day)
        self._update_s7_info(day)

    def _update_s7_info(self, day):
        r = self.results
        # rewrite the prediction block in the text widget
        self.data_text.config(state='normal')
        try:
            self.data_text.delete(self._s7_info_start, 'end')
        except Exception:
            pass
        nd = r['raw'].shape[0]
        days_arr = np.arange(1, nd + 1).reshape(-1, 1)
        X = np.hstack([np.ones_like(days_arr), days_arr])
        X_pred = np.array([[1, day]])
        pred = (X_pred @ r['s7_c']).flatten() + r['mean']
        self._w(f"\nDAY {day} PREDICTIONS:", 'gold')
        for s, p, last in zip(STOCKS, pred, r['raw'][-1]):
            delta = p - last
            tag = 'green' if delta >= 0 else 'red'
            arrow = '▲' if delta >= 0 else '▼'
            self._w(f"  {s:<5}  ${p:>7.2f}  {arrow}{abs(delta):.2f}", tag)
        self._sep()
        self._w(f"Day {nd} actual:", 'gold')
        for s, p in zip(STOCKS, r['raw'][-1]):
            self._w(f"  {s:<5}  ${p:>7.2f}", 'dim')
        self.data_text.config(state='disabled')

    def _draw_s7_chart(self, predict_day):
        # clear old chart
        for w in self._s7_chart_frame.winfo_children():
            w.destroy()
        r = self.results
        nd = r['raw'].shape[0]
        days_arr = np.arange(1, nd+1).reshape(-1,1)
        X_pred = np.array([[1, predict_day]])
        pred_real = (X_pred @ r['s7_c']).flatten() + r['mean']

        fig = self._fig()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.1)
        for i, s in enumerate(STOCKS):
            col = SCOLS[i % len(SCOLS)]
            ax.plot(range(1, nd+1), r['raw'][:, i],
                    color=col, lw=1.8, marker='o', ms=3)
            c = r['s7_c'][:, i]; m = r['mean'][i]
            tx = np.array([1, predict_day])
            ty = c[0] + c[1]*tx + m
            ax.plot(tx, ty, color=col, ls='--', lw=1, alpha=0.6)
            ax.scatter([predict_day], [pred_real[i]],
                       color=col, marker='D', s=70, zorder=6,
                       label=f"{s} ◆${pred_real[i]:.0f}")
            ax.plot([nd, predict_day], [r['raw'][-1, i], pred_real[i]],
                    color=col, lw=1.5, ls='--', alpha=0.7)
        ax.axvline(nd + 0.5, color=GOLD_DIM, lw=0.8, ls=':')
        ax.axvline(predict_day, color=GREEN, lw=0.8, ls=':', alpha=0.5)
        ax.text(nd + 0.6, r['raw'].min() - 2, "NOW",
                color=GOLD_DIM, fontsize=7, fontfamily='monospace')
        ax.text(predict_day - 0.4, r['raw'].min() - 2, f"D{predict_day}",
                color=GREEN, fontsize=7, fontfamily='monospace')
        self._ax_style(ax, f"LEAST SQUARES PREDICTION  ·  Day {predict_day}")
        ax.set_xlabel("Day", color=TEXT3, fontsize=7, fontfamily='monospace')
        ax.set_ylabel("Price ($)", color=TEXT3, fontsize=7, fontfamily='monospace')
        ax.legend(fontsize=6.5, facecolor=PANEL, labelcolor=TEXT, ncol=2)
        fig.patch.set_facecolor(PANEL2)

        canvas = FigureCanvasTkAgg(fig, master=self._s7_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── S8 ────────────────────────────────────────────────────────
    def _r7(self):
        r = self.results
        evals = r['s8_ev']; evecs = r['s8_vec']; total = evals.sum()
        self._w("▸ EIGENANALYSIS", 'head'); self._sep()
        self._w("Cov = AᵀA / (n-1)\n", 'gold')
        self._w("EIGENVALUES  (% variance):", 'gold')
        for i, ev in enumerate(evals):
            pct = 100*ev/total if total else 0
            bar_len = int(pct / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            tag = 'green' if i == 0 else ('amber' if i == 1 else 'dim')
            self._w(f"  λ{i+1} {bar}  {pct:.1f}%", tag)
        self._sep()
        self._w("PC1 EIGENVECTOR:", 'gold')
        for s, v in zip(STOCKS, evecs[:, 0]):
            tag = 'green' if v > 0 else 'red'
            self._w(f"  {s:<5}  {v:+.4f}", tag)

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('LEFT: scree plot (% variance)', 'dim')
        self._w('  tall bar = dominant trend', 'green')
        self._w('  drop-off shows how many PCs', 'dim')
        self._w('  are actually meaningful', 'dim')
        self._w('RIGHT: eigenvector loadings', 'dim')
        self._w('  bar height = stock contribution', 'dim')
        self._w('  PC1 = biggest market driver', 'green')
        self._w('  same-sign = stocks co-move', 'amber')

        fig = self._fig()
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                               left=0.1, right=0.97, top=0.88, bottom=0.12)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        pcts = [100*ev/total if total else 0 for ev in evals]
        bars = ax1.bar(range(1, len(pcts)+1), pcts,
                       color=[SCOLS[i % len(SCOLS)] for i in range(len(pcts))],
                       edgecolor=BORDER, lw=0.8, width=0.6)
        for bar, pct in zip(bars, pcts):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                     f"{pct:.0f}%", ha='center', color=TEXT3,
                     fontsize=7, fontfamily='monospace')
        self._ax_style(ax1, "EIGENVALUE  ·  % VARIANCE")
        ax1.set_xlabel("PC", color=TEXT3, fontsize=7, fontfamily='monospace')
        ax1.set_ylabel("Variance %", color=TEXT3, fontsize=7, fontfamily='monospace')
        x = np.arange(len(STOCKS))
        for k in range(min(3, evecs.shape[1])):
            ax2.bar(x+k*0.25, evecs[:, k], width=0.25,
                    color=[GOLD, GREEN, AMBER][k], label=f"PC{k+1}", alpha=0.9)
        ax2.set_xticks(x+0.25); ax2.set_xticklabels(STOCKS, color=TEXT, fontsize=max(6,8-max(0,len(STOCKS)-5)))
        ax2.axhline(0, color=BORDER2, lw=0.5)
        self._ax_style(ax2, "TOP 3 EIGENVECTORS")
        ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)

    # ── S9 ────────────────────────────────────────────────────────
    def _r8(self):
        r = self.results
        sv = r['s9_sv']
        max_k = len(sv)

        self._w("▸ SVD COMPRESSION", 'head'); self._sep()
        self._w("A  =  U · Σ · Vᵀ", 'gold')
        self._w("Drag slider to change k\n", 'dim')
        self._w("GRAPH EXPLANATION:", 'gold')
        self._w("• TOP: original centered data", 'dim')
        self._w("• BOTTOM: SVD reconstruction", 'dim')
        self._w("  with chosen k components", 'dim')
        self._w("• Closer match = higher k", 'dim')
        self._w("• Smoother curves = lower k", 'green')
        self._w("• Watch noise disappear as", 'dim')
        self._w("  k drops from max to 1", 'green')
        self._sep()
        self._w("SINGULAR VALUES:", 'gold')
        for i, s in enumerate(sv):
            self._w(f"  σ{i+1}  =  {s:.4f}", 'green' if i == 0 else 'dim')
        self._sep()
        self._s9_info_start = self.data_text.index('end')
        self._update_s9_info(min(2, max_k))

        # chart outer frame
        outer = tk.Frame(self.chart_outer, bg=PANEL2)
        outer.pack(fill='both', expand=True)

        # slider bar
        ctrl = tk.Frame(outer, bg=PANEL2, pady=6)
        ctrl.pack(fill='x', padx=10)
        tk.Label(ctrl, text="COMPONENTS k:", font=("Courier", 8, "bold"),
                 fg=GOLD, bg=PANEL2).pack(side='left', padx=(0, 8))
        self._s9_k_lbl = tk.Label(ctrl, text=f"k = {min(2, max_k)}",
                                   font=("Courier", 10, "bold"),
                                   fg=GREEN, bg=PANEL2, width=6)
        self._s9_k_lbl.pack(side='left')
        self._s9_var_lbl = tk.Label(ctrl, text="",
                                     font=("Courier", 9, "bold"),
                                     fg=AMBER, bg=PANEL2, width=14)
        self._s9_var_lbl.pack(side='left', padx=6)
        tk.Label(ctrl, text="k=1", font=("Courier", 7), fg=TEXT3, bg=PANEL2).pack(side='left')
        self._s9_slider = tk.Scale(ctrl, from_=1, to=max_k,
                                    orient='horizontal',
                                    bg=PANEL2, fg=GOLD, troughcolor=GOLD_DARK,
                                    highlightthickness=0, relief='flat',
                                    font=("Courier", 7), showvalue=0,
                                    command=self._on_s9_slider)
        self._s9_slider.set(min(2, max_k))
        self._s9_slider.pack(side='left', fill='x', expand=True, padx=4)
        tk.Label(ctrl, text=f"k={max_k}", font=("Courier", 7),
                 fg=TEXT3, bg=PANEL2).pack(side='left')

        tk.Frame(outer, bg=BORDER, height=1).pack(fill='x', padx=6)

        self._s9_chart_frame = tk.Frame(outer, bg=PANEL2)
        self._s9_chart_frame.pack(fill='both', expand=True)
        self._draw_s9_chart(min(2, max_k))

    def _on_s9_slider(self, val):
        k = int(float(val))
        self._s9_k_lbl.config(text=f"k = {k}")
        self._draw_s9_chart(k)
        self._update_s9_info(k)

    def _update_s9_info(self, k):
        r = self.results
        sv = r['s9_sv']
        var = float(np.sum(sv[:k]**2) / np.sum(sv**2) * 100)
        nd = r['s1'].shape[0]
        n_stocks = len(STOCKS)
        orig_params = nd * n_stocks
        kept_params = nd * k + k + k * n_stocks
        savings = max(0, (1 - kept_params / orig_params) * 100)
        try:
            self._s9_var_lbl.config(text=f"{var:.1f}% retained")
        except Exception:
            pass
        self.data_text.config(state='normal')
        try:
            self.data_text.delete(self._s9_info_start, 'end')
        except Exception:
            pass
        self._w(f"\nk = {k}  COMPONENTS:", 'gold')
        col_tag = 'green' if var >= 95 else ('amber' if var >= 80 else 'red')
        self._w(f"  Variance retained: {var:.2f}%", col_tag)
        self._w(f"  Params: {orig_params} → {kept_params}", 'dim')
        self._w(f"  Compression: {savings:.0f}%", 'green')
        self._sep()
        self._w("COMPONENTS USED:", 'gold')
        for i in range(len(sv)):
            mark = '✓' if i < k else '✗'
            pct = sv[i]**2 / np.sum(sv**2) * 100
            tag = 'green' if i < k else 'dim'
            self._w(f"  {mark} σ{i+1}={sv[i]:.3f} ({pct:.1f}%)", tag)
        self.data_text.config(state='disabled')

    def _draw_s9_chart(self, k):
        for w in self._s9_chart_frame.winfo_children():
            w.destroy()
        r = self.results
        sv = r['s9_sv']
        U, _, Vt = np.linalg.svd(r['s1'], full_matrices=False)
        compressed = U[:, :k] @ np.diag(sv[:k]) @ Vt[:k, :]
        var = float(np.sum(sv[:k]**2) / np.sum(sv**2) * 100)
        nd = r['s1'].shape[0]

        fig = self._fig()
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5,
                               left=0.1, right=0.97, top=0.92, bottom=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        for i, s in enumerate(STOCKS):
            col = SCOLS[i % len(SCOLS)]
            ax1.plot(range(1, nd+1), r['s1'][:, i], color=col, lw=1.8, label=s)
            ax2.plot(range(1, nd+1), compressed[:, i], color=col, lw=1.8, ls='--')

        err = np.linalg.norm(r['s1'] - compressed, 'fro')
        self._ax_style(ax1, "ORIGINAL  (centered)")
        self._ax_style(ax2, f"SVD k={k}  ·  {var:.1f}% retained  ·  error={err:.3f}")
        for ax in [ax1, ax2]:
            ax.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT, ncol=5)
        # colour the title green/amber/red based on quality
        col = GREEN if var >= 95 else (AMBER if var >= 80 else RED)
        ax2.title.set_color(col)
        fig.patch.set_facecolor(PANEL2)

        canvas = FigureCanvasTkAgg(fig, master=self._s9_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── S10 ───────────────────────────────────────────────────────
    def _r9(self):
        r = self.results
        enh = r['s10_enh']; raw = r['raw']
        pred_real = r['s7_pred'] + r['mean']
        self._w("▸ FINAL MARKET MODEL", 'head'); self._sep()
        self._w("SVD-compressed + denoised\n+ linear prediction.\n", 'dim')
        self._w("ENHANCED vs ORIGINAL (D1):", 'gold')
        for s, o, e in zip(STOCKS, raw[0], enh[0]):
            delta = e - o
            tag = 'green' if delta >= 0 else 'red'
            self._w(f"  {s:<5}  {o:.1f} → {e:.1f}  ({delta:+.2f})", tag)
        self._sep()
        self._w("DAY 11 FORECAST:", 'gold')
        for s, p, last in zip(STOCKS, pred_real, raw[-1]):
            d = p - last
            arrow = '▲' if d >= 0 else '▼'
            tag = 'green' if d >= 0 else 'red'
            self._w(f"  {s:<5}  ${p:>7.2f}  {arrow}{abs(d):.2f}", tag)
        self._sep()
        self._w("PIPELINE STATUS:", 'gold')
        self._w("  ✓ All 10 stages complete", 'green')
        self._w("  ✓ Matrix: 10×5", 'green')
        self._w(f"  ✓ Rank  = {r['s3_rank']}", 'green')
        self._w(f"  ✓ SVD variance = {r['s9_var']:.1f}%", 'green')

        self._sep()
        self._w('GRAPH EXPLANATION:', 'gold')
        self._w('DOTTED = raw original prices', 'dim')
        self._w('SOLID = SVD-enhanced clean', 'green')
        self._w('  prices (denoised model)', 'green')
        self._w('◆ DIAMOND = Day N+1 forecast', 'gold')
        self._w('DASHED tail = predicted path', 'dim')
        self._w('Green vert. line = future', 'dim')
        self._w('Diverging tails = stocks with', 'dim')
        self._w('different predicted momentum', 'amber')

        fig = self._fig()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.1)
        nd = raw.shape[0]
        for i, s in enumerate(STOCKS):
            ax.plot(range(1, nd+1), raw[:, i],
                    color=SCOLS[i], lw=1, ls=':', alpha=0.4)
            ax.plot(range(1, nd+1), enh[:, i],
                    color=SCOLS[i], lw=2.2, label=s)
            ax.scatter([11], [pred_real[i]], color=SCOLS[i],
                       marker='D', s=80, zorder=6)
            ax.plot([10, 11], [enh[-1, i], pred_real[i]],
                    color=SCOLS[i], lw=1.8, ls='--')
        ax.axvline(10.5, color=GOLD_DIM, lw=0.8, ls=':')
        ax.text(10.55, raw.min()-2, "D11", color=GOLD_DIM,
                fontsize=7, fontfamily='monospace')
        self._ax_style(ax, "FINAL MODEL  ·  clean (solid) · raw (dot) · D11 forecast (◆)")
        ax.set_xlabel("Day", color=TEXT3, fontsize=7, fontfamily='monospace')
        ax.set_ylabel("Price ($)", color=TEXT3, fontsize=7, fontfamily='monospace')
        from matplotlib.lines import Line2D
        handles = [Line2D([0],[0], color=SCOLS[i], lw=2, label=STOCKS[i])
                   for i in range(5)]
        ax.legend(handles=handles, fontsize=8, facecolor=PANEL,
                  labelcolor=TEXT, ncol=5)
        fig.patch.set_facecolor(PANEL2)
        self._show(fig)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = BloombergPipelineApp(root)
    root.mainloop()
