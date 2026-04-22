# LA-QUANT · Stock Market Linear Algebra Pipeline

> **PES University · UE24MA241B — Linear Algebra and Its Applications**  
> Mini Project · Department of Computer Science and Engineering

---

## Overview

LA-QUANT is an interactive **Bloomberg-style terminal application** that applies a complete 10-stage Linear Algebra pipeline to stock market price data. Raw closing prices are transformed through matrix operations, orthogonalization, projection, eigenanalysis, and SVD compression — ending with a denoised market model and future price predictions.

**Dataset:** 10 trading days × 5 stocks (AAPL, GOOG, MSFT, AMZN, TSLA)  
**Language:** Python 3 · tkinter · numpy · scipy · matplotlib  
**UI Theme:** Bloomberg Terminal — Green & Gold

---

## Screenshot

<!-- Replace with: ![App Screenshot](screenshots/app_overview.png) -->
> _Paste a screenshot of the full application window here_

---

## Getting Started

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

> Python 3.7+ required. `tkinter` is included with standard Python installs.

### Run

```bash
python stock_pipeline.py
```

---

## How to Use

| Action | Description |
|---|---|
| **Click a stage (S1–S10)** | Navigate to that pipeline stage |
| **DATA OUTPUT panel** | Shows numbers, matrix values, and graph explanation |
| **VISUALIZATION panel** | Shows the interactive chart for the selected stage |
| **S7 — Prediction slider** | Drag to choose any future day (Day N+1 to N+30) |
| **S9 — k-components slider** | Drag to vary SVD compression quality live |
| **LOAD CSV** *(top-right)* | Upload your own stock price dataset |
| **EXECUTE FULL PIPELINE** | Re-runs all 10 stages, jumps to final output |

---

## CSV Upload Format

```
Day,AAPL,GOOG,MSFT,...
1,182.0,140.0,375.0,...
2,185.0,142.0,378.0,...
```

- **First row:** column headers (first column can be a date/day label)
- **Remaining rows:** one row per trading day, numeric prices only
- **Minimum:** 3 days × 2 stocks &nbsp;|&nbsp; **Maximum:** any days, up to 10 stocks

A `sample_data.csv` file is included for testing.

---

## The Linear Algebra Pipeline

```
REAL-WORLD DATA
      │
      ▼
S1  NORMALIZE        Matrix Representation · Mean Centering
      │
      ▼
S2  LU DECOMP        LU Factorization of AᵀA
      │
      ▼
S3  RANK/NULLITY     Rank via SVD · Rank-Nullity Theorem
      │
      ▼
S4  BASIS SELECT     Basis Extraction from SVD
      │
      ▼
S5  GRAM-SCHMIDT     Orthogonalization of Basis Vectors
      │
      ▼
S6  PROJECTION       Orthogonal Projection  P = QQᵀA
      │
      ▼
S7  LEAST SQUARES    Normal Equations  x̂ = (AᵀA)⁻¹Aᵀb
      │
      ▼
S8  EIGENANALYSIS    Eigendecomposition of Covariance Matrix
      │
      ▼
S9  SVD COMPRESS     Truncated SVD · top-k singular values
      │
      ▼
S10 ENHANCE          Final Denoised Market Model + Predictions
```

---

## Stage Details

### S1 — Normalize
**Concept:** Matrix Representation · Mean Centering  
**What:** Raw prices loaded into a 10×5 matrix. Each column is mean-centered by subtracting its average.  
**Why:** Different stocks have different price scales. Centering ensures no single stock dominates later calculations.  
**Output:** Mean-centered matrix where each column sums to zero.  
**Graph:** TOP — raw absolute prices per stock. BOTTOM — centered prices oscillating around zero.  
**Alternate:** Z-score standardization / Min-Max scaling / Log Returns

<img width="1917" height="1034" alt="image" src="https://github.com/user-attachments/assets/3b2165d4-8a53-4410-9f1e-b110d93884fb" />


---

### S2 — LU Decomposition
**Concept:** LU Factorization of AᵀA  
**What:** Factorizes the covariance matrix as AᵀA = P·L·U (permutation × lower × upper triangular).  
**Why:** Reduces cost of solving Ax = b from O(n³) to O(n²) per right-hand side — used in S7.  
**Output:** Three matrices P, L, U. L has non-zero entries only below the diagonal; U only above.  
**Graph:** Three heatmaps side by side. Warm colours = large magnitudes. Triangular structure clearly visible.  
**Alternate:** Cholesky decomposition / QR decomposition / Gaussian Elimination

<img width="1919" height="1033" alt="image" src="https://github.com/user-attachments/assets/97b611b1-3b45-4104-8481-47c50c942fea" />


---

### S3 — Rank & Nullity
**Concept:** Rank via SVD · Rank-Nullity Theorem  
**What:** Rank = number of non-zero singular values. Nullity = columns − rank.  
**Why:** Rank tells us how many truly independent market directions exist among the 5 stocks.  
**Output:** Rank = 4, Nullity = 1 → 4 independent price patterns found among the stocks.  
**Graph:** Bar chart of singular values. GREEN = rank space (independent). RED = null space (redundant).  
**Alternate:** RREF pivot count / Eigenvalue count / Determinant check

<img width="1919" height="1032" alt="image" src="https://github.com/user-attachments/assets/49a53ff6-5dc6-4c98-b7a2-b7be57b196b5" />

---

### S4 — Basis Selection
**Concept:** SVD right singular vectors as basis  
**What:** Top `rank` rows of Vt from SVD form the basis — minimal independent directions spanning all movements.  
**Why:** Removes redundant stock directions. Only the essential independent vectors are kept.  
**Output:** Basis matrix of shape (4×5) — 4 independent market direction vectors.  
**Graph:** Heatmap — rows = basis vectors, columns = stocks. Warm = large positive, cool = large negative.  
**Alternate:** RREF pivot columns / Gram-Schmidt directly on original columns

<img width="1919" height="1029" alt="image" src="https://github.com/user-attachments/assets/a3e6bb25-e32b-428e-b534-54e5f977042f" />

---

### S5 — Gram-Schmidt Orthogonalization
**Concept:** Gram-Schmidt Orthogonalization  
**What:** Converts basis vectors into mutually orthogonal unit vectors → matrix Q where QᵀQ = Identity.  
**Why:** Orthogonal vectors have zero dot product — each captures a completely pure, independent market trend.  
**Output:** Matrix Q verified: diagonal of QᵀQ = 1, off-diagonal ≈ 0.  
**Graph:** LEFT — Q matrix heatmap. RIGHT — QᵀQ ≈ Identity (bright diagonal, dark off-diagonal).  
**Alternate:** Modified Gram-Schmidt / Householder Reflections / QR Decomposition

<img width="1919" height="1034" alt="image" src="https://github.com/user-attachments/assets/611d5568-d580-4151-8734-3385bb7052dc" />


---

### S6 — Projection
**Concept:** Orthogonal Projection P = QQᵀA  
**What:** Each day's price vector is projected onto the orthogonal subspace. Residual = noise removed.  
**Why:** Separates true market signal from random daily fluctuations; enables missing value estimation.  
**Output:** Projected matrix with noise residual < 0.0001 — near-perfect projection.  
**Graph:** DASHED = original noisy data. SOLID = projected clean data. Gap = noise removed.  
**Alternate:** SVD Truncation / Oblique Projection

<img width="1919" height="1034" alt="image" src="https://github.com/user-attachments/assets/80d7c871-1a57-40fb-8f8c-b0220b4e328e" />


---

### S7 — Least Squares *(interactive slider)*
**Concept:** Normal Equations &nbsp; x̂ = (AᵀA)⁻¹Aᵀb  
**What:** Fits a linear trend y = a + b·day to each stock. Slider predicts prices at any future day.  
**Why:** The system is overdetermined (10 equations, 2 unknowns). Least Squares finds the best linear approximation.  
**Output:** Predicted prices for any chosen day (N+1 to N+30). Slope sign = bullish (+) or bearish (−).  
**Graph:** Solid = actual prices. Dashed = fitted trend. ◆ = predicted price. Gold line = now, green = chosen day.  
**Alternate:** QR-based Least Squares / SVD Pseudoinverse / Ridge Regression

11th Day prediction
<img width="1919" height="1030" alt="image" src="https://github.com/user-attachments/assets/3b8953c0-4699-43b4-b218-e54f120bae57" />
24th Day prediction
<img width="1919" height="1034" alt="image" src="https://github.com/user-attachments/assets/8c7b1d0c-c050-43c6-b57b-fbfd8d3dc3d5" />


---

### S8 — Eigenanalysis
**Concept:** Eigendecomposition of Covariance Matrix  
**What:** Eigenvalues and eigenvectors of C = AᵀA/(n-1) computed using `numpy.linalg.eigh`.  
**Why:** Large eigenvalues = directions of maximum price variance = dominant market trends.  
**Output:** PC1 explains the dominant variance. All stocks have same-sign PC1 loading (they co-move).  
**Graph:** LEFT — scree plot (% variance per PC). RIGHT — stock loadings on PC1, PC2, PC3.  
**Alternate:** Power Iteration / QR Algorithm / Jacobi Method

<img width="1919" height="1034" alt="image" src="https://github.com/user-attachments/assets/84513e44-838a-48ff-9f83-62b24ea0e56f" />

---

### S9 — SVD Compression *(interactive slider)*
**Concept:** Truncated SVD &nbsp; A = UΣVᵀ  
**What:** Reconstructs A using only top k singular values. Live slider varies k from 1 to max.  
**Why:** SVD gives the optimal rank-k approximation (Eckart-Young theorem) — no other rank-k matrix is closer.  
**Output:** At k=1: very smooth (dominant trend only). At k=max: perfect match. Chart title colour indicates quality.  
**Graph:** TOP — original centered data. BOTTOM — SVD reconstruction at chosen k. Frobenius error shown live.  
**Alternate:** PCA / Randomized SVD / NMF

<img width="1919" height="1033" alt="image" src="https://github.com/user-attachments/assets/83d04b6e-1022-4cd3-8bb5-f4ea3c7b3477" />


---

### S10 — Enhanced Market Model
**Concept:** SVD Reconstruction + Linear Trend Model  
**What:** SVD-compressed matrix has column means added back → final denoised model. Forecasts overlaid.  
**Why:** Combines denoising (SVD), trend alignment (projection), and prediction (least squares) into one output.  
**Output:** Final denoised price curves for all 5 stocks + Day N+1 forecasts as diamond markers.  
**Graph:** DOTTED = raw original. SOLID = denoised model. ◆ = Day N+1 forecast. Dashed tails = predicted path.  
**Alternate:** Kalman Filter / Wiener Filter / Polynomial Fitting

<img width="1919" height="1034" alt="image" src="https://github.com/user-attachments/assets/37859393-0441-4468-b024-d936539987df" />

---

## Input Matrix

| Property | Value |
|---|---|
| Shape | 10 rows (days) × 5 columns (stocks) |
| Stocks | AAPL · GOOG · MSFT · AMZN · TSLA |
| Entry A\[i\]\[j\] | Closing price of stock j on day i (USD) |
| Rank | 4 |
| Nullity | 1 |

We have also used Indian stock data:
<img width="533" height="251" alt="image" src="https://github.com/user-attachments/assets/01254812-d24a-4da2-b11b-cca992b6bd11" />

---

## Output Interpretation Guide

| What you see | What it means |
|---|---|
| Rising solid line | Genuine upward trend after noise removal |
| Falling solid line | Genuine downward trend |
| Solid ≈ Dotted | Very little noise; clean underlying signal |
| Solid far from Dotted | Significant noise corrected by the model |
| ◆ above last point | Least Squares predicts a price increase |
| ◆ below last point | Least Squares predicts a price decrease |
| Diverging dashed tails | Stocks have different predicted momentum |

---


---

## File Structure

```
stock_pipeline/
├── stock_pipeline.py    ← main application (run this)
├── requirements.txt     ← Python dependencies
├── sample_data.csv      ← example CSV for upload testing
├── screenshots/         ← add your screenshots here
└── README.md            ← this file
```

---
## Future Scope

This problem statement can be further extended and applied on real-time data from stock markets, where it scrapes data you wish to analyse and ity would provide insights for it 



Thank You 
