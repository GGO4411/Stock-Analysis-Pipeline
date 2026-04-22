# 📈 LA-QUANT: Stock Market Linear Algebra Pipeline

**Course:** UE24MA241B — Linear Algebra and Its Applications  
**Institution:** PES University, Dept. of CSE  

---

## 🚀 Overview

LA-QUANT is a Python-based application that applies a **10-stage Linear Algebra pipeline** to stock market data.

It transforms raw price data into:
- 📊 Denoised market trends  
- 📉 Dimensional insights (rank, eigenvalues)  
- 🔮 Future price predictions  

The system uses concepts like **SVD, eigenanalysis, least squares, projections, and orthogonalization** to model real-world financial data.

---

## 🎯 Objectives

- Identify independent market trends using **rank analysis**
- Remove noise via **orthogonal projections**
- Predict future prices using **least squares regression**
- Compress datasets using **Singular Value Decomposition (SVD)**

---

## 🧠 Pipeline Architecture

The pipeline consists of 10 stages:

| Stage | Name | Concept | Purpose |
|------|------|--------|--------|
| S1 | Normalize | Mean Centering | Remove price bias |
| S2 | LU Decomposition | Factorization | Efficient solving |
| S3 | Rank & Nullity | Linear Independence | Detect redundancy |
| S4 | Basis Selection | Vector Spaces | Minimal representation |
| S5 | Gram-Schmidt | Orthogonalization | Independent trends |
| S6 | Projection | Subspaces | Noise removal |
| S7 | Least Squares | Regression | Price prediction |
| S8 | Eigenanalysis | PCA | Trend discovery |
| S9 | SVD Compression | Low-rank approx | Data compression |
| S10 | Final Model | Reconstruction | Denoised + forecast |

---

## 📊 Input Data

- Format: CSV file  
- Structure:
  - Rows → Days  
  - Columns → Stocks  

Example:
```
Day,AAPL,GOOG,MSFT
1,182,140,375
2,185,142,378
```

- Default dataset: **5 stocks × 10 days**

---

## ⚙️ Features

- 📂 Upload custom CSV datasets  
- 📉 Interactive visualizations for each stage  
- 🎚️ Sliders for:
  - Future prediction (S7)
  - SVD compression level (S9)  
- 🖥️ Bloomberg-terminal-style GUI  
- 🔄 Full pipeline execution with one click  

---

## 🔍 Key Concepts Used

- Matrix Representation  
- LU Factorization  
- Rank–Nullity Theorem  
- Gram-Schmidt Orthogonalization  
- Orthogonal Projection  
- Least Squares Approximation  
- Eigenvalues & Eigenvectors  
- Singular Value Decomposition (SVD)  

---

## 🧪 Example Insights

- High correlation between stocks → **low rank**
- Dominant eigenvalue → **market-wide trend**
- Low-rank SVD → **strong compression with minimal loss**
- Projection → **noise filtering**

---

## 🛠️ Installation

```bash
pip install numpy scipy matplotlib
```

---

## ▶️ Running the Application

```bash
python stock_pipeline.py
```

---

## 🖱️ How to Use

1. Select any stage (S1–S10) from sidebar  
2. View:
   - Left → mathematical outputs  
   - Right → visual graphs  
3. Use:
   - S7 slider → future prediction  
   - S9 slider → adjust compression  
4. Upload CSV via **LOAD CSV**
5. Click **EXECUTE FULL PIPELINE** for final output  

---

## 📈 Final Output

The final model (S10) provides:
- ✔️ Denoised stock price curves  
- ✔️ Comparison with raw data  
- ✔️ Next-day predictions  

---

## 🎓 Viva Tips

Explain each stage as:

> **Concept → Why → Outcome**

Example:
- **Least Squares** → No exact solution → Best-fit prediction  
- **SVD** → Reduce dimensions → Preserve trends  

---

## 📦 Project Structure (Suggested)

```
LA-QUANT/
│── stock_pipeline.py
│── data/
│── README.md
│── requirements.txt
```

---

## 👨‍💻 Author

PES University — CSE  
Linear Algebra Mini Project  
