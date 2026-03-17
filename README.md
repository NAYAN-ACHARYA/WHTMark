# WHT Watermark Lab

A complete implementation of:
> **"A Novel Dual Color Image Watermarking Algorithm Using Walsh–Hadamard Transform with Difference-Based Embedding Positions"**
> Symmetry 2026, 18, 65

---

## Quick Start

### 1. Install dependencies

```bash
cd watermark-app
pip install -r requirements.txt
```

### 2. Run the server

```bash
python app.py
```

### 3. Open in browser

```
http://localhost:5000
```

---

## Project Structure

```
watermark-app/
├── app.py              # Flask backend — API routes
├── watermark.py        # Core WHT watermarking algorithm
├── requirements.txt    # Python dependencies
├── static/
│   ├── style.css       # UI styling
│   └── script.js       # Frontend fetch logic
└── templates/
    └── index.html      # Main UI page
```

---

## Algorithm Pipeline

| Step | Description |
|------|-------------|
| 1 | RGB channel separation |
| 2 | 4×4 non-overlapping block partitioning |
| 3 | Entropy-based block selection (visual + edge entropy) |
| 4 | Logistic chaotic encryption (μ=4, x₀=0.398) |
| 5 | Walsh-Hadamard Transform (H₄ left-multiply) |
| 6 | Difference-based coefficient pair selection (4 smallest) |
| 7 | Quantization embedding (T=8): avg±T/2 |
| 8 | Inverse WHT + channel recombination |

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/embed` | Embed watermark (cover + watermark images) |
| POST | `/extract` | Extract watermark from watermarked/attacked image |
| POST | `/attack` | Apply attack simulation |
| GET  | `/metrics` | Get all computed metrics |

---

## Metrics

- **PSNR** — Peak Signal-to-Noise Ratio (target ≥ 35 dB)
- **SSIM** — Structural Similarity Index (target ≥ 0.93)
- **NC** — Normalized Correlation (target ≥ 0.90)
- **BER** — Bit Error Rate (target ≈ 0.00)

---

## Supported Attacks

- Gaussian Noise (variance 0.001–0.1)
- JPEG Compression (quality 10–95)
- Cropping (5–40%)
- Rotation (1–30°)
- Scaling (0.5×–2.0×)

---

## Requirements

- Python 3.8+
- Flask, NumPy, OpenCV, Pillow, scikit-image
