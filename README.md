# 🔐 WHT Watermark Lab

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)](.)

> **A Complete Implementation of:**
> 
> *"A Novel Dual Color Image Watermarking Algorithm Using Walsh–Hadamard Transform with Difference-Based Embedding Positions"*
>
> 📄 **Symmetry 2026**, 18, 65

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
cd watermark-app
pip install -r requirements.txt
```

### 2️⃣ Run the Server
```bash
python app.py
```

### 3️⃣ Open in Browser
```
🌐 http://localhost:5000
```

---

## 📁 Project Structure
```
watermark-app/
│
├── 🎯 app.py                 # Flask backend — API routes & server logic
├── 🔬 watermark.py           # Core WHT watermarking algorithm
├── 📦 requirements.txt       # Python dependencies
│
├── 🎨 static/
│   ├── style.css             # Modern UI styling & animations
│   └── script.js             # Frontend fetch logic & interactions
│
└── 📄 templates/
    └── index.html            # Main interactive UI page
```

---

## ⚙️ Algorithm Pipeline

| 🔢 Step | 📋 Description | 🎯 Purpose |
|:---:|---|---|
| **1** | 🌈 RGB channel separation | Decompose color image into channels |
| **2** | 🔲 4×4 non-overlapping block partitioning | Divide image into manageable blocks |
| **3** | 🧠 Entropy-based block selection | Identify visually robust regions (visual + edge entropy) |
| **4** | 🔐 Logistic chaotic encryption | Encrypt watermark (μ=4, x₀=0.398) |
| **5** | 🔄 Walsh-Hadamard Transform | Apply WHT (H₄ left-multiply) on blocks |
| **6** | 📊 Difference-based coefficient selection | Pick 4 smallest coefficient pairs |
| **7** | 💾 Quantization embedding | Insert watermark bits (T=8): avg±T/2 |
| **8** | ⚡ Inverse WHT + recombination | Reconstruct watermarked image |

---

## 🔌 API Endpoints

### 📤 Embed Watermark
```http
POST /embed
Content-Type: multipart/form-data

Parameters:
  - cover_image (file)       # Host image
  - watermark_image (file)   # Watermark to embed
  
Response:
  {
    "watermarked_image": "base64_encoded_image",
    "psnr": 38.45,
    "ssim": 0.95,
    "message": "✅ Watermark embedded successfully"
  }
```

### 🔍 Extract Watermark
```http
POST /extract
Content-Type: multipart/form-data

Parameters:
  - image (file)  # Watermarked or attacked image
  
Response:
  {
    "extracted_watermark": "base64_encoded_image",
    "nc": 0.92,
    "ber": 0.02,
    "message": "✅ Watermark extracted successfully"
  }
```

### 💥 Apply Attack Simulation
```http
POST /attack
Content-Type: multipart/form-data

Parameters:
  - image (file)           # Watermarked image
  - attack_type (string)   # gaussian_noise | jpeg | crop | rotate | scale
  - intensity (float)      # Attack strength (0.0–1.0)
  
Response:
  {
    "attacked_image": "base64_encoded_image",
    "attack_applied": "jpeg_compression",
    "message": "✅ Attack simulation completed"
  }
```

### 📊 Get All Metrics
```http
GET /metrics

Response:
  {
    "psnr": 38.45,
    "ssim": 0.95,
    "nc": 0.92,
    "ber": 0.02,
    "timestamp": "2026-03-17 14:32:45"
  }
```

---

## 📈 Performance Metrics

| 📊 Metric | 🎯 Target | 📝 Description |
|---|---|---|
| **PSNR** | ≥ 35 dB | Peak Signal-to-Noise Ratio — image quality preservation |
| **SSIM** | ≥ 0.93 | Structural Similarity Index — perceptual fidelity |
| **NC** | ≥ 0.90 | Normalized Correlation — watermark robustness |
| **BER** | ≈ 0.00 | Bit Error Rate — extraction accuracy |

---

## 🛡️ Supported Attacks

Comprehensive robustness testing against real-world attacks:

| 🎯 Attack Type | 🔧 Parameters | 🎨 Visual Impact |
|---|---|---|
| 🔊 **Gaussian Noise** | Variance: 0.001–0.1 | Adds random pixel noise |
| 📦 **JPEG Compression** | Quality: 10–95 | Lossy compression artifacts |
| ✂️ **Cropping** | Area: 5–40% removed | Removes image regions |
| 🔄 **Rotation** | Angle: 1–30° | Geometric transformation |
| 📏 **Scaling** | Factor: 0.5×–2.0× | Resize up or down |

---

## 📋 Requirements
```txt
Python 3.8+
├── Flask 2.0+              # Web framework
├── NumPy 1.20+             # Numerical computing
├── OpenCV 4.5+             # Computer vision
├── Pillow 8.0+             # Image processing
└── scikit-image 0.18+      # Advanced image algorithms
```

### Install All at Once
```bash
pip install Flask>=2.0 numpy>=1.20 opencv-python>=4.5 Pillow>=8.0 scikit-image>=0.18
```

---

## 🎓 How to Use

### Step 1: Prepare Images
- **Cover Image**: Your host image (RGB, any size)
- **Watermark**: Binary or grayscale watermark (typically 64×64 or 128×128)

### Step 2: Embed Watermark
```bash
curl -X POST http://localhost:5000/embed \
  -F "cover_image=@path/to/cover.jpg" \
  -F "watermark_image=@path/to/watermark.png"
```

### Step 3: Test Robustness
```bash
# Apply attack
curl -X POST http://localhost:5000/attack \
  -F "image=@watermarked.png" \
  -F "attack_type=jpeg" \
  -F "intensity=0.8"

# Extract watermark
curl -X POST http://localhost:5000/extract \
  -F "image=@attacked.png"
```

### Step 4: Analyze Results
```bash
curl http://localhost:5000/metrics
```

---

## 🔬 Technical Details

### Why Walsh-Hadamard Transform?

✨ **WHT Advantages:**
- ⚡ Fast computation (recursive structure)
- 🔐 Frequency domain analysis without floating-point overhead
- 🎯 Excellent energy compaction in discrete domains
- 🛡️ Robust to common image attacks

### Entropy-Based Block Selection

🧠 **Smart Embedding:**
- Analyzes both visual entropy and edge entropy
- Selects high-entropy blocks (textured regions)
- Avoids smooth areas (more visible artifacts)
- Balances imperceptibility & robustness

### Difference-Based Embedding

📊 **Why Coefficient Pairs?**
- Exploits correlations between neighboring coefficients
- Uses 4 smallest coefficients (minimal perceptual impact)
- Quantization maintains stability under compression

---

## 🎨 Web UI Features

✅ **Interactive Dashboard**
- Real-time image preview
- Drag-and-drop file upload
- Live metric visualization
- Attack simulation playground
- One-click watermark extraction

---

## 📚 References
```bibtex
@article{WHT_Watermark_2026,
  title={A Novel Dual Color Image Watermarking Algorithm Using 
         Walsh–Hadamard Transform with Difference-Based Embedding Positions},
  journal={Symmetry},
  year={2026},
  volume={18},
  number={65},
  doi={10.3390/sym18030065}
}
```

---

## 🤝 Contributing

Found a bug? Have an idea? 

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔀 Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author & Support

**Created with ❤️ for digital image watermarking research**

For questions or support:
- 📧 Open an issue on GitHub
- 💬 Check existing discussions
- 📖 Review the documentation

---

<div align="center">

### ⭐ Found this useful? Please consider starring the repository!

**Built with Python 🐍 | Flask 🌶️ | OpenCV 📷**

</div>