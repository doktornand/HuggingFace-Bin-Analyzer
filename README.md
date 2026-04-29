# HuggingFace Bin Analyzer 🔬

> Progressive deep analysis of HuggingFace `.bin` model files — architecture reconstruction, tensor statistics, anomaly detection, performance estimation, and text report generation.

HuggingFace Bin Analyzer is a Python tool for inspecting PyTorch `.bin` model checkpoints as produced by the HuggingFace ecosystem. It loads each shard with `torch.load(..., weights_only=True)`, reconstructs the model architecture from tensor names and `config.json`, performs statistical analysis on weight distributions, detects anomalies (dead neurons, saturated weights, irregular variance), estimates theoretical performance, and generates a structured text report — all from a single Python class.

---

## ✨ Features

### 🔍 Level 1 — Structural Analysis
- Discovers all `.bin` files in a model directory
- Loads each checkpoint safely with `map_location='cpu'` and `weights_only=True`
- Reads `config.json` and `tokenizer_config.json` if present
- Reports: file count, total size (MB), per-file tensor count, and first 10 tensor names per file
- Stores a lightweight tensor registry (shape, dtype, size, source file) for all subsequent analyses

### 🔬 Level 2 — Tensor Analysis
- Classifies all tensors by role: `embedding`, `attention`, `mlp`, `normalization`, `output`, `other`
- Builds dtype distribution and size distribution across the full tensor inventory
- Identifies the top 10 largest tensors by memory footprint
- Per-tensor deep analysis (on a configurable sample):
  - Shape, dtype, total parameter count, memory footprint (MB)
  - Statistics: mean, std, min, max, zero percentage
  - Sparsity detection: flags tensors where >10% of values are zero
  - Quantization pattern detection: flags tensors where unique values < 10% of total values, with compression ratio

### 🏗️ Level 3 — Architecture Reconstruction
- Infers number of layers, hidden size, number of attention heads, intermediate size, and vocab size directly from tensor shapes
- Cross-validates inferred values against `config.json` fields
- Supports naming conventions from GPT-2, Llama/Mistral (`layers.N`), and BERT-family (`h.N`, `transformer.h.N`) models
- Builds per-layer breakdown: tensor list and component classification (attention / MLP / normalization)
- Attention layer analysis: classifies each tensor as `query`, `key`, `value`, or `output` projection
- Parameter distribution by component (embedding / attention / MLP / normalization / output) and by layer, with efficiency ratios
- Embedding analysis: separates word, positional, and token type embeddings; computes average cosine distance and diversity score over a 50-embedding sample
- Architectural pattern detection with confidence scores: Standard Transformer, GPT-like, BERT-like, Quantized
- Model topology graph via NetworkX: sequential layer nodes and edges, graph density

### 🔬 Level 4 — Advanced Patterns
- **Weight distribution analysis**: global statistics (mean, std, skewness, kurtosis, entropy), per-tensor distribution type classification (normal / skewed / heavy-tailed / light-tailed via scipy normtest), outlier detection via Z-score (threshold = 3σ)
- **Anomaly detection** on a sample of up to 15 tensors:
  - Dead neurons: rows that are entirely zero
  - Saturated weights: values beyond 3σ exceeding 1% of tensor elements
  - Irregular patterns: variance below 1e-8 or above 100
  - Suspicious tensors: load errors flagged by name
- **Performance estimation**: theoretical FLOPs (attention O(n²d + nd²) + MLP O(nd_ff)), model memory, inference memory (×1.5), training memory (×4)
- **Optimization suggestions**: pruning (dead neurons), regularization (saturated weights), vocabulary reduction (embeddings > 30% of parameters), quantization (< 50% quantized tensors)

### 📄 Report Generation
- `generate_report(save_path=None)` produces a structured text report covering all four analysis levels
- Sections: model structure, tensor type/dtype distributions, top-5 largest tensors, per-tensor detailed stats, quantization and sparsity flags
- Optional save to `.txt` or `.md` file

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/doktornand/HuggingFace-Bin-Analyzer.git
cd HuggingFace-Bin-Analyzer
```

### 2. Install dependencies

```bash
pip install torch numpy matplotlib seaborn scipy scikit-learn networkx
```

> Python 3.8+ required. No GPU needed — all analysis runs on CPU.

### 3. Run the analyzer

```python
from HFAnalyser4 import HuggingFaceBinAnalyzer

analyzer = HuggingFaceBinAnalyzer("/path/to/model/directory")

# Run all four levels in sequence
structure = analyzer.analyze_structure()
tensors   = analyzer.analyze_tensors(sample_size=5)
arch      = analyzer.analyze_architecture()
advanced  = analyzer.analyze_advanced_patterns()

# Generate and save the full text report
report = analyzer.generate_report(save_path="report.txt")
print(report)
```

> **Important:** `analyze_architecture()` and `analyze_advanced_patterns()` depend on results stored by the previous levels. Always run them in order.

---

## 📁 Expected Model Directory Layout

```
my-model/
├── config.json                   # Model config (optional but recommended)
├── tokenizer_config.json         # Tokenizer config (optional)
├── pytorch_model.bin             # Single-shard model
│   — or —
├── pytorch_model-00001-of-00003.bin   # Multi-shard model
├── pytorch_model-00002-of-00003.bin
└── pytorch_model-00003-of-00003.bin
```

Single-file and multi-shard models are both fully supported. All `.bin` files in the directory are discovered and loaded automatically.

---

## 📊 Analysis Levels in Detail

### `analyze_structure()`
Returns file inventory, total size, model type and architecture from `config.json`, and a per-file summary including the first 10 tensor names.

### `analyze_tensors(sample_size=5)`
Returns layer type counts, dtype distribution, size distribution, top 10 largest tensors, and detailed per-tensor stats for the first `sample_size` tensors in the registry.

### `analyze_architecture()`
Returns reconstructed architecture parameters, per-layer component breakdown, attention tensor classification, parameter distribution by component and layer (with efficiency ratios), embedding quality metrics, detected architectural patterns with confidence scores, and a NetworkX topology graph.

### `analyze_advanced_patterns()`
Returns weight distribution statistics with distribution type per tensor, anomaly flags (dead neurons, saturated weights, irregular variance, load errors), theoretical FLOPs and memory estimates, and ranked optimization suggestions.

### `generate_report(save_path=None)`
Returns a formatted multi-section text report. If `save_path` is provided, the report is also written to disk.

---

## 🏗️ Supported Naming Conventions

The architecture reconstructor handles tensor name patterns from major model families:

| Pattern | Models |
|---|---|
| `model.layers.N.*` | Llama, Mistral, Qwen |
| `transformer.h.N.*` | GPT-2 (full path) |
| `h.N.*` | GPT-2 (short form) |
| `encoder.layer.N.*` | BERT, RoBERTa |
| `layers.N.*` | Generic Transformer |

Attention tensor classification supports: `q_proj` / `query`, `k_proj` / `key`, `v_proj` / `value`, `o_proj` / `out_proj`.

---

## ⚠️ Security Note

This tool uses `torch.load(..., weights_only=True)` to mitigate the arbitrary code execution risk inherent in the pickle-based `.bin` format. However, **loading `.bin` files always carries residual risk** when files come from untrusted sources. For a fully safe alternative, consider using the [SafeTensors format](https://github.com/huggingface/safetensors) and the companion [SafeTensors-Analyzer](https://github.com/doktornand/SafeTensors-Analyzer).

---

## 🛠️ Tech Stack

- **Python 3.8+**
- [PyTorch](https://pytorch.org/) — checkpoint loading and tensor operations
- [NumPy](https://numpy.org/) — numerical analysis
- [SciPy](https://scipy.org/) — statistical analysis (skewness, kurtosis, Z-score, normtest, cosine distance)
- [scikit-learn](https://scikit-learn.org/) — PCA, t-SNE for weight space analysis
- [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) — visualizations
- [NetworkX](https://networkx.org/) — model topology graph

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the project
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

**doktornand** — [github.com/doktornand](https://github.com/doktornand)

---

⭐ If this tool is useful to you, a star on the repo is appreciated!
