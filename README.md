# Edge-RAG-Eval

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A standalone, headless Python research testbed designed to benchmark edge foundation models (e.g., Gemma-2B) under noisy context injection and modular CALM (Composition to Augment Language Models) cross-attention composition.

---

## Architecture & Directory Layout

```
Edge-RAG-Eval/
├── configs/
│   └── baseline_config.yaml         # Baseline experiment configurations
├── data/
│   ├── raw/                         # Raw benchmark corpora & evaluation sets
│   └── processed/                   # Tokenized / cached tensor datasets
├── notebooks/                       # Exploratory analysis & attention visualization
├── paper/                           # LaTeX paper drafts & ablation plots
├── src/
│   ├── __init__.py
│   ├── pipeline.py                  # CLI entrypoint & diagnostic smoke test runner
│   ├── data/
│   │   ├── __init__.py
│   │   └── distractor_sampler.py    # Noisy context sampler & distractor injector
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── grounding.py             # Grounding precision, cosine similarity, hit rates
│   ├── models/
│   │   ├── __init__.py
│   │   ├── inference.py             # SentenceTransformer embedding client (MiniLM)
│   │   └── calm_mini.py             # Modular cross-attention CALM composition adapter
│   └── visualization/
│       └── __init__.py              # Metric plotting & diagnostic charts
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py             # Pytest test suite for embedding client & CLI
├── pyproject.toml                   # Modern PEP-621 build configuration
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Clean ML-oriented gitignore
└── README.md
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10 - 3.14)
- CPU, Apple Silicon (MPS), or NVIDIA GPU (CUDA)

### 2. Virtual Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in editable mode
pip install -e .
```

---

## CLI Execution & Smoke Test

Edge-RAG-Eval provides a CLI entrypoint for running diagnostic health checks, hardware inspection, and embedding smoke tests.

### Run Diagnostic Smoke Test
```powershell
python -m src.pipeline --test
```

#### Expected Output
1. **System Diagnostics**:
   - Host Operating System & architecture
   - Python & PyTorch runtime versions
   - CUDA / MPS hardware acceleration detection
   - GPU device name and allocated VRAM
2. **Embedding Verification**:
   - Loads `sentence-transformers/all-MiniLM-L6-v2`
   - Encodes a sample reference string: `"Edge-RAG-Eval: Decoupled retrieval grounding testbed."`
   - Verifies dense vector output shape is strictly `(1, 384)`
   - Computes cosine similarity with semantic query: `"Benchmarking retrieval systems under noise."`
   - Measures end-to-end inference latency (ms/sample)
   - Outputs a clean summary table via `rich`
   - Exits with return code `0`

### Run Deterministic Gemma-2B Generation Test
```powershell
python -m src.pipeline --test-generation
```

#### Expected Output
1. **Ollama / Backend Reachability**:
   - Queries local Ollama daemon at `http://localhost:11434` for model `gemma:2b`
   - If Ollama is unavailable, displays start-up instructions and falls back gracefully to local deterministic engine
2. **Determinism & Reproducibility Verification**:
   - Executes two consecutive generation passes under `temperature=0.0` and `seed=42`
   - Prompt: `"State the chemical formula for water and briefly explain its molecular geometry in one sentence."`
   - Verifies 100% bit-for-bit exact string equivalence (`Pass: Determinism Verified`)
   - Logs tokens generated, throughput (tokens/sec), and latency in a clean `rich` table

### Run CALM Modular Cross-Attention Architecture Audit
```powershell
python -m src.pipeline --test-calm
```

#### Expected Output
1. **Parameter Efficiency & Budget Audit**:
   - Anchor hidden dimension: `2048` (Gemma-2B, frozen)
   - Augment dimension: `384` (`sentence-transformers/all-MiniLM-L6-v2`)
   - Cross-attention projection dimension: `512` (8 heads $\times$ 64)
   - Base model parameters: `2,000,000,000` (frozen, 0% trained)
   - Trainable adapter parameters: `2,490,369` (passes $<5\text{M}$ budget)
   - Trainable capacity fraction: `0.124%` (passes $<1.0\%$ budget)
   - Learned gating scalar ($\gamma$): Initialized strictly to `0.0000`
2. **Functional & Gradient Flow Verification**:
   - Anchor shape `(2, 16, 2048)` and augment shape `(2, 32, 384)`
   - Output shape verified as `(2, 16, 2048)`
   - Zero degradation at init: `torch.equal(out, anchor)` passes
   - Gradient isolation: Base tensors have `grad is None` while gradients flow to $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$ and $\gamma$
   - Formally documented in [CALM Replication Memo](paper/CALM_REPLICATION_MEMO.md)

### Custom Config Execution
```powershell
python -m src.pipeline --config configs/baseline_config.yaml
```

---

## Running Unit & Integration Tests

The test suite validates model initialization, output dimensionality, batched inference, metric computations, Ollama mock parsing, deterministic generation, and CALM cross-attention gating:

```powershell
pytest tests/
```

---

## Core Modules Overview

- **`src.models.calm_mini.CALMCrossAttentionLayer`**:
  Modular cross-attention layer replicating CALM (Bansal et al., Google DeepMind, 2024, arXiv:2401.02412) with zero-initialized gating ($\gamma = 0.0$) ensuring zero baseline degradation at initialization.
- **`src.models.calm_mini.CALMCompositionalWrapper`**:
  Compositional container integrating frozen base LLM representations with trainable cross-attention adapters and parameter audit tools.
- **`src.models.inference.GemmaInferenceClient`**:
  Deterministic text generation engine wrapping Ollama REST API and Hugging Face pipelines with explicit context budgeting and reproducible decoding (`temperature=0.0`, `seed=42`).
- **`src.models.inference.GenerationResult`**:
  Validated Pydantic model encapsulating response text, token counts, throughput, latency, and ISO timestamps.
- **`src.models.inference.EmbeddingClient`**:
  High-throughput, device-agnostic sentence embedding wrapper with automatic GPU/MPS/CPU dispatch and L2 normalization.
- **`src.data.distractor_sampler.DistractorSampler`**:
  Configurable context corruption module designed to measure retrieval grounding degradation across varying signal-to-noise ratios.
- **`src.metrics.grounding`**:
  Standardized vector similarity and retrieval hit-rate evaluation functions.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
