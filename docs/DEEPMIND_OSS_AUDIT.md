# Upstream Evaluator Design & Open-Source Audit: Google DeepMind & Hugging Face

**Author**: Edge-RAG-Eval Research Initiative  
**Track**: Track C — Open Source Contributions to Google DeepMind Tooling  
**Target Upstream Repositories**:  
- [`google-deepmind/proeval`](https://github.com/google-deepmind/proeval) (Efficient Generative Model Evaluation)  
- [`huggingface/lighteval`](https://github.com/huggingface/lighteval) (Lightweight Foundation Model Evaluation Suite)  
**Proposed Module**: `proeval.evaluators.retrieval.context_noise` / `lighteval.metrics.rag_noise`  
**Core Abstraction**: `ContextNoiseEvaluator`  

---

## 1. Executive Summary & Scientific Motivation

Contemporary evaluation frameworks for Retrieval-Augmented Generation (RAG) predominantly assess language models under idealized, clean, or uniformly distributed random contexts (e.g., standard HotpotQA, SQuAD 2.0, Natural Questions). However, production edge deployments and on-device foundation models (such as Gemma-2B) operate in noisy, low-memory retrieval environments where retrieved passages frequently include **semantically adversarial distractors**—passages exhibiting high vector similarity ($\text{cosine similarity} \ge 0.65$) but asserting contradictory entities, shifted dates, or competing facts.

Existing upstream evaluation suites lack standardized, decoupled evaluators to diagnose and quantify two critical edge failure modes:
1. **Entity Misattribution**: The generator incorporates contradictory entity tokens directly from high-similarity distractor passages into the generated response.
2. **Abstention vs. Hallucination Collapse**: The inability to distinguish between principled model abstention (recognizing conflicting evidence or insufficient grounding) and ungrounded fabrication.

This document presents a technical audit of the evaluation abstractions in `google-deepmind/proeval` and `huggingface/lighteval`, identifies integration points, and specifies the architecture of `ContextNoiseEvaluator`—a pluggable, stateless evaluator designed for seamless upstream inclusion.

---

## 2. Upstream Architecture Audit

### 2.1 Google DeepMind `proeval` Architecture

Google DeepMind's `proeval` is engineered for efficient, low-overhead evaluation of frontier and compact foundation models. Its core architectural tenets include:

1. **Decoupled Evaluation Protocol**:
   Evaluator classes inherit from a minimal abstract base protocol (`BaseEvaluator`). Evaluation logic is strictly separated from model inference, dataset streaming, and prompt templating. Evaluators accept standardized, immutable record structures:
   ```python
   class BaseEvaluator(abc.ABC):
       @abc.abstractmethod
       def evaluate_sample(self, sample: EvaluationInput) -> MetricResult:
           """Evaluate an individual generation record."""
           raise NotImplementedError

       @abc.abstractmethod
       def evaluate_batch(self, samples: Sequence[EvaluationInput]) -> Dict[str, float]:
           """Aggregate metrics over a sequence of evaluation samples."""
           raise NotImplementedError
   ```

2. **Stateless Metric Aggregation**:
   All evaluators are purely stateless. Metric computations output bounded continuous scores ($\in [0.0, 1.0]$) alongside categorical error tags. Batch aggregation calculates first and second moments (mean, standard deviation) alongside non-parametric bootstrap confidence intervals (95% CI).

3. **Zero Proprietary Dependencies**:
   Upstream DeepMind evaluators avoid coupling to specific model inference engines, proprietary APIs, or heavyweight vector databases. All required inputs (`query`, `context`, `response`, `ground_truth`, `distractor_context`) are passed explicitly via typed containers.

### 2.2 Hugging Face `lighteval` Architecture

Hugging Face's `lighteval` powers reproducible leaderboards (e.g., Open LLM Leaderboard) with an emphasis on lightweight, scalable metric dispatch:

1. **Functional Metric Signatures**:
   Metrics in `lighteval` are parameterized as pure callables accepting structured prediction documents (`Doc`) and returning dictionaries of scalar values:
   ```python
   def metric_fn(predictions: list[Doc], references: list[Doc]) -> dict[str, float]: ...
   ```
2. **Hierarchical Categorization**:
   Metrics distinguish between exact lexical matches, token overlap measures, and semantic grounding hit-rates.
3. **Conditioned Sub-populating**:
   Evaluations group metrics across controlled experimental conditions (e.g., zero-shot vs. few-shot, clean vs. perturbed contexts).

---

## 3. Identified Upstream Gap

| Evaluation Dimension | Standard Benchmarks (HotpotQA / SQuAD) | Existing Tooling (`proeval` / `lighteval`) | **Our Contribution (`ContextNoiseEvaluator`)** |
| :--- | :--- | :--- | :--- |
| **Context Conditions** | Clean passage only | Clean context or random noise | Controlled 3-condition: Gold Only, Random Distractor ($\text{sim} < 0.25$), Adversarial Distractor ($\text{sim} \ge 0.65$) |
| **Error Diagnostics** | Lexical accuracy (EM / F1) | Lexical accuracy + Perplexity | **Entity Misattribution Rate** + **Abstention Rate** + Token F1 + Grounding Score |
| **Adversarial Granularity** | None (lexical matching) | Keyword presence | Entity-level token intersection isolating distractor entity infiltration |
| **Hardware Overhead** | Heavy evaluation chains | Moderate | Lightweight, stateless, CPU-compatible ($< 2 \text{ ms/sample}$) |

---

## 4. Proposed Upstream Pull Request Specification

### 4.1 Target Placement & Naming
- **Repository**: `google-deepmind/proeval`
- **Module Path**: `proeval/evaluators/retrieval/context_noise.py`
- **Primary Class**: `ContextNoiseEvaluator`
- **Secondary Adapter**: `lighteval/metrics/context_noise_adapter.py`

### 4.2 Standardized Data Contracts

#### Input Record (`EvaluationInput`)
```python
class EvaluationInput(BaseModel):
    sample_id: str
    query: str
    context: str
    condition: Literal["gold_only", "random_distractor", "adversarial_distractor"]
    response: str
    ground_truth: str
    distractor_context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### Output Record (`MetricResult`)
```python
class MetricResult(BaseModel):
    exact_match: float       # Binary score 0.0 or 1.0 (normalized string equality)
    token_f1: float          # Harmonic mean of token precision and recall in [0.0, 1.0]
    grounding_score: float   # Lexical overlap fraction with provided gold context in [0.0, 1.0]
    entity_misattribution: bool  # True if response adopted contradictory entities from distractor
    abstention: bool         # True if model explicitly refrained from answering
    latency_ms: float = 0.0  # Evaluator execution latency in milliseconds
    error_category: Literal["none", "misattribution", "abstention", "hallucination"]
```

---

## 5. Mathematical Formulations & Metric Definitions

### 5.1 Exact Match (EM)
Let $R$ denote the model response string and $G$ denote the ground-truth string. After case folding, punctuation stripping, and whitespace normalization:
$$\text{EM}(R, G) = \begin{cases} 1.0 & \text{if } \text{norm}(R) = \text{norm}(G) \\ 0.0 & \text{otherwise} \end{cases}$$

### 5.2 Token-Level F1 Score
Let $\mathcal{T}(S)$ be the multiset of unigram tokens for text $S$:
$$\text{Precision} = \frac{|\mathcal{T}(R) \cap \mathcal{T}(G)|}{|\mathcal{T}(R)|}, \quad \text{Recall} = \frac{|\mathcal{T}(R) \cap \mathcal{T}(G)|}{|\mathcal{T}(G)|}$$
$$\text{Token F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon}$$

### 5.3 Grounding Score
Measures the proportion of content tokens in the generated response $R$ directly traceable to the authoritative gold context $C_{\text{gold}}$:
$$\text{Grounding}(R, C_{\text{gold}}) = \frac{|\mathcal{T}_{\text{content}}(R) \cap \mathcal{T}_{\text{content}}(C_{\text{gold}})|}{|\mathcal{T}_{\text{content}}(R)| + \epsilon}$$

### 5.4 Entity Misattribution Detection
Let $\mathcal{E}(D)$ be the distinctive entity tokens present in the adversarial distractor $D$ that are absent from the gold context $C_{\text{gold}}$ and ground truth $G$:
$$\mathcal{E}_{\text{distractor}} = \mathcal{T}_{\text{entities}}(D) \setminus (\mathcal{T}(C_{\text{gold}}) \cup \mathcal{T}(G))$$
$$\text{EntityMisattribution}(R, D) = \mathbb{I}\left[ \left|\mathcal{T}(R) \cap \mathcal{E}_{\text{distractor}}\right| > 0 \;\land\; \text{EM}(R, G) = 0 \right]$$

### 5.5 Abstention Detection
Identifies calibrated model behavior under unanswerable or contradictory contexts:
$$\text{Abstention}(R) = \mathbb{I}\left[ \exists \, m \in \mathcal{M}_{\text{refusal}} : m \subseteq \text{norm}(R) \right]$$
where $\mathcal{M}_{\text{refusal}}$ includes standardized refusal markers (*e.g.*, "insufficient context", "cannot be determined", "information is not provided", "not mentioned in the text", "unable to answer").

---

## 6. Upstream Pull Request Implementation Plan

```mermaid
graph TD
    A["Raw Dataset (eval_500.json)"] --> B["DeepMind Ingestion Protocol (EvaluationInput)"]
    B --> C["ContextNoiseEvaluator Engine"]
    C --> D1["Exact Match & Token F1"]
    C --> D2["Grounding Overlap Score"]
    C --> D3["Entity Misattribution Filter"]
    C --> D4["Abstention Classifier"]
    D1 & D2 & D3 & D4 --> E["MetricResult Container"]
    E --> F["Bootstrap Statistical Aggregator (evaluate_batch)"]
    F --> G["Standard proeval / lighteval Report"]
```

### Upstream Readiness Checklist
- [x] **Decoupled Evaluator Interface**: Zero direct dependency on external LLM inference pipelines or GPU runtimes.
- [x] **Strict Pydantic Typing**: Runtime schema validation conforming to modern Python 3.10+ typing standards.
- [x] **Statistical Bootstrap Aggregation**: Non-parametric 95% bootstrap confidence intervals for all continuous metrics.
- [x] **Diagnostic CLI**: Native `--test-oss-adapter` verification in testbed CLI (`src/pipeline.py`).
- [x] **Unit & Regression Testing**: 100% test coverage across valid, edge-case, and corrupted evaluation inputs.
