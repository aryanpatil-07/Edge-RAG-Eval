# CALM Replication Memo: Modular Cross-Attention Composition for Noise-Resilient Edge RAG

**Authors:** ML Systems & Evaluation Research Testbed (`Edge-RAG-Eval`)  
**Reference Paper:** *Composition to Augment Language Models* (CALM), Bansal, Samanta, Gupta, Vashishth et al., Google DeepMind India (arXiv:2401.02412, 2024).  
**Target Architecture:** Gemma-2B Anchor Model + `sentence-transformers/all-MiniLM-L6-v2` Retrieval Embeddings.

---

## 1. Executive Summary & Mathematical Problem Formulation

### 1.1 The In-Context Concatenation Bottleneck
In standard Retrieval-Augmented Generation (RAG), external knowledge retrieved from vector stores or sparse indices is injected directly into the LLM context window as raw text tokens:

$$\mathbf{X}_{\text{input}} = [\mathbf{D}_1, \mathbf{D}_2, \dots, \mathbf{D}_K \,;\, \mathbf{Q}]$$

While conceptually straightforward, this architecture exhibits severe degradation in edge-scale foundation models (such as Gemma-2B):
1. **Self-Attention Pollution:** Every token in the prompt attends to all other tokens via standard multi-head self-attention:
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
   When $K$ contains distractor passages or noisy retrieved contexts, the attention mass of compact models is diluted, leading to entity misattribution and hallucination.
2. **Context Window Saturation:** Edge devices with constrained SRAM and unified memory budgets suffer from quadratic $\mathcal{O}(L^2)$ KV-cache explosion when long context chunks are ingested.

---

## 2. The CALM Composition Architectural Blueprint

Rather than polluting the generative context window with raw tokens, CALM decouples knowledge assimilation into an **Anchor Model** and an **Augmenting Representation**:

```
+-------------------------------------------------------------+
|                      Retrieved Documents                    |
+-------------------------------------------------------------+
                               |
                               v
               [Retriever / Context Encoder]
                               |
                               v
                     H_aug in R^(B x L_aug x 384)
                               |
                               +-------------------+
                                                   | Key, Value
                                                   v
   H_anchor in R^(B x L x 2048) ---------> [Cross-Attention Layer]
        (Frozen Gemma-2B)            Query         |
               |                                   | AttentionOutput
               |                                   v
               |                             ( * gamma = 0 )
               |                                   |
               +---------------------------------> (+) Residual Addition
                                                   |
                                                   v
                                     H_tilde in R^(B x L x 2048)
```

### 2.1 Formal Cross-Attention Formulation
1. **Anchor Model Freeze:**
   The base model parameters $\theta_{\text{anchor}}$ (Gemma-2B, $\sim 2\text{B}$ parameters) remain strictly **frozen** ($\nabla_{\theta_{\text{anchor}}} \mathcal{L} = \mathbf{0}$).
2. **Augmenting Representation Extraction:**
   An augmenting representation $\mathbf{H}_{\text{aug}} \in \mathbb{R}^{B \times L_{\text{aug}} \times D_{\text{aug}}}$ ($D_{\text{aug}} = 384$) is extracted from the retrieved context.
3. **Cross-Attention Projection:**
   Lightweight projection matrices map anchor queries and augmenting keys/values into a shared attention manifold ($d_k$):
   $$\mathbf{Q} = \mathbf{H}_{\text{anchor}} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}_{\text{aug}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}_{\text{aug}} \mathbf{W}_V$$
   where $\mathbf{W}_Q \in \mathbb{R}^{D_{\text{anchor}} \times D_{\text{attn}}}$, $\mathbf{W}_K \in \mathbb{R}^{D_{\text{aug}} \times D_{\text{attn}}}$, $\mathbf{W}_V \in \mathbb{R}^{D_{\text{aug}} \times D_{\text{attn}}}$, and $\mathbf{W}_O \in \mathbb{R}^{D_{\text{attn}} \times D_{\text{anchor}}}$.
4. **Zero-Initialized Gated Residual Connection:**
   $$\text{AttnOutput} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V} \mathbf{W}_O$$
   $$\tilde{\mathbf{H}}_{\text{anchor}} = \mathbf{H}_{\text{anchor}} + \gamma \cdot \text{AttnOutput}$$
   where $\gamma \in \mathbb{R}$ is a learned scalar gate **initialized strictly to $0.0$**.

### 2.2 Critical Properties of Zero-Initialized Gating ($\gamma = 0.0$)
- **Zero Initial Degradation:** Because $\gamma = 0.0$ at step 0, $\tilde{\mathbf{H}}_{\text{anchor}} \equiv \mathbf{H}_{\text{anchor}}$. The base model's pre-trained language modeling capabilities are guaranteed zero disruption before optimization begins.
- **Adaptive Knowledge Admission:** The gradient $\frac{\partial \mathcal{L}}{\partial \gamma} = \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{H}}} \cdot \text{AttnOutput}$ allows the model to monotonically admit augmenting context only as the cross-attention projections align.

---

## 3. Parameter Efficiency Audit

| Component | Dimensions | Parameter Formula | Param Count | Trainable? |
|---|---|---|---|---|
| Anchor Base Model (Gemma-2B) | $D_{\text{anchor}} = 2048$ | Full transformer weights | $\sim 2,000,000,000$ | **FROZEN (0%)** |
| Query Projection $\mathbf{W}_Q$ | $2048 \times 512$ | $D_{\text{anchor}} \times D_{\text{attn}}$ | $1,048,576$ | **Yes** |
| Key Projection $\mathbf{W}_K$ | $384 \times 512$ | $D_{\text{aug}} \times D_{\text{attn}}$ | $196,608$ | **Yes** |
| Value Projection $\mathbf{W}_V$ | $384 \times 512$ | $D_{\text{aug}} \times D_{\text{attn}}$ | $196,608$ | **Yes** |
| Output Projection $\mathbf{W}_O$ | $512 \times 2048$ | $D_{\text{attn}} \times D_{\text{anchor}}$ | $1,048,576$ | **Yes** |
| Gating Scalar $\gamma$ | $1$ | Scalar parameter | $1$ | **Yes** |
| **Total Adapter Parameters** | — | — | **2,490,369** ($\sim 2.49\text{M}$) | **Trainable** |

### Efficiency Ratio
$$\text{Trainable Ratio} = \frac{2,490,369}{2,000,000,000 + 2,490,369} \approx \mathbf{0.124\%}$$

- **Compliance with Budget:** $2.49\text{M} \ll 5.0\text{M}$ parameter ceiling.
- **Capacity Ratio:** $0.124\% \ll 1.0\%$ total parameter capacity.

---

## 4. Core Research Hypothesis

> **Central Hypothesis:**  
> *Does modular cross-attention gating isolate distractor noise better than raw in-context token concatenation under noisy retrieval conditions?*

### Theoretical Justification:
1. **Bounded Cross-Attention Manifold:** In concatenated RAG, noisy distractors shift intermediate activations across all self-attention heads at every layer. Under CALM, retrieved context is isolated to the cross-attention Key-Value projection space.
2. **Gated Attention Damping:** If retrieved documents contain high-entropy distractor noise, the scalar gate $\gamma$ and cross-attention softmax scores can selectively down-weight spurious tokens without perturbing the anchor query's causal self-attention path.

---

## 5. Verification Protocol

The implementation is verified via:
1. **Dimension Invariance:** $\text{shape}(\tilde{\mathbf{H}}) = (B, L, 2048)$ for any context length $L_{\text{aug}}$.
2. **Zero-Init Equivalence:** $\tilde{\mathbf{H}} = \mathbf{H}_{\text{anchor}}$ when $\gamma = 0$.
3. **Gradient Isolation:** $\nabla_{\mathbf{H}_{\text{anchor}}} \mathcal{L} = \text{None}$, while $\nabla_{\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O, \gamma} \mathcal{L} \neq \text{None}$.
4. **CLI Diagnostic:** `python -m src.pipeline --test-calm` (Exit Code 0).

---

## 6. Bibliography
- Bansal, H., Samanta, B., Gupta, V., Vashishth, S., et al. (2024). *CALM: Composition to Augment Language Models*. Google DeepMind India. arXiv:2401.02412.
- Team, G., et al. (2024). *Gemma: Open Models Based on Gemini Research and Technology*. Google DeepMind. arXiv:2403.08295.
