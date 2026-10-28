"""CLI entrypoint and execution pipeline for Edge-RAG-Eval.

Usage:
    python -m src.pipeline --test
    python -m src.pipeline --test-generation
    python -m src.pipeline --test-calm
    python -m src.pipeline --config configs/baseline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from src.data.distractor_sampler import (
    CURATED_SEED_CORPUS,
    DistractorSampler,
    build_evaluation_dataset,
    compute_statistical_separation,
)
from src.models.calm_mini import (
    CALMCompositionalWrapper,
    CALMCrossAttentionLayer,
)
from src.models.inference import (
    EmbeddingClient,
    GemmaInferenceClient,
    GenerationResult,
    check_ollama,
)

logger = logging.getLogger("edge_rag_eval")


def get_system_diagnostics() -> Dict[str, str]:
    """Collect hardware and runtime environment diagnostic information."""
    cuda_available = torch.cuda.is_available()
    device_name = "N/A"
    total_vram = "N/A"

    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            total_vram = f"{total_vram_gb:.2f} GB"
        except Exception as e:
            device_name = f"Detected (Query error: {e})"

    return {
        "Operating System": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": "Yes" if cuda_available else "No (CPU/MPS Mode)",
        "GPU Device": device_name,
        "Total VRAM": total_vram,
    }


def display_results_rich(
    diagnostics: Dict[str, str],
    test_results: Dict[str, Any],
) -> None:
    """Render diagnostics and test outcomes using rich console tables."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # System Diagnostics Table
    diag_table = Table(title="Edge-RAG-Eval | System Diagnostics", show_header=True, header_style="bold cyan")
    diag_table.add_column("Property", style="bold white", width=22)
    diag_table.add_column("Value", style="green")
    for key, value in diagnostics.items():
        diag_table.add_row(key, str(value))

    # Test Results Table
    res_table = Table(title="Edge-RAG-Eval | Embedding Smoke Test", show_header=True, header_style="bold magenta")
    res_table.add_column("Check / Metric", style="bold white", width=26)
    res_table.add_column("Observed Value", style="cyan")
    res_table.add_column("Status", style="bold green", justify="center")

    res_table.add_row("Model Identifier", str(test_results["model_name"]), "READY")
    res_table.add_row("Execution Device", str(test_results["device"]), "ACTIVE")
    res_table.add_row("Embedding Dimension", str(test_results["dimension"]), "384")
    res_table.add_row("Output Shape (Ref)", str(test_results["ref_shape"]), "PASS (1, 384)")
    res_table.add_row("Output Shape (Query)", str(test_results["query_shape"]), "PASS (1, 384)")
    res_table.add_row("Cosine Similarity", f"{test_results['cosine_similarity']:.4f}", "COMPUTED")
    res_table.add_row("Inference Latency", f"{test_results['latency_ms']:.2f} ms", "BENCHMARKED")

    console.print()
    console.print(Panel.fit("[bold green]Edge-RAG-Eval Testbed Initialized Successfully[/bold green]", border_style="green"))
    console.print(diag_table)
    console.print()
    console.print(res_table)
    console.print()


def display_results_ascii(
    diagnostics: Dict[str, str],
    test_results: Dict[str, Any],
) -> None:
    """Fallback plain-text ASCII renderer if rich is not available."""
    print("=" * 65)
    print("Edge-RAG-Eval | System Diagnostics")
    print("=" * 65)
    for k, v in diagnostics.items():
        print(f"  {k:<22} : {v}")
    print("\n" + "=" * 65)
    print("Edge-RAG-Eval | Embedding Smoke Test")
    print("=" * 65)
    print(f"  Model Identifier       : {test_results['model_name']}")
    print(f"  Execution Device       : {test_results['device']}")
    print(f"  Embedding Dimension    : {test_results['dimension']}")
    print(f"  Output Shape (Ref)     : {test_results['ref_shape']}")
    print(f"  Output Shape (Query)   : {test_results['query_shape']}")
    print(f"  Cosine Similarity      : {test_results['cosine_similarity']:.4f}")
    print(f"  Inference Latency      : {test_results['latency_ms']:.2f} ms")
    print("=" * 65)
    print("[SUCCESS] Smoke test completed cleanly (Exit code 0)\n")


def display_generation_rich(
    ollama_info: Dict[str, Any],
    run1: GenerationResult,
    run2: GenerationResult,
    exact_match: bool,
    prompt: str,
) -> None:
    """Render deterministic generation verification table with rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Ollama notice panel
    if ollama_info["server_up"] and ollama_info["model_ready"]:
        panel_msg = f"[bold green]Ollama Status:[/bold green] Connected to {ollama_info['base_url']} with model '{ollama_info['model_name']}'."
        border = "green"
    else:
        panel_msg = (
            f"[bold yellow]Ollama Status Notice:[/bold yellow] {ollama_info['message']}\n"
            f"[cyan]To run native Ollama inference locally:[/cyan]\n"
            f"  1. Ensure Ollama service is running (`ollama serve`)\n"
            f"  2. Pull target model: `ollama pull {ollama_info['model_name']}`\n"
            f"[bold green]Active Generation Engine:[/bold green] Deterministic / Transformers Fallback"
        )
        border = "yellow"

    console.print()
    console.print(Panel(panel_msg, title="Engine Connectivity & Backend Dispatch", border_style=border))
    console.print()

    # Generation Verification Table
    gen_table = Table(
        title="Edge-RAG-Eval | Deterministic Gemma-2B Inference Verification",
        show_header=True,
        header_style="bold cyan",
    )
    gen_table.add_column("Parameter / Metric", style="bold white", width=25)
    gen_table.add_column("Run 1 (Seed 42, Temp 0.0)", style="cyan", width=26)
    gen_table.add_column("Run 2 (Seed 42, Temp 0.0)", style="cyan", width=26)
    gen_table.add_column("Verification", style="bold green", justify="center", width=24)

    gen_table.add_row("Model Identifier", run1.model_name, run2.model_name, "ALIGNED")
    gen_table.add_row("Prompt Tokens", str(run1.prompt_tokens), str(run2.prompt_tokens), "MATCH")
    gen_table.add_row("Completion Tokens", str(run1.completion_tokens), str(run2.completion_tokens), "MATCH")
    gen_table.add_row("Inference Latency", f"{run1.total_latency_ms:.2f} ms", f"{run2.total_latency_ms:.2f} ms", "BENCHMARKED")
    gen_table.add_row("Tokens Per Second", f"{run1.tokens_per_second:.2f} tok/s", f"{run2.tokens_per_second:.2f} tok/s", "COMPUTED")
    gen_table.add_row("Output String Match", "100% Bit-for-Bit", "100% Bit-for-Bit", "Pass: Determinism Verified" if exact_match else "[red]FAIL[/red]")

    console.print(gen_table)
    console.print()

    # Display prompt and output snippet
    snippet = run1.response_text[:160] + "..." if len(run1.response_text) > 160 else run1.response_text
    console.print(Panel(
        f"[bold white]Prompt:[/bold white] {prompt}\n\n"
        f"[bold white]Deterministic Output:[/bold white] {snippet}",
        title="Output Verification Sample",
        border_style="magenta",
    ))
    console.print()


def display_generation_ascii(
    ollama_info: Dict[str, Any],
    run1: GenerationResult,
    run2: GenerationResult,
    exact_match: bool,
    prompt: str,
) -> None:
    """Fallback plain-text ASCII renderer for generation verification."""
    print("\n" + "=" * 70)
    print("Edge-RAG-Eval | Deterministic Gemma-2B Inference Verification")
    print("=" * 70)
    print(f"  Ollama Status          : {ollama_info['message']}")
    print(f"  Model Identifier       : {run1.model_name}")
    print(f"  Prompt                 : {prompt}")
    print("-" * 70)
    print(f"  Run 1 Completion Tokens: {run1.completion_tokens} (Latency: {run1.total_latency_ms:.2f} ms, Throughput: {run1.tokens_per_second:.2f} tok/s)")
    print(f"  Run 2 Completion Tokens: {run2.completion_tokens} (Latency: {run2.total_latency_ms:.2f} ms, Throughput: {run2.tokens_per_second:.2f} tok/s)")
    print(f"  Exact String Match     : {'Pass: Determinism Verified' if exact_match else 'FAIL: Output Mismatch'}")
    print("=" * 70)
    print(f"  Generated Text Sample  : {run1.response_text}")
    print("=" * 70 + "\n")


def display_calm_rich(results: Dict[str, Any]) -> None:
    """Render CALM cross-attention architecture audit and parameter tables."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Panel header
    header_text = (
        "[bold cyan]CALM Architecture Audit & Cross-Attention Blueprint[/bold cyan]\n"
        "[white]Replicating Bansal et al., Google DeepMind India (arXiv:2401.02412)[/white]\n"
        "[green]Freeze Anchor Model (Gemma-2B) + Gated Cross-Attention (MiniLM)[/green]"
    )
    console.print()
    console.print(Panel.fit(header_text, border_style="cyan"))
    console.print()

    # Table 1: Blueprint & Parameter Budget Table
    param_table = Table(
        title="CALM Blueprint | Parameter Efficiency & Budget Audit",
        show_header=True,
        header_style="bold magenta",
    )
    param_table.add_column("Architecture Attribute", style="bold white", width=30)
    param_table.add_column("Configured Value", style="cyan", width=22)
    param_table.add_column("Evaluation / Budget", style="bold green", justify="center", width=22)

    param_table.add_row("Anchor Hidden Dimension (Gemma-2B)", str(results["anchor_dim"]), "FROZEN (2048)")
    param_table.add_row("Augment Dimension (MiniLM)", str(results["augment_dim"]), "CONTEXT (384)")
    param_table.add_row("Cross-Attention Dimension", str(results["attn_dim"]), f"{results['num_heads']} heads x 64")
    param_table.add_row("Base Anchor Model Parameters", f"{results['frozen_parameters']:,}", "FROZEN (0.0% trained)")
    param_table.add_row("Trainable Adapter Parameters", f"{results['trainable_parameters']:,}", "PASS (<5.0M Budget)")
    param_table.add_row("Trainable Capacity Fraction", f"{results['trainable_percentage']:.3f}%", "PASS (<1.0% Ratio)")
    param_table.add_row("Learned Gating Scalar (gamma)", f"{results['gamma_value']:.4f}", "INIT EXACT (0.0)")

    console.print(param_table)
    console.print()

    # Table 2: Functional Diagnostic Verification Table
    diag_table = Table(
        title="CALM Diagnostic | Functional & Gradient Flow Verification",
        show_header=True,
        header_style="bold green",
    )
    diag_table.add_column("Diagnostic Check", style="bold white", width=34)
    diag_table.add_column("Observed Output / Behavior", style="cyan", width=26)
    diag_table.add_column("Status", style="bold green", justify="center", width=14)

    diag_table.add_row("Anchor Input Tensor Shape", str(results["anchor_shape"]), "PASS")
    diag_table.add_row("Augment Input Tensor Shape", str(results["augment_shape"]), "PASS")
    diag_table.add_row("Output Tensor Dimension", str(results["output_shape"]), "VERIFIED")
    diag_table.add_row("Zero Degradation at Init (gamma=0)", "torch.equal(out, anchor)", "PASS")
    diag_table.add_row("Base Anchor Tensor Gradient", "grad is None (Frozen)", "ISOLATED")
    diag_table.add_row("Augment Input Tensor Gradient", "grad is None (Frozen)", "ISOLATED")
    diag_table.add_row("Gating Factor (gamma) Gradient", "grad computed (Active)", "FLOWING")
    diag_table.add_row("Adapter Projections Gradient", "W_Q, W_K, W_V, W_O active", "TRAINABLE")

    console.print(diag_table)
    console.print()


def display_calm_ascii(results: Dict[str, Any]) -> None:
    """Fallback plain-text ASCII renderer for CALM architecture audit."""
    print("\n" + "=" * 70)
    print("Edge-RAG-Eval | CALM Architecture Audit & Diagnostic")
    print("Reference: Bansal et al., Google DeepMind (arXiv:2401.02412)")
    print("=" * 70)
    print(f"  Anchor Dimension (Gemma-2B)  : {results['anchor_dim']}")
    print(f"  Augment Dimension (MiniLM)   : {results['augment_dim']}")
    print(f"  Cross-Attention Dimension    : {results['attn_dim']} ({results['num_heads']} heads)")
    print(f"  Base Frozen Parameters       : {results['frozen_parameters']:,}")
    print(f"  Trainable Adapter Parameters : {results['trainable_parameters']:,} (<5M: {results['is_under_5m']})")
    print(f"  Trainable Fraction           : {results['trainable_percentage']:.3f}% (<1%: {results['is_under_1_percent']})")
    print(f"  Gating Factor (gamma)        : {results['gamma_value']:.4f}")
    print("-" * 70)
    print(f"  Anchor Input Shape           : {results['anchor_shape']}")
    print(f"  Augment Input Shape          : {results['augment_shape']}")
    print(f"  Output Tensor Shape          : {results['output_shape']}")
    print(f"  Zero Init Equivalence        : {results['init_exact']}")
    print(f"  Base Tensor Gradient Status  : grad is None (Frozen: True)")
    print(f"  Adapter Gradient Status      : Active (Flowing: True)")
    print("=" * 70)
    print("[SUCCESS] CALM modular cross-attention audit passed cleanly (Exit code 0)\n")


def run_calm_test(
    anchor_dim: int = 2048,
    augment_dim: int = 384,
    attn_dim: int = 512,
    num_heads: int = 8,
) -> int:
    """Execute architectural audit and gradient isolation diagnostic for CALM.

    Args:
        anchor_dim: Anchor model dimension (default: 2048 for Gemma-2B).
        augment_dim: Augment context dimension (default: 384 for MiniLM).
        attn_dim: Cross-attention projection dimension (default: 512).
        num_heads: Number of attention heads (default: 8).

    Returns:
        int: Exit status code (0 for success).
    """
    logger.info("Executing CALM cross-attention architecture audit...")

    # 1. Instantiate CALM layer and compositional wrapper
    layer = CALMCrossAttentionLayer(
        anchor_dim=anchor_dim,
        augment_dim=augment_dim,
        attn_dim=attn_dim,
        num_heads=num_heads,
    )
    wrapper = CALMCompositionalWrapper(
        anchor_dim=anchor_dim,
        augment_dim=augment_dim,
        attn_dim=attn_dim,
        num_heads=num_heads,
    )

    # 2. Generate dummy tensors
    batch_size = 2
    seq_len = 16
    ctx_len = 32

    anchor_hidden_states = torch.randn(batch_size, seq_len, anchor_dim, requires_grad=False)
    augment_hidden_states = torch.randn(batch_size, ctx_len, augment_dim, requires_grad=False)

    # 3. Forward pass verification
    out = layer(anchor_hidden_states, augment_hidden_states)
    expected_shape = (batch_size, seq_len, anchor_dim)
    if out.shape != expected_shape:
        raise ValueError(f"Output shape mismatch: expected {expected_shape}, got {out.shape}")

    # Check zero-init exact equivalence (gamma = 0.0)
    init_exact = torch.equal(out, anchor_hidden_states)
    if not init_exact:
        raise AssertionError("Output must be exactly identical to anchor hidden states at gamma=0.0")

    # 4. Parameter breakdown and budget verification
    breakdown = wrapper.get_parameter_breakdown()
    trainable_params = layer.trainable_parameters
    if trainable_params >= 5_000_000:
        raise AssertionError(f"Trainable parameters ({trainable_params}) must be < 5M")
    if not breakdown["is_under_1_percent"]:
        raise AssertionError(f"Trainable percentage ({breakdown['trainable_percentage']:.2f}%) must be < 1%")

    # 5. Backward gradient isolation check
    # Check 1: Base inputs do not require grad and retain grad is None
    loss_init = out.sum()
    loss_init.backward()

    gamma_has_grad = layer.gamma.grad is not None
    anchor_frozen = anchor_hidden_states.grad is None
    augment_frozen = augment_hidden_states.grad is None

    if not anchor_frozen or not augment_frozen:
        raise AssertionError("Base tensors must remain frozen with grad is None")
    if not gamma_has_grad:
        raise AssertionError("Gating scalar gamma must receive gradients")

    # Check 2: When gamma > 0, gradients flow to cross-attention projection matrices
    with torch.no_grad():
        layer.gamma.data.fill_(1.0)
    layer.zero_grad()

    out_active = layer(anchor_hidden_states, augment_hidden_states)
    loss_active = out_active.sum()
    loss_active.backward()

    adapter_gradients_active = (
        layer.q_proj.weight.grad is not None
        and layer.k_proj.weight.grad is not None
        and layer.v_proj.weight.grad is not None
        and layer.out_proj.weight.grad is not None
        and layer.gamma.grad is not None
        and anchor_hidden_states.grad is None
    )

    if not adapter_gradients_active:
        raise AssertionError("Cross-attention adapter projections must receive gradients during backward pass")

    # Reset gamma to 0.0
    with torch.no_grad():
        layer.gamma.data.fill_(0.0)
    layer.zero_grad()

    results: Dict[str, Any] = {
        "anchor_dim": anchor_dim,
        "augment_dim": augment_dim,
        "attn_dim": attn_dim,
        "num_heads": num_heads,
        "total_parameters": breakdown["total_parameters"],
        "trainable_parameters": trainable_params,
        "frozen_parameters": breakdown["frozen_parameters"],
        "trainable_percentage": breakdown["trainable_percentage"],
        "is_under_1_percent": breakdown["is_under_1_percent"],
        "is_under_5m": breakdown["is_under_5m"],
        "gamma_value": breakdown["gamma_value"],
        "anchor_shape": tuple(anchor_hidden_states.shape),
        "augment_shape": tuple(augment_hidden_states.shape),
        "output_shape": tuple(out.shape),
        "init_exact": init_exact,
    }

    try:
        display_calm_rich(results)
    except ImportError:
        display_calm_ascii(results)

    return 0


def run_smoke_test(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> int:
    """Execute smoke test passing dummy strings through the embedding client.

    Args:
        model_name: Hugging Face model repository identifier.
        device: Target compute device or None for auto-detection.

    Returns:
        int: Exit status code (0 for success).
    """
    logger.info("Executing Edge-RAG-Eval smoke test...")
    diagnostics = get_system_diagnostics()

    import time
    start_bench = time.perf_counter()

    client = EmbeddingClient(model_name=model_name, device=device)

    ref_text = "Edge-RAG-Eval: Decoupled retrieval grounding testbed."
    query_text = "Benchmarking retrieval systems under noise."

    ref_emb = client.encode(ref_text)
    query_emb = client.encode(query_text)

    total_latency_ms = (time.perf_counter() - start_bench) * 1000.0

    # Verification: output shape must be exactly (1, 384)
    expected_shape = (1, 384)
    if ref_emb.shape != expected_shape:
        raise ValueError(
            f"Embedding shape mismatch: expected {expected_shape}, got {ref_emb.shape}"
        )
    if query_emb.shape != expected_shape:
        raise ValueError(
            f"Query embedding shape mismatch: expected {expected_shape}, got {query_emb.shape}"
        )

    # Compute cosine similarity
    dot_product = float(np.dot(ref_emb[0], query_emb[0]))
    norm_ref = float(np.linalg.norm(ref_emb[0]))
    norm_query = float(np.linalg.norm(query_emb[0]))
    cosine_sim = dot_product / (norm_ref * norm_query)

    test_results = {
        "model_name": model_name,
        "device": client.device,
        "dimension": client.dimension,
        "ref_shape": ref_emb.shape,
        "query_shape": query_emb.shape,
        "cosine_similarity": cosine_sim,
        "latency_ms": total_latency_ms,
    }

    try:
        display_results_rich(diagnostics, test_results)
    except ImportError:
        display_results_ascii(diagnostics, test_results)

    return 0


def run_generation_test(
    model_name: str = "gemma:2b",
    ollama_base_url: str = "http://localhost:11434",
    device: Optional[str] = None,
) -> int:
    """Execute determinism and reproducibility test for Gemma-2B inference engine.

    Args:
        model_name: Model identifier (defaults to 'gemma:2b').
        ollama_base_url: Base endpoint for Ollama REST API.
        device: Target compute device.

    Returns:
        int: Exit status code (0 for success).
    """
    logger.info("Executing Gemma-2B deterministic inference verification test...")

    # 1. Check Ollama reachability and model availability
    server_up, model_ready, status_msg = check_ollama(
        base_url=ollama_base_url,
        model_name=model_name,
    )
    ollama_info = {
        "server_up": server_up,
        "model_ready": model_ready,
        "base_url": ollama_base_url,
        "model_name": model_name,
        "message": status_msg,
    }

    # 2. Instantiate GemmaInferenceClient
    client = GemmaInferenceClient(
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        device=device,
        backend="auto",
    )

    # 3. Benchmark prompt
    prompt = "State the chemical formula for water and briefly explain its molecular geometry in one sentence."

    # 4. Consecutive runs under temperature=0.0 and seed=42
    logger.info("Executing Run 1 (temperature=0.0, seed=42)...")
    run1 = client.generate(
        prompt=prompt,
        temperature=0.0,
        seed=42,
        max_tokens=150,
    )

    logger.info("Executing Run 2 (temperature=0.0, seed=42)...")
    run2 = client.generate(
        prompt=prompt,
        temperature=0.0,
        seed=42,
        max_tokens=150,
    )

    # 5. Bit-for-bit determinism assertion
    exact_match = (run1.response_text == run2.response_text)
    if not exact_match:
        logger.error("Determinism failure: Run 1 != Run 2")
        logger.error("Run 1: %s", run1.response_text)
        logger.error("Run 2: %s", run2.response_text)
        raise AssertionError("Inference outputs are not bit-for-bit deterministic under temperature=0.0")

    # 6. Render verification table
    try:
        display_generation_rich(ollama_info, run1, run2, exact_match, prompt)
    except ImportError:
        display_generation_ascii(ollama_info, run1, run2, exact_match, prompt)

    return 0


def display_sampler_rich(results: list[dict[str, Any]]) -> None:
    """Render 5-sample distractor verification using rich table."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    table = Table(
        title="Edge-RAG-Eval | Adversarial & Random Distractor Sampler Verification (5 Samples)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Sample ID", style="bold white", width=10)
    table.add_column("Query & Gold Evidence", style="cyan", width=34)
    table.add_column("Random Distractor (sim < 0.25)", style="green", width=32)
    table.add_column("Adversarial Distractor (sim >= 0.65)", style="yellow", width=32)
    table.add_column("Status", style="bold green", justify="center", width=10)

    for item in results:
        q_gold = f"[bold]{item['query']}[/bold]\n[dim]Gold (sim={item['sim_gold_query']:.4f}):[/dim] {item['gold_context'][:80]}..."
        rand = f"[bold]sim={item['sim_random_query']:.4f}[/bold]\n{item['random_distractor'][:80]}..."
        adv = f"[bold]sim={item['sim_adversarial_query']:.4f}[/bold]\n{item['adversarial_distractor'][:80]}..."
        table.add_row(item["sample_id"], q_gold, rand, adv, "PASS")

    console.print()
    console.print(Panel.fit("[bold green]Adversarial Distractor Sampler Audit Passed Cleanly[/bold green]", border_style="green"))
    console.print(table)
    console.print()


def display_sampler_ascii(results: list[dict[str, Any]]) -> None:
    """Fallback plain-text ASCII renderer for distractor sampler verification."""
    print("\n" + "=" * 80)
    print("Edge-RAG-Eval | Distractor Sampler Verification (5 Samples)")
    print("=" * 80)
    for item in results:
        print(f"[{item['sample_id']}] Query: {item['query']}")
        print(f"  Gold Sim         : {item['sim_gold_query']:.4f}")
        print(f"  Random Distractor: sim={item['sim_random_query']:.4f} (< 0.25: PASS)")
        print(f"  Adv Distractor   : sim={item['sim_adversarial_query']:.4f} (>= 0.65: PASS)")
        print("-" * 80)
    print("[SUCCESS] Distractor sampler audit passed cleanly.\n")


def run_sampler_test(device: Optional[str] = None) -> int:
    """Execute a 5-sample verification test for adversarial and random distractor sampling.

    Args:
        device: Compute device identifier ('auto', 'cpu', 'cuda').

    Returns:
        int: Exit status code (0 for success).
    """
    logger.info("Executing 5-sample distractor sampler verification test...")
    sampler = DistractorSampler(device=device, seed=42)

    test_items = CURATED_SEED_CORPUS[:5]
    results: list[dict[str, Any]] = []

    for i, item in enumerate(test_items):
        sample_id = f"eval_{i+1:03d}"
        query = item["query"]
        gold_passage = item["gold_context"]
        candidate_pool = [
            item["random_context"],
            item["adversarial_context"],
            "Additional unrelated passage about tectonic subduction zones in planetary geology.",
        ]

        rand_dist, adv_dist, sim_rand, sim_adv = sampler.sample_distractors(
            query=query,
            gold_passage=gold_passage,
            candidate_pool=candidate_pool,
            random_threshold=0.25,
            adversarial_threshold=0.65,
        )

        q_emb = sampler.embedding_client.encode(query, normalize=True)
        gold_emb = sampler.embedding_client.encode(gold_passage, normalize=True)
        sim_gold = float(np.dot(q_emb.flatten(), gold_emb.flatten()))

        if sim_rand >= 0.25:
            raise AssertionError(f"Random distractor violated threshold: {sim_rand:.4f} >= 0.25")
        if sim_adv < 0.65:
            raise AssertionError(f"Adversarial distractor violated threshold: {sim_adv:.4f} < 0.65")

        results.append({
            "sample_id": sample_id,
            "query": query,
            "gold_context": gold_passage,
            "random_distractor": rand_dist,
            "adversarial_distractor": adv_dist,
            "sim_gold_query": sim_gold,
            "sim_random_query": sim_rand,
            "sim_adversarial_query": sim_adv,
        })

    try:
        display_sampler_rich(results)
    except ImportError:
        display_sampler_ascii(results)

    return 0


def display_dataset_summary_rich(stats: dict[str, Any], output_path: str) -> None:
    """Render dataset construction summary and Mann-Whitney U test results using rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    table = Table(
        title=f"Edge-RAG-Eval | 500-Sample Dataset Statistical Separation Audit ({output_path})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Condition / Metric", style="bold white", width=28)
    table.add_column("p25", style="cyan", justify="right", width=10)
    table.add_column("Median (p50)", style="bold cyan", justify="right", width=14)
    table.add_column("p75", style="cyan", justify="right", width=10)
    table.add_column("p90", style="cyan", justify="right", width=10)
    table.add_column("Range [Min, Max]", style="magenta", justify="center", width=20)
    table.add_column("Status / Check", style="bold green", justify="center", width=16)

    gold = stats["gold"]
    rand = stats["random"]
    adv = stats["adversarial"]

    table.add_row(
        "Condition 1 (Gold Only)",
        f"{gold['p25']:.4f}",
        f"{gold['p50']:.4f}",
        f"{gold['p75']:.4f}",
        f"{gold['p90']:.4f}",
        f"[{gold['min']:.4f}, {gold['max']:.4f}]",
        "GROUND TRUTH",
    )
    table.add_row(
        "Condition 2 (Random Distractor)",
        f"{rand['p25']:.4f}",
        f"{rand['p50']:.4f}",
        f"{rand['p75']:.4f}",
        f"{rand['p90']:.4f}",
        f"[{rand['min']:.4f}, {rand['max']:.4f}]",
        "PASS (< 0.25)",
    )
    table.add_row(
        "Condition 3 (Adversarial Distractor)",
        f"{adv['p25']:.4f}",
        f"{adv['p50']:.4f}",
        f"{adv['p75']:.4f}",
        f"{adv['p90']:.4f}",
        f"[{adv['min']:.4f}, {adv['max']:.4f}]",
        "PASS (>= 0.65)",
    )

    console.print()
    console.print(Panel.fit(
        f"[bold green]Dataset Constructed & Validated Successfully ({stats['sample_count']} samples)[/bold green]\n"
        f"[bold white]Mann-Whitney U Test:[/] statistic = {stats['statistic']:.1f}, "
        f"[bold green]p-value = {stats['p_value']:.4e}[/] (Statistically Distinct: [bold green]p < 0.001[/])\n"
        f"[bold white]Separation Margin (Min Adv - Max Rand):[/] [bold cyan]{stats['separation_margin']:.4f}[/]\n"
        f"[bold white]Dataset Artifact:[/] [cyan]{output_path}[/]",
        border_style="green",
    ))
    console.print(table)
    console.print()


def display_dataset_summary_ascii(stats: dict[str, Any], output_path: str) -> None:
    """Fallback plain-text ASCII renderer for dataset construction summary."""
    print("\n" + "=" * 80)
    print(f"Edge-RAG-Eval | Dataset Construction Summary ({stats['sample_count']} samples)")
    print("=" * 80)
    print(f"  Artifact Path            : {output_path}")
    print(f"  Mann-Whitney U Statistic : {stats['statistic']}")
    print(f"  Mann-Whitney U p-value   : {stats['p_value']:.4e} (p < 0.001: {stats['statistically_distinct']})")
    print(f"  Separation Margin        : {stats['separation_margin']:.4f}")
    print("-" * 80)
    for name, key in [("Gold Only", "gold"), ("Random Distractor", "random"), ("Adversarial Distractor", "adversarial")]:
        d = stats[key]
        print(f"  {name:<24} : p25={d['p25']:.4f}, p50={d['p50']:.4f}, p75={d['p75']:.4f}, p90={d['p90']:.4f}, range=[{d['min']:.4f}, {d['max']:.4f}]")
    print("=" * 80 + "\n")


def run_build_dataset(
    output_path: str = "data/processed/eval_500.json",
    num_samples: int = 500,
    device: Optional[str] = None,
) -> int:
    """Construct, validate, and persist the evaluation benchmark dataset.

    Args:
        output_path: Path to target json file.
        num_samples: Number of benchmark samples.
        device: Compute device identifier.

    Returns:
        int: Exit status code (0 for success).
    """
    logger.info("Starting dataset build: %d samples -> %s", num_samples, output_path)
    samples, stats = build_evaluation_dataset(
        num_samples=num_samples,
        output_path=output_path,
        device=device,
    )

    try:
        display_dataset_summary_rich(stats, output_path)
    except ImportError:
        display_dataset_summary_ascii(stats, output_path)

    return 0


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to configuration YAML.

    Returns:
        Dict[str, Any]: Parsed configuration mapping.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="edge_rag_eval",
        description="Edge-RAG-Eval: Standalone headless research testbed for edge foundation models.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the end-to-end embedding smoke test and system diagnostic verification.",
    )
    parser.add_argument(
        "--test-generation",
        action="store_true",
        help="Run the deterministic Gemma-2B inference engine verification and exact string match test.",
    )
    parser.add_argument(
        "--test-calm",
        action="store_true",
        help="Run the CALM cross-attention architecture audit and gradient isolation diagnostic.",
    )
    parser.add_argument(
        "--test-sampler",
        action="store_true",
        help="Run 5-sample verification test for adversarial and random distractor sampling engine.",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Construct and validate the 500-sample benchmark dataset (data/processed/eval_500.json).",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=500,
        help="Number of samples to generate for benchmark dataset (default: 500).",
    )
    parser.add_argument(
        "--dataset-output",
        type=str,
        default="data/processed/eval_500.json",
        help="Target filepath for generated benchmark dataset (default: data/processed/eval_500.json).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_config.yaml",
        help="Path to YAML configuration file (default: configs/baseline_config.yaml).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging output.",
    )
    return parser


def main(args_list: Optional[list[str]] = None) -> int:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load defaults from config if present
    generator_model = "gemma:2b"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    device = "auto"

    if Path(args.config).is_file():
        try:
            cfg = load_config(args.config)
            models_cfg = cfg.get("models", {})
            embedding_model = models_cfg.get("embedding", embedding_model)
            generator_model = models_cfg.get("generator", generator_model)
            device = models_cfg.get("device", device)
        except Exception as e:
            logger.warning("Could not read config '%s': %s", args.config, e)

    if args.test_sampler:
        return run_sampler_test(device=device)

    if args.build_dataset:
        return run_build_dataset(
            output_path=args.dataset_output,
            num_samples=args.dataset_size,
            device=device,
        )

    if args.test_calm:
        return run_calm_test(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)

    if args.test_generation:
        return run_generation_test(model_name=generator_model, device=device)

    if args.test:
        return run_smoke_test(model_name=embedding_model, device=device)

    # If no test flag specified, display config or usage help
    if Path(args.config).is_file():
        config_data = load_config(args.config)
        print(f"Loaded config from {args.config}:")
        print(yaml.dump(config_data, default_flow_style=False))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

