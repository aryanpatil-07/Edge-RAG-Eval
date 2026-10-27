"""CLI entrypoint and execution pipeline for Edge-RAG-Eval.

Usage:
    python -m src.pipeline --test
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

from src.models.inference import EmbeddingClient

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

    if args.test:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        device = "auto"
        # If config file exists, attempt reading model defaults
        if Path(args.config).is_file():
            try:
                cfg = load_config(args.config)
                models_cfg = cfg.get("models", {})
                model_name = models_cfg.get("embedding", model_name)
                device = models_cfg.get("device", device)
            except Exception as e:
                logger.warning("Could not read config '%s': %s", args.config, e)

        return run_smoke_test(model_name=model_name, device=device)

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
