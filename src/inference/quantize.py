"""GGUF model conversion and quantization utilities.

Converts a HuggingFace model to the GGUF format used by llama.cpp, with
optional quantization to reduce model size for edge deployment.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported quantization levels in order of increasing size/quality.
_QUANT_LEVELS: dict[str, str] = {
    "Q4_K_M": "Produces a good balance of size and quality (~4 bits/token).",
    "Q5_K_M": "Slightly larger than Q4 with better quality (~5 bits/token).",
    "Q8_0": "Near-original quality at higher size (~8 bits/token).",
}


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantize: bool = True,
    quant_level: str = "Q4_K_M",
    llama_cpp_dir: Optional[str] = None,
) -> str:
    """Convert a HuggingFace model to GGUF format, optionally quantized.

    The conversion process:
    1. Export the model to the unquantized GGUF (F32 or F16) using
       ``llama.cpp``'s ``convert_hf_to_gguf.py`` script.
    2. If *quantize* is True, run ``llama-quantize`` to produce a compressed
       GGUF file.

    Args:
        model_path: Path to the HuggingFace model directory.
        output_path: Destination path for the output GGUF file.
        quantize: Whether to apply quantization after conversion.
        quant_level: Quantization level (``"Q4_K_M"``, ``"Q5_K_M"``, ``"Q8_0"``).
        llama_cpp_dir: Path to the llama.cpp repository.  When *None*, the
            function attempts to find ``convert_hf_to_gguf.py`` on ``PATH`` or
            uses the ``gguf`` Python package.

    Returns:
        Path to the generated GGUF file.

    Raises:
        FileNotFoundError: If the model directory or llama.cpp tools are missing.
        ValueError: If the quantization level is unsupported.
    """
    if quant_level not in _QUANT_LEVELS:
        raise ValueError(
            f"Unsupported quantization level '{quant_level}'. "
            f"Choose from: {', '.join(_QUANT_LEVELS.keys())}"
        )

    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Attempt conversion via Python gguf package first (preferred).
    try:
        unquantized = _convert_via_gguf_package(model_dir, output_dir)
    except (ImportError, subprocess.CalledProcessError) as exc:
        logger.info(
            "gguf package conversion failed (%s), falling back to llama.cpp script.",
            exc,
        )
        unquantized = _convert_via_llama_cpp(model_dir, output_dir, llama_cpp_dir)

    logger.info("Unquantized GGUF written to %s", unquantized)

    if not quantize:
        # Rename to the final output path
        final = Path(output_path)
        shutil.copy2(str(unquantized), str(final))
        return str(final)

    # Step 2: Quantize
    quantized = _quantize_gguf(unquantized, output_dir, quant_level, llama_cpp_dir)

    # Copy to final output path
    final = Path(output_path)
    shutil.copy2(str(quantized), str(final))

    # Report sizes
    original_size = unquantized.stat().st_size / (1024 ** 3)
    final_size = final.stat().st_size / (1024 ** 3)
    ratio = final_size / original_size if original_size > 0 else 0.0

    logger.info(
        "Quantization complete: %.2f GB -> %.2f GB (%.1f%% of original)",
        original_size,
        final_size,
        100.0 * ratio,
    )

    # Clean up the intermediate unquantized file
    try:
        unquantized.unlink()
    except OSError:
        pass

    return str(final)


# ---------------------------------------------------------------------------
# Conversion strategies
# ---------------------------------------------------------------------------

def _convert_via_gguf_package(
    model_dir: Path,
    output_dir: Path,
) -> Path:
    """Convert using the ``gguf`` Python package."""
    try:
        from gguf import GGUFWriter  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'gguf' Python package is required.  "
            "Install it with: pip install gguf"
        ) from exc

    # Use llama.cpp's convert_hf_to_gguf.py if available as a script
    convert_script = shutil.which("convert_hf_to_gguf.py")
    if convert_script is not None:
        output_file = output_dir / "model-f16.gguf"
        cmd = [sys.executable, convert_script, str(model_dir), "--outfile", str(output_file), "--outtype", "f16"]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_file

    # Fall back to calling llama_cpp_python conversion if installed
    try:
        from llama_cpp import Llama  # type: ignore[import-untyped]
        logger.info("llama_cpp_python detected but does not support conversion. "
                     "Please install llama.cpp tools.")
    except ImportError:
        pass

    raise subprocess.CalledProcessError(
        1, "convert_hf_to_gguf.py",
        stderr="No conversion tool found. Install llama.cpp or gguf package.",
    )


def _convert_via_llama_cpp(
    model_dir: Path,
    output_dir: Path,
    llama_cpp_dir: Optional[str],
) -> Path:
    """Convert using llama.cpp's convert_hf_to_gguf.py script."""
    script_path = _find_convert_script(llama_cpp_dir)
    output_file = output_dir / "model-f16.gguf"

    cmd = [
        sys.executable,
        str(script_path),
        str(model_dir),
        "--outfile", str(output_file),
        "--outtype", "f16",
    ]

    logger.info("Running conversion: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.stderr:
        logger.debug("Conversion stderr:\n%s", result.stderr)

    if not output_file.exists():
        raise FileNotFoundError(
            f"Conversion script did not produce output file: {output_file}"
        )

    return output_file


def _find_convert_script(llama_cpp_dir: Optional[str]) -> Path:
    """Locate the convert_hf_to_gguf.py script."""
    # Explicit path
    if llama_cpp_dir is not None:
        candidate = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate

    # Common install locations
    candidates = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/opt") / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.cwd() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Try to find on PATH
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "Cannot find convert_hf_to_gguf.py.  Clone llama.cpp and pass "
        "--llama-cpp-dir /path/to/llama.cpp, or install the gguf package."
    )


def _quantize_gguf(
    input_path: Path,
    output_dir: Path,
    quant_level: str,
    llama_cpp_dir: Optional[str],
) -> Path:
    """Run llama-quantize on an unquantized GGUF file."""
    quantize_bin = _find_quantize_binary(llama_cpp_dir)

    suffix = f"{quant_level.lower()}.gguf"
    output_file = output_dir / f"model-{suffix}"

    cmd = [str(quantize_bin), str(input_path), str(output_file), quant_level]

    logger.info("Running quantization: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.stderr:
        logger.debug("Quantization stderr:\n%s", result.stderr)

    if not output_file.exists():
        raise FileNotFoundError(
            f"Quantization did not produce output file: {output_file}"
        )

    return output_file


def _find_quantize_binary(llama_cpp_dir: Optional[str]) -> Path:
    """Locate the llama-quantize binary."""
    if llama_cpp_dir is not None:
        candidate = Path(llama_cpp_dir) / "llama-quantize"
        if candidate.exists():
            return candidate

    found = shutil.which("llama-quantize")
    if found:
        return Path(found)

    # Try with .exe on Windows
    found_exe = shutil.which("llama-quantize.exe")
    if found_exe:
        return Path(found_exe)

    raise FileNotFoundError(
        "Cannot find llama-quantize binary.  Build llama.cpp and ensure "
        "llama-quantize is on your PATH, or pass --llama-cpp-dir."
    )
