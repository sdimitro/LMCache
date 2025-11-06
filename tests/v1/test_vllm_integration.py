# SPDX-License-Identifier: Apache-2.0
"""
vLLM Integration Test for LMCache.

Simple test that runs actual vLLM inference with LMCache enabled using Weka backend.

NOTE: Due to CUDA + multiprocessing issues in pytest, it's recommended to use
the standalone script for testing: python test_vllm_simple.py

If you want to run this test, run it in isolation:
    pytest tests/v1/test_vllm_integration.py::test_vllm_with_lmcache -v -s --forked

The --forked flag requires pytest-forked: pip install pytest-forked
"""

# Standard
import os
import subprocess
import sys

# Third Party
import pytest

# Test if vLLM is available
try:
    # Third Party
    from vllm.version import __version__ as VLLM_VERSION

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VLLM_VERSION = "unknown"


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
def test_vllm_with_lmcache():
    """
    Test vLLM inference with LMCache using Weka backend.

    This test runs the standalone script in a subprocess to avoid
    CUDA initialization issues with pytest's multiprocessing.

    Run with: pytest tests/v1/test_vllm_integration.py -v -s
    """
    # Get the path to the standalone script
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    script_path = os.path.join(repo_root, "scripts", "vllm_integration_test.py")

    if not os.path.exists(script_path):
        pytest.skip(f"Standalone script not found at {script_path}")

    print("\n=== Running vLLM integration test via subprocess ===")
    print(f"Script: {script_path}\n")

    # Run the script in a subprocess with clean environment
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout
    )

    # Print output for visibility
    print("=== STDOUT ===")
    print(result.stdout)

    if result.stderr:
        print("\n=== STDERR ===")
        # Filter out common vLLM warnings
        stderr_lines = result.stderr.split("\n")
        important_lines = [
            line
            for line in stderr_lines
            if "ERROR" in line or "FAILED" in line or "Traceback" in line
        ]
        if important_lines:
            print("\n".join(important_lines))

    # Check if the script succeeded
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"

    # Verify expected output
    assert "Tokyo" in result.stdout, "Expected 'Tokyo' in output"
    assert "✓ Done!" in result.stdout, "Expected successful completion message"

    print("\n✓ vLLM integration test passed")
