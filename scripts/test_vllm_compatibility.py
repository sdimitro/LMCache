#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test LMCache compatibility with different vLLM versions.

This script creates temporary uv environments, installs different vLLM versions
along with the current LMCache repo in editable mode, and runs the integration test.

Usage:
    python test_vllm_compatibility.py
    [--versions VERSION1 VERSION2 ...]
    [--all-versions]
    [--no-logs]
    [--output OUTPUT_FILE]
"""

# Standard
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import shutil
import subprocess
import sys
import tempfile


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")


def print_section(text: str):
    """Print a formatted section."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'-' * len(text)}{Colors.END}")


def run_command(
    cmd: List[str],
    env: Optional[dict] = None,
    capture_output: bool = False,
    check: bool = True,
    cwd: Optional[str] = None,
) -> Tuple[int, str, str]:
    """
    Run a shell command and return the result.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        if capture_output:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, cwd=cwd
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, env=env, check=check, text=True, cwd=cwd)
            return result.returncode, "", ""
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return e.returncode, "", str(e)


def get_lmcache_repo_path() -> Path:
    """Get the path to the LMCache repository (parent of scripts directory)."""
    return Path(__file__).parent.parent.resolve()


def create_uv_env(env_path: Path) -> bool:
    """Create a new uv environment."""
    print(f"  Creating uv environment at {env_path}...")
    try:
        run_command(["uv", "venv", str(env_path)], check=True)
        print(f"  {Colors.GREEN}✓{Colors.END} Environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  {Colors.RED}✗{Colors.END} Failed to create environment: {e}")
        return False


def install_packages_in_env(env_path: Path, packages: List[str]) -> bool:
    """Install packages in the uv environment."""
    python_path = env_path / "bin" / "python"

    for package in packages:
        print(f"  Installing {package}...")
        returncode, stdout, stderr = run_command(
            ["uv", "pip", "install", "--python", str(python_path), package],
            capture_output=True,
            check=False,
        )
        if returncode != 0:
            print(f"  {Colors.RED}✗{Colors.END} Failed to install {package}")
            print(f"  Error: {stderr}")
            return False

    print(f"  {Colors.GREEN}✓{Colors.END} All packages installed")
    return True


def install_lmcache_editable(env_path: Path, lmcache_path: Path) -> bool:
    """Install LMCache in editable mode."""
    python_path = env_path / "bin" / "python"

    print(f"  Installing LMCache from {lmcache_path} in editable mode...")
    returncode, stdout, stderr = run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_path),
            "-e",
            str(lmcache_path),
            "--no-build-isolation",
        ],
        capture_output=True,
        check=False,
    )

    if returncode != 0:
        print(f"  {Colors.RED}✗{Colors.END} Failed to install LMCache")
        print(f"  Error: {stderr}")
        return False

    print(f"  {Colors.GREEN}✓{Colors.END} LMCache installed in editable mode")
    return True


def run_integration_test(env_path: Path, test_script: Path) -> Tuple[bool, str]:
    """
    Run the vLLM integration test.

    Returns:
        Tuple of (success, output)
    """
    python_path = env_path / "bin" / "python"

    print(f"  Running integration test: {test_script.name}")

    returncode, stdout, stderr = run_command(
        [str(python_path), str(test_script)], capture_output=True, check=False
    )

    output = stdout + "\n" + stderr
    success = returncode == 0 and "✓ Done!" in output

    if success:
        print(f"  {Colors.GREEN}✓{Colors.END} Test passed!")
    else:
        print(f"  {Colors.RED}✗{Colors.END} Test failed!")

    return success, output


def test_vllm_version(
    version: str, lmcache_path: Path, test_script: Path, keep_logs: bool = True
) -> Dict:
    """
    Test a specific vLLM version.

    Returns:
        Dictionary with test results
    """
    print_section(f"Testing vLLM {version}")

    result = {
        "version": version,
        "success": False,
        "error": None,
        "output": "",
        "timestamp": datetime.now().isoformat(),
    }

    # Create temporary directory for this test
    temp_dir = Path(tempfile.mkdtemp(prefix=f"vllm_{version.replace('.', '_')}_"))
    env_path = temp_dir / "venv"

    try:
        # Step 1: Create environment
        if not create_uv_env(env_path):
            result["error"] = "Failed to create uv environment"
            return result

        # Step 2: Install build dependencies
        print("  Installing build dependencies...")
        build_deps = [
            "ninja",
            "packaging>=24.2",
            "setuptools>=77.0.3,<81.0.0",
            "setuptools_scm>=8",
            "wheel",
        ]
        if not install_packages_in_env(env_path, build_deps):
            result["error"] = "Failed to install build dependencies"
            return result

        # Step 3: Install PyTorch (required for both vLLM and LMCache)
        print("  Installing PyTorch...")
        python_path = env_path / "bin" / "python"
        returncode, _, stderr = run_command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_path),
                "torch==2.8.0",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ],
            capture_output=True,
            check=False,
        )
        if returncode != 0:
            print(
                f"  {Colors.YELLOW}⚠{Colors.END} PyTorch installation "
                "had warnings, continuing..."
            )

        # Step 4: Install vLLM
        print(f"  Installing vLLM {version}...")
        vllm_package = f"vllm=={version}"
        if not install_packages_in_env(env_path, [vllm_package]):
            result["error"] = f"Failed to install vLLM {version}"
            return result

        # Step 5: Install LMCache in editable mode
        if not install_lmcache_editable(env_path, lmcache_path):
            result["error"] = "Failed to install LMCache"
            return result

        # Step 6: Run the integration test
        success, output = run_integration_test(env_path, test_script)
        result["success"] = success
        result["output"] = output

        if not success:
            result["error"] = "Integration test failed"

        # Save logs if requested
        if keep_logs:
            log_dir = lmcache_path / "compatibility_logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"vllm_{version.replace('.', '_')}.log"
            with open(log_file, "w") as f:
                f.write(f"vLLM Version: {version}\n")
                f.write(f"Test Time: {result['timestamp']}\n")
                f.write(f"Success: {result['success']}\n")
                f.write(f"Error: {result['error']}\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("Test Output:\n")
                f.write("=" * 80 + "\n")
                f.write(output)
            print(f"  Log saved to: {log_file}")

    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        print(f"  {Colors.RED}✗{Colors.END} Error: {e}")

    finally:
        # Clean up temporary directory
        print("  Cleaning up temporary directory...")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(
                f"  {Colors.YELLOW}⚠{Colors.END} Warning: "
                f"Failed to clean up {temp_dir}: {e}"
            )

    return result


def print_compatibility_matrix(results: List[Dict]):
    """Print a formatted compatibility matrix."""
    print_header("COMPATIBILITY MATRIX")

    # Summary statistics
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed

    print(f"{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Total versions tested: {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
    print(f"  Success rate: {(passed / total * 100):.1f}%\n")

    # Detailed results table
    print(f"{Colors.BOLD}Detailed Results:{Colors.END}\n")
    print(f"{'vLLM Version':<20} {'Status':<15} {'Notes':<45}")
    print("-" * 80)

    for result in results:
        version = result["version"]
        if result["success"]:
            status = f"{Colors.GREEN}✓ PASS{Colors.END}"
            notes = "Compatible"
        else:
            status = f"{Colors.RED}✗ FAIL{Colors.END}"
            error = result.get("error", "Unknown error")
            notes = error[:45]

        print(f"{version:<20} {status:<24} {notes:<45}")

    print("\n")


def save_results_json(results: List[Dict], output_file: Path):
    """Save results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


def get_vllm_versions_to_test(args) -> List[str]:
    """Get the list of vLLM versions to test based on arguments."""

    # Default list of important vLLM versions
    default_versions = [
        "0.6.4.post1",
        "0.6.5",
        "0.6.6",
        "0.7.0",
        "0.7.1",
        "0.7.2",
        "0.7.3",
    ]

    # Extended list for comprehensive testing
    all_versions = [
        "0.6.0",
        "0.6.1",
        "0.6.2",
        "0.6.3",
        "0.6.4",
        "0.6.4.post1",
        "0.6.5",
        "0.6.6",
        "0.7.0",
        "0.7.1",
        "0.7.2",
        "0.7.3",
    ]

    if args.all_versions:
        return all_versions
    elif args.versions:
        return args.versions
    else:
        return default_versions


def main():
    parser = argparse.ArgumentParser(
        description="Test LMCache compatibility with different vLLM versions"
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        help="Specific vLLM versions to test (e.g., 0.6.5 0.7.0)",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Test all known vLLM versions (comprehensive test)",
    )
    parser.add_argument(
        "--no-logs", action="store_true", help="Don't save individual test logs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results "
        "(default: compatibility_results_TIMESTAMP.json)",
    )

    args = parser.parse_args()

    # Get paths
    lmcache_path = get_lmcache_repo_path()
    test_script = lmcache_path / "scripts" / "vllm_integration_test.py"

    # Verify test script exists
    if not test_script.exists():
        print(f"{Colors.RED}Error:{Colors.END} Test script not found: {test_script}")
        sys.exit(1)

    # Get versions to test
    versions = get_vllm_versions_to_test(args)

    print_header("LMCache × vLLM Compatibility Test Suite")
    print(f"{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  LMCache path: {lmcache_path}")
    print(f"  Test script: {test_script.name}")
    print(f"  Versions to test: {len(versions)}")
    print(f"  Keep logs: {not args.no_logs}")
    print()

    # Test each version
    results = []
    for i, version in enumerate(versions, 1):
        print(
            f"\n{Colors.BOLD}{Colors.MAGENTA}[{i}/{len(versions)}]{Colors.END} ", end=""
        )
        result = test_vllm_version(
            version, lmcache_path, test_script, keep_logs=not args.no_logs
        )
        results.append(result)

    # Print compatibility matrix
    print_compatibility_matrix(results)

    # Save results to JSON
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = lmcache_path / f"compatibility_results_{timestamp}.json"

    save_results_json(results, output_file)

    # Exit with appropriate code
    all_passed = all(r["success"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
