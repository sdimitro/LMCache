#!/bin/bash
# AMG Test Suite - Run core tests before creating a release

set -e

echo "=================================="
echo "Checking Test Dependencies"
echo "=================================="
echo ""
uv pip install -r requirements/test.txt

echo "=================================="
echo "Running AMG Test Suite"
echo "=================================="
echo ""

pytest tests/v1/test_weka.py tests/test_observability.py tests/v1/test_gpu_connector.py tests/v1/test_vllm_integration.py "$@"

