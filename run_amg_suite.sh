#!/bin/bash
# AMG Test Suite - Run core tests before creating a release

set -e

echo "=================================="
echo "Running AMG Test Suite"
echo "=================================="
echo ""

pytest tests/v1/test_weka.py tests/test_observability.py "$@"

