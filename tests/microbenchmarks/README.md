# LMCache Microbenchmarks

This directory contains microbenchmarks for performance testing individual LMCache functions to detect regressions and compare different configurations.

## Quick Start

**🚀 Easiest way - Use the CLI runner:**
```bash
cd /path/to/LMCache

# Quick development benchmark (fastest)
./tests/microbenchmarks/run_benchmarks.py --quick

# Comprehensive performance analysis
./tests/microbenchmarks/run_benchmarks.py --comprehensive

# Scaling analysis across batch sizes
./tests/microbenchmarks/run_benchmarks.py --scaling

# Compare single vs batched operations
./tests/microbenchmarks/run_benchmarks.py --comparison
```

**📊 Example Output:**
```
============================================================
BENCHMARK: batched_get_blocking_quick
Config: quick_test
Batch size: 5, Shape: (2, 16, 8, 128)
============================================================
Mean time:    4.16 ms
Median time:  4.52 ms
Std dev:      1.56 ms
Min time:     2.46 ms
Max time:     5.51 ms
Ops/sec:      1201.90
Throughput:   75.12 MB/s
Device:       cuda:0
CUDA mem:     128.00 MB
```

**Alternative - Direct pytest:**
```bash
# Quick development test
python -m pytest tests/microbenchmarks/test_weka_gds_bench.py::test_quick_benchmark -v -s

# All benchmarks
python -m pytest tests/microbenchmarks/ -m benchmark -v -s

# Custom configuration
python -m pytest tests/microbenchmarks/ -m benchmark \
    --benchmark-runs=20 \
    --benchmark-warmup=5 \
    --save-results -s
```

## Structure

- `benchmark_config.py` - Configuration classes and settings
- `benchmark_utils.py` - Core benchmarking utilities and result processing
- `conftest.py` - Pytest fixtures and configuration  
- `test_weka_gds_bench.py` - WekaGDS backend benchmarks
- `README.md` - This documentation

## Available Benchmarks

### WekaGDS Backend (`test_weka_gds_bench.py`)

1. **`test_batched_get_blocking_performance`** - Basic performance test across different batch sizes and tensor shapes
2. **`test_batched_get_blocking_scaling`** - Scaling analysis showing how performance changes with batch size
3. **`test_batched_get_blocking_comprehensive`** - Comprehensive test with different backend configurations
4. **`test_single_vs_batched_comparison`** - Comparison of single vs batched operations
5. **`test_quick_benchmark`** - Fast benchmark for development

## Command Line Options

### CLI Runner Options (`run_benchmarks.py`):
- `--quick` - Fast development benchmark (~7 seconds)
- `--scaling` - Scaling analysis across batch sizes  
- `--comprehensive` - All benchmarks with different configurations (slow)
- `--comparison` - Single vs batched performance comparison
- `--custom` - Custom benchmark configuration
- `--runs N` - Number of timing runs per benchmark (default: 10)
- `--warmup N` - Number of warmup runs (default: 3)
- `--save-results` - Save detailed results to JSON files

### Pytest Options:
- `--benchmark-runs=N` - Number of timing runs per benchmark (default: 10)
- `--benchmark-warmup=N` - Number of warmup runs (default: 3) 
- `--save-results` - Save detailed results to JSON files
- `-m benchmark` - Run only benchmark tests
- `-m "benchmark and not slow"` - Skip slow comprehensive tests
- `-s` - Show output (required to see benchmark results)
- `-v` - Verbose output

## Adding New Benchmarks

### 1. For New Functions in Existing Backends

Add test methods to `test_weka_gds_bench.py`:

```python
@pytest.mark.benchmark
def test_submit_put_task_performance(self, benchmark_config, weka_backend_factory, test_data_generator):
    """Benchmark submit_put_task performance."""
    backend = weka_backend_factory()
    keys, memory_objs = test_data_generator(backend, batch_size=10)
    
    runner = BenchmarkRunner("submit_put_task")
    
    def submit_puts():
        futures = backend.batched_submit_put_task(keys, memory_objs)
        for future in futures:
            future.result()  # Wait for completion
    
    result = runner.run_benchmark(
        function_name="submit_put_task",
        func=submit_puts,
        func_args=(),
        func_kwargs={},
        batch_size=10,
        tensor_shape=(2, 16, 8, 128),
        backend_config={"chunk_size": 256, "cufile_buffer_size": 128, "gds_io_threads": 32},
        warmup_runs=benchmark_config.warmup_runs,
        benchmark_runs=benchmark_config.benchmark_runs,
    )
    
    runner.print_summary(result)
    
    # Optional: Performance assertions
    assert result.mean_time < 0.1  # Should complete within 100ms
```

### 2. For New Backends

Create a new test file following the pattern:

```python
# test_my_backend_bench.py
@pytest.mark.benchmark
class TestMyBackendBenchmarks:
    def test_my_backend_function(self, benchmark_config, my_backend_fixture):
        # Similar structure to WekaGDS tests
        pass
```

### 3. Add Fixtures for New Backends

In `conftest.py`, add fixtures for your backend:

```python
@pytest.fixture
def my_backend_factory():
    def _create_backend(**config):
        # Create your backend
        pass
    return _create_backend
```

## Configuration

Modify `benchmark_config.py` to adjust:
- Batch sizes to test
- Tensor shapes 
- Backend configurations
- Number of runs and warmup

## Results Analysis

### Metrics Captured
Benchmark results include:
- **Timing:** Mean, median, min, max, std dev of execution times
- **Throughput:** MB/s and operations per second
- **System Info:** CUDA device, memory usage
- **Consistency:** Standard deviation for performance variance detection
- **Raw Data:** All individual timing measurements

### Example Results
```
Mean time:    4.16 ms      # Average execution time
Median time:  4.52 ms      # Middle value (less affected by outliers)
Std dev:      1.56 ms      # Consistency measure
Min time:     2.46 ms      # Best performance
Max time:     5.51 ms      # Worst performance
Ops/sec:      1201.90      # Operations throughput
Throughput:   75.12 MB/s   # Data throughput
Device:       cuda:0       # GPU device used
CUDA mem:     128.00 MB    # Memory allocated
```

### Saving and Loading Results
Results can be saved to JSON for regression tracking:

```python
from tests.microbenchmarks.benchmark_utils import BenchmarkRunner

# Save results during benchmark
runner = BenchmarkRunner("my_analysis")
# ... run benchmarks ...
runner.save_results("baseline_results.json")

# Load and analyze later
results = runner.load_results("baseline_results.json")
for result in results:
    print(f"{result.function_name}: {result.mean_time*1000:.2f}ms")
```

## Performance Regression Detection

### Automatic Thresholds
The benchmarks include built-in regression detection with assertions:
```python
# Performance assertions in tests
max_time_per_item = 0.1  # 100ms per item max
expected_max_time = max_time_per_item * result.batch_size

assert result.mean_time <= expected_max_time, (
    f"Performance regression detected: {result.mean_time:.3f}s > {expected_max_time:.3f}s"
)
```

### Compare Saved Results
For detailed regression analysis:
```python
from tests.microbenchmarks.benchmark_utils import compare_results

baseline = runner.load_results("baseline_results.json")[0]
current = runner.load_results("current_results.json")[0]

comparison = compare_results(baseline, current, threshold_percent=5.0)

if comparison["is_regression"]:
    print(f"🔴 Performance regression: {comparison['mean_diff_percent']:.1f}% slower")
elif comparison["is_improvement"]:
    print(f"🟢 Performance improvement: {comparison['mean_diff_percent']:.1f}% faster")
else:
    print(f"✅ Performance stable: {comparison['mean_diff_percent']:.1f}% change")

# Detailed comparison data:
# comparison["baseline_mean"]     - Original timing
# comparison["current_mean"]      - New timing  
# comparison["mean_diff_percent"] - Percentage change
# comparison["median_diff_percent"] - Median comparison
```

## Best Practices

### 🎯 For Reliable Results
1. **Consistent hardware** - Results vary between systems; always benchmark on target hardware
2. **Multiple runs** - Use `--runs 20` for production validation (default 10 is fine for development)
3. **Clean system** - Close other applications during benchmarking to reduce noise
4. **Realistic scenarios** - Test with batch sizes and tensor shapes from your actual workloads

### 📈 For Regression Detection
1. **Save baselines** - Use `--save-results` before making changes:
   ```bash
   ./run_benchmarks.py --comprehensive --save-results  # Save baseline
   # Make your changes...
   ./run_benchmarks.py --comprehensive --save-results  # Compare results
   ```

2. **Regular monitoring** - Run benchmarks in CI/CD:
   ```bash
   # In CI pipeline
   ./run_benchmarks.py --custom --runs 5  # Faster for CI
   ```

3. **Development workflow**:
   ```bash
   ./run_benchmarks.py --quick           # Fast feedback during development
   ./run_benchmarks.py --scaling         # Check scaling behavior
   ./run_benchmarks.py --comprehensive   # Full validation before commit
   ```

### ⚡ Development Tips
- Use `--quick` for fastest feedback (3 runs, ~7 seconds)
- Use `--scaling` to verify batching efficiency 
- Use `--comparison` to validate batched vs single performance gains
- The framework automatically handles CUDA synchronization and memory management

## Troubleshooting

### ❌ Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch sizes or increase buffer
# Edit benchmark_config.py:
batch_sizes: List[int] = [1, 2, 5, 10]  # Smaller batches
cufile_buffer_sizes: List[int] = [256, 512]  # Larger buffers
```

**Slow Benchmarks:**
```bash
# Skip comprehensive tests during development
./run_benchmarks.py --quick              # ~7 seconds
python -m pytest -m "benchmark and not slow" -s  # Skip slow tests
```

**No Output Visible:**
```bash
# Always use -s flag with pytest to see benchmark results
python -m pytest tests/microbenchmarks/ -m benchmark -v -s
# Or use the CLI runner (automatically includes -s)
./run_benchmarks.py --quick
```

**Missing Weka Mount:**
The tests expect `/mnt/weka/` to be available. For testing without Weka:
1. Edit `conftest.py` to change `weka_test_dir` fixture to point to regular directory
2. Performance will be degraded but functionality should work for development

**Permission Issues:**
```bash
chmod +x tests/microbenchmarks/run_benchmarks.py  # Make runner executable
```

### ✅ Verification
Test your setup works:
```bash
./tests/microbenchmarks/run_benchmarks.py --quick
# Should show timing results like:
# Mean time: ~4ms, Throughput: ~75 MB/s, Ops/sec: ~1200
```
