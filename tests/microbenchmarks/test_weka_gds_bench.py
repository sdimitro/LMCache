# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks for WekaGDS backend functions."""

# Standard
# Standard Library

# Third Party
import pytest

# Local
from .benchmark_config import BenchmarkConfig
from .benchmark_utils import BenchmarkRunner


@pytest.mark.benchmark
class TestWekaGdsBenchmarks:
    """Benchmark tests for WekaGDS backend methods."""

    def test_batched_get_blocking_performance(
        self,
        benchmark_config: BenchmarkConfig,
        populated_backend,
        request,
    ):
        """Benchmark batched_get_blocking with different configurations."""

        results = []
        runner = BenchmarkRunner("batched_get_blocking")

        print("\n" + "=" * 80)
        print("BENCHMARKING: batched_get_blocking()")
        print("=" * 80)

        # Test different batch sizes
        for batch_size in benchmark_config.batch_sizes:
            print(f"\n--- Testing batch size: {batch_size} ---")

            # Test different tensor shapes
            for tensor_shape in benchmark_config.tensor_shapes:
                print(f"  Shape: {tensor_shape}")

                # Test different backend configurations
                for chunk_size in [
                    benchmark_config.chunk_sizes[0]
                ]:  # Start with one config
                    for cufile_buffer_size in [benchmark_config.cufile_buffer_sizes[0]]:
                        for gds_threads in [benchmark_config.gds_io_threads[0]]:
                            backend_config = {
                                "chunk_size": chunk_size,
                                "cufile_buffer_size": cufile_buffer_size,
                                "gds_io_threads": gds_threads,
                            }

                            # Create populated backend
                            backend, keys, memory_objs = populated_backend(
                                batch_size, tensor_shape, backend_config
                            )

                            # Run benchmark
                            result = runner.run_benchmark(
                                function_name="batched_get_blocking",
                                func=backend.batched_get_blocking,
                                func_args=(keys,),
                                func_kwargs={},
                                batch_size=batch_size,
                                tensor_shape=tensor_shape,
                                backend_config=backend_config,
                                warmup_runs=benchmark_config.warmup_runs,
                                benchmark_runs=benchmark_config.benchmark_runs,
                            )

                            results.append(result)
                            if benchmark_config.verbose:
                                runner.print_summary(result)

        # Save results if requested
        if benchmark_config.save_results:
            filename = (
                f"batched_get_blocking_{request.node.nodeid.replace('::', '_')}.json"
            )
            runner.save_results(filename)
            print(f"\nResults saved to: {filename}")

        # Performance assertions (fail if too slow)
        for result in results:
            # Reasonable performance thresholds
            max_time_per_item = 0.1  # 100ms per item max
            expected_max_time = max_time_per_item * result.batch_size

            assert result.mean_time <= expected_max_time, (
                f"Performance regression detected: {result.mean_time:.3f}s > "
                f"{expected_max_time:.3f}s for batch_size={result.batch_size}"
            )

    def test_batched_get_blocking_scaling(
        self,
        benchmark_config: BenchmarkConfig,
        populated_backend,
    ):
        """Test how batched_get_blocking scales with batch size."""

        runner = BenchmarkRunner("scaling_analysis")
        results = []

        print("\n" + "=" * 80)
        print("SCALING ANALYSIS: batched_get_blocking()")
        print("=" * 80)

        # Fixed configuration for scaling test
        tensor_shape = benchmark_config.tensor_shapes[0]  # Use first shape
        backend_config = {
            "chunk_size": benchmark_config.chunk_sizes[0],
            "cufile_buffer_size": benchmark_config.cufile_buffer_sizes[0],
            "gds_io_threads": benchmark_config.gds_io_threads[0],
        }

        # Test scaling across batch sizes
        batch_sizes = [
            1,
            5,
            10,
            20,
            50,
            100,
            150,
            200,
            300,
            450,
            650,
            800,
            1000,
        ]  # Extended range for scaling

        for batch_size in batch_sizes:
            print(f"\n--- Scaling test: batch size {batch_size} ---")

            backend, keys = populated_backend(batch_size, tensor_shape, backend_config)

            # Wrapper to free memory objects between iterations
            def batched_get_with_cleanup(keys_arg, backend_arg):
                memory_objs = backend_arg.batched_get_blocking(keys_arg)
                if memory_objs:
                    for memory_obj in memory_objs:
                        if memory_obj is not None:
                            memory_obj.ref_count_down()
                return memory_objs

            result = runner.run_benchmark(
                function_name="batched_get_blocking_scaling",
                func=batched_get_with_cleanup,
                func_args=(keys, backend),
                func_kwargs={},
                batch_size=batch_size,
                tensor_shape=tensor_shape,
                backend_config=backend_config,
                warmup_runs=max(
                    1, benchmark_config.warmup_runs // 2
                ),  # Fewer warmup for large batches
                benchmark_runs=max(
                    3, benchmark_config.benchmark_runs // 2
                ),  # Fewer runs for large batches
            )

            results.append(result)
            runner.print_summary(result)

        # Analyze scaling efficiency
        print("\n" + "=" * 60)
        print("SCALING EFFICIENCY ANALYSIS")
        print("=" * 60)

        baseline = results[0]  # batch_size=1 baseline

        for result in results[1:]:
            expected_time = baseline.mean_time * result.batch_size
            actual_time = result.mean_time
            efficiency = expected_time / actual_time  # >1 means better than linear

            print(
                f"Batch {result.batch_size:3d}: "
                f"{actual_time * 1000:6.1f}ms "
                f"(efficiency: {efficiency:.2f}x)"
            )

            # Assert that we get at least some batching benefit for larger sizes
            if result.batch_size >= 10:
                min_efficiency = 1.5  # Should be at least 1.5x better than linear
                assert efficiency >= min_efficiency, (
                    f"Poor batching efficiency: {efficiency:.2f}x < {min_efficiency}x "
                    f"for batch_size={result.batch_size}"
                )

    @pytest.mark.slow
    def test_batched_get_blocking_comprehensive(
        self,
        benchmark_config: BenchmarkConfig,
        populated_backend,
        request,
    ):
        """Comprehensive benchmark testing different backend configurations."""

        runner = BenchmarkRunner("comprehensive")
        results = []

        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK: batched_get_blocking()")
        print("=" * 80)

        # Test matrix of configurations
        test_cases = [
            # (batch_size, tensor_shape, backend_config_name, backend_config)
            (
                512,
                (2, 16, 8, 128),
                "small_config",
                {"chunk_size": 256, "cufile_buffer_size": 8192, "gds_io_threads": 32},
            ),
            (
                512,
                (2, 32, 16, 128),
                "medium_config",
                {"chunk_size": 256, "cufile_buffer_size": 8192, "gds_io_threads": 32},
            ),
            (
                512,
                (2, 32, 32, 128),
                "large_config",
                {"chunk_size": 256, "cufile_buffer_size": 8192, "gds_io_threads": 32},
            ),
            (
                512,
                (2, 128, 128, 128),
                "vllm_sample_buffer",
                {"chunk_size": 256, "cufile_buffer_size": 8192, "gds_io_threads": 32},
            ),
        ]

        for batch_size, tensor_shape, config_name, backend_config in test_cases:
            print(
                f"\n--- Testing {config_name}: batch={batch_size}, "
                f"shape={tensor_shape} ---"
            )

            backend, keys = populated_backend(batch_size, tensor_shape, backend_config)

            # Wrapper to free memory objects between iterations
            def batched_get_with_cleanup(keys_arg, backend_arg):
                memory_objs = backend_arg.batched_get_blocking(keys_arg)
                # Free the returned memory objects immediately after the call
                if memory_objs:
                    for memory_obj in memory_objs:
                        if memory_obj is not None:
                            memory_obj.ref_count_down()
                return memory_objs

            result = runner.run_benchmark(
                function_name=f"batched_get_blocking_{config_name}",
                func=batched_get_with_cleanup,
                func_args=(keys, backend),
                func_kwargs={},
                batch_size=batch_size,
                tensor_shape=tensor_shape,
                backend_config=backend_config,
                warmup_runs=benchmark_config.warmup_runs,
                benchmark_runs=benchmark_config.benchmark_runs,
            )

            results.append(result)
            runner.print_summary(result)

            # Explicit cleanup to free cuFile buffer before next test case
            try:
                backend.close()
                # Also explicitly close the memory allocator
                if hasattr(backend, "memory_allocator"):
                    backend.memory_allocator.close()
            except Exception:
                pass

        # Save comprehensive results
        if benchmark_config.save_results:
            filename = "comprehensive_batched_get_blocking.json"
            runner.save_results(filename)
            print(f"\nComprehensive results saved to: {filename}")

        # Compare configurations
        print("\n" + "=" * 60)
        print("CONFIGURATION COMPARISON")
        print("=" * 60)

        for i, result in enumerate(results):
            config_name = test_cases[i][2]
            throughput = result.throughput_mbps or 0
            ops_per_sec = result.ops_per_second or 0

            print(
                f"{config_name:15s}: "
                f"{result.mean_time * 1000:6.1f}ms, "
                f"{throughput:6.1f} MB/s, "
                f"{ops_per_sec:6.1f} ops/s"
            )

    def test_single_vs_batched_comparison(
        self,
        benchmark_config: BenchmarkConfig,
        populated_backend,
    ):
        """Compare single get_blocking vs batched_get_blocking performance."""

        runner = BenchmarkRunner("single_vs_batched")

        print("\n" + "=" * 80)
        print("SINGLE vs BATCHED COMPARISON")
        print("=" * 80)

        batch_sizes_to_test = [5, 10, 20]
        tensor_shape = benchmark_config.tensor_shapes[0]
        backend_config = {
            "chunk_size": benchmark_config.chunk_sizes[0],
            "cufile_buffer_size": benchmark_config.cufile_buffer_sizes[0],
            "gds_io_threads": benchmark_config.gds_io_threads[0],
        }

        for batch_size in batch_sizes_to_test:
            print(f"\n--- Comparing single vs batched for {batch_size} items ---")

            backend, keys = populated_backend(batch_size, tensor_shape, backend_config)

            # Benchmark single get_blocking calls
            def single_gets_with_cleanup(keys_arg, backend_arg):
                results = []
                for key in keys_arg:
                    result = backend_arg.get_blocking(key)
                    results.append(result)
                # Free the returned memory objects immediately
                for result in results:
                    if result is not None:
                        result.ref_count_down()
                return results

            single_result = runner.run_benchmark(
                function_name="single_get_blocking",
                func=single_gets_with_cleanup,
                func_args=(keys, backend),
                func_kwargs={},
                batch_size=batch_size,
                tensor_shape=tensor_shape,
                backend_config=backend_config,
                warmup_runs=benchmark_config.warmup_runs,
                benchmark_runs=benchmark_config.benchmark_runs,
            )

            # Benchmark batched get_blocking
            def batched_get_with_cleanup(keys_arg, backend_arg):
                memory_objs = backend_arg.batched_get_blocking(keys_arg)
                if memory_objs:
                    for memory_obj in memory_objs:
                        if memory_obj is not None:
                            memory_obj.ref_count_down()
                return memory_objs

            batched_result = runner.run_benchmark(
                function_name="batched_get_blocking",
                func=batched_get_with_cleanup,
                func_args=(keys, backend),
                func_kwargs={},
                batch_size=batch_size,
                tensor_shape=tensor_shape,
                backend_config=backend_config,
                warmup_runs=benchmark_config.warmup_runs,
                benchmark_runs=benchmark_config.benchmark_runs,
            )

            # Compare results
            speedup = single_result.mean_time / batched_result.mean_time
            print(f"Single calls:   {single_result.mean_time * 1000:.2f} ms")
            print(f"Batched call:   {batched_result.mean_time * 1000:.2f} ms")
            print(f"Speedup:        {speedup:.2f}x")

            # Assert that batching provides benefit
            min_speedup = 1.5  # Should be at least 1.5x faster
            assert speedup >= min_speedup, (
                f"Insufficient batching benefit: {speedup:.2f}x < {min_speedup}x "
                f"for batch_size={batch_size}"
            )


# Utility test for running quick benchmarks during development
@pytest.mark.benchmark
def test_quick_benchmark(populated_backend):
    """Quick benchmark for development/debugging."""

    batch_size = 512
    tensor_shape = (2, 128, 128, 128)
    backend_config = {
        "chunk_size": 256,
        "cufile_buffer_size": 8192,
        "gds_io_threads": 32,
    }

    backend, keys = populated_backend(batch_size, tensor_shape, backend_config)

    # Wrapper to free memory objects between iterations
    def batched_get_with_cleanup(keys_arg, backend_arg):
        memory_objs = backend_arg.batched_get_blocking(keys_arg)
        if memory_objs:
            for memory_obj in memory_objs:
                if memory_obj is not None:
                    memory_obj.ref_count_down()
        return memory_objs

    runner = BenchmarkRunner("quick_test")
    result = runner.run_benchmark(
        function_name="batched_get_blocking_quick",
        func=batched_get_with_cleanup,
        func_args=(keys, backend),
        func_kwargs={},
        batch_size=batch_size,
        tensor_shape=tensor_shape,
        backend_config=backend_config,
        warmup_runs=5,
        benchmark_runs=100,
    )

    runner.print_summary(result)

    # Basic sanity check
    assert result.mean_time > 0
    assert result.mean_time < 0.250  # Should not take more than 0.250 seconds
    assert len(result.raw_times) == 100
