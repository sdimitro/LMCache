# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

# First Party
from lmcache.observability import LMCStatsMonitor


@pytest.fixture(scope="function")
def stats_monitor():
    LMCStatsMonitor.DestroyInstance()
    return LMCStatsMonitor.GetOrCreate()


def test_on_retrieve_request(stats_monitor):
    stats_monitor.on_retrieve_request(num_tokens=100)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_retrieve_requests == 1
    assert stats.retrieve_hit_rate == 0
    assert stats.local_cache_usage_bytes == 0
    assert stats.remote_cache_usage_bytes == 0
    assert len(stats.time_to_retrieve) == 0


def test_on_retrieve_finished(stats_monitor):
    request_id = stats_monitor.on_retrieve_request(num_tokens=100)
    stats_monitor.on_retrieve_finished(
        request_id=request_id,
        retrieved_tokens=100,
    )
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_retrieve_requests == 1
    assert stats.retrieve_hit_rate == 1.0
    assert len(stats.time_to_retrieve) == 1


def test_on_store_request_and_finished(stats_monitor):
    request_id = stats_monitor.on_store_request(num_tokens=50)
    stats_monitor.on_store_finished(request_id=request_id)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_store_requests == 1
    assert stats.interval_store_requested_tokens == 50
    assert stats.interval_store_stored_tokens == 50
    assert len(stats.time_to_store) == 1


def test_on_store_request_partial_store(stats_monitor):
    # Test case where fewer tokens are stored than requested
    request_id = stats_monitor.on_store_request(num_tokens=100)
    stats_monitor.on_store_finished(request_id=request_id, num_tokens=60)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_store_requests == 1
    assert stats.interval_store_requested_tokens == 100
    assert stats.interval_store_stored_tokens == 60
    assert len(stats.time_to_store) == 1


def test_update_local_cache_usage(stats_monitor):
    stats_monitor.update_local_cache_usage(usage=1024)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.local_cache_usage_bytes == 1024


def test_update_remote_cache_usage(stats_monitor):
    stats_monitor.update_remote_cache_usage(usage=2048)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.remote_cache_usage_bytes == 2048


def test_update_local_storage_usage(stats_monitor):
    stats_monitor.update_local_storage_usage(usage=4096)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.local_storage_usage_bytes == 4096


def test_on_lookup_request(stats_monitor):
    stats_monitor.on_lookup_request(num_tokens=50)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 1
    assert stats.interval_lookup_requested_tokens == 50
    assert stats.lookup_hit_rate == 0


def test_on_lookup_finished(stats_monitor):
    stats_monitor.on_lookup_request(num_tokens=100)
    stats_monitor.on_lookup_finished(num_hit_tokens=80)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 1
    assert stats.interval_lookup_requested_tokens == 100
    assert stats.interval_lookup_hit_tokens == 80
    assert stats.lookup_hit_rate == 0.8


def test_remote_read_metrics(stats_monitor):
    stats_monitor.update_interval_remote_read_metrics(read_bytes=1024)
    stats_monitor.update_interval_remote_read_metrics(read_bytes=2048)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_read_requests == 2
    assert stats.interval_remote_read_bytes == 3072


def test_remote_write_metrics(stats_monitor):
    stats_monitor.update_interval_remote_write_metrics(write_bytes=512)
    stats_monitor.update_interval_remote_write_metrics(write_bytes=1024)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_write_requests == 2
    assert stats.interval_remote_write_bytes == 1536


def test_remote_time_metrics(stats_monitor):
    stats_monitor.update_interval_remote_time_to_get(get_time=10.5)
    stats_monitor.update_interval_remote_time_to_put(put_time=15.2)
    stats_monitor.update_interval_remote_time_to_get_sync(get_time_sync=12.3)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_time_to_get == [10.5]
    assert stats.interval_remote_time_to_put == [15.2]
    assert stats.interval_remote_time_to_get_sync == [12.3]


def test_remote_ping_metrics(stats_monitor):
    # Test successful ping
    stats_monitor.update_remote_ping_latency(latency=25.5)
    stats_monitor.update_remote_ping_error_code(error_code=0)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_ping_latency == 25.5
    assert stats.interval_remote_ping_success == 1
    assert stats.interval_remote_ping_errors == 0
    assert stats.interval_remote_ping_error_code == 0


def test_remote_ping_errors(stats_monitor):
    # Test ping errors
    stats_monitor.update_remote_ping_error_code(error_code=404)
    stats_monitor.update_remote_ping_error_code(error_code=500)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_ping_errors == 2
    assert stats.interval_remote_ping_success == 0
    assert stats.interval_remote_ping_error_code == 500


def test_retrieve_and_store_speed(stats_monitor):
    # Test retrieve speed calculation
    retrieve_id = stats_monitor.on_retrieve_request(num_tokens=1000)
    stats_monitor.on_retrieve_finished(request_id=retrieve_id, retrieved_tokens=1000)

    # Test store speed calculation
    store_id = stats_monitor.on_store_request(num_tokens=500)
    stats_monitor.on_store_finished(request_id=store_id)

    stats = stats_monitor.get_stats_and_clear()
    assert len(stats.retrieve_speed) == 1
    assert len(stats.store_speed) == 1
    assert stats.retrieve_speed[0] > 0  # Should be tokens/second
    assert stats.store_speed[0] > 0  # Should be tokens/second


def test_multiple_lookup_operations(stats_monitor):
    # Test multiple lookup operations
    stats_monitor.on_lookup_request(num_tokens=100)
    stats_monitor.on_lookup_finished(num_hit_tokens=80)
    stats_monitor.on_lookup_request(num_tokens=200)
    stats_monitor.on_lookup_finished(num_hit_tokens=150)

    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 2
    assert stats.interval_lookup_requested_tokens == 300
    assert stats.interval_lookup_hit_tokens == 230
    assert stats.lookup_hit_rate == 230 / 300


def test_mixed_remote_operations(stats_monitor):
    # Test a mix of remote operations
    stats_monitor.update_interval_remote_read_metrics(read_bytes=1024)
    stats_monitor.update_interval_remote_write_metrics(write_bytes=512)
    stats_monitor.update_interval_remote_time_to_get(get_time=10.0)
    stats_monitor.update_interval_remote_time_to_put(put_time=20.0)
    stats_monitor.update_interval_remote_time_to_get_sync(get_time_sync=15.0)
    stats_monitor.update_remote_ping_latency(latency=30.0)
    stats_monitor.update_remote_ping_error_code(error_code=0)

    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_read_requests == 1
    assert stats.interval_remote_read_bytes == 1024
    assert stats.interval_remote_write_requests == 1
    assert stats.interval_remote_write_bytes == 512
    assert stats.interval_remote_time_to_get == [10.0]
    assert stats.interval_remote_time_to_put == [20.0]
    assert stats.interval_remote_time_to_get_sync == [15.0]
    assert stats.interval_remote_ping_latency == 30.0
    assert stats.interval_remote_ping_success == 1
    assert stats.interval_remote_ping_errors == 0


def test_combined_operations(stats_monitor):
    retrieve_id = stats_monitor.on_retrieve_request(num_tokens=200)
    stats_monitor.on_retrieve_finished(
        request_id=retrieve_id,
        retrieved_tokens=200,
    )
    store_id = stats_monitor.on_store_request(num_tokens=100)
    stats_monitor.on_store_finished(store_id)
    stats_monitor.update_local_cache_usage(usage=512)
    stats_monitor.update_remote_cache_usage(usage=1024)
    stats_monitor.update_local_storage_usage(usage=2048)

    stats_monitor2 = LMCStatsMonitor.GetOrCreate()
    stats = stats_monitor2.get_stats_and_clear()

    assert stats.interval_retrieve_requests == 1
    assert stats.interval_store_requests == 1
    assert stats.retrieve_hit_rate == 1.0
    assert stats.local_cache_usage_bytes == 512
    assert stats.remote_cache_usage_bytes == 1024
    assert stats.local_storage_usage_bytes == 2048
    assert len(stats.time_to_retrieve) == 1
    assert len(stats.time_to_store) == 1


def test_stats_clearing(stats_monitor):
    # Add some data
    stats_monitor.on_lookup_request(num_tokens=100)
    stats_monitor.update_interval_remote_read_metrics(read_bytes=1024)
    stats_monitor.update_remote_ping_latency(latency=25.0)

    # Get stats (which should clear them)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 1
    assert stats.interval_remote_read_requests == 1
    assert stats.interval_remote_ping_latency == 25.0

    # Get stats again - should be cleared
    stats2 = stats_monitor.get_stats_and_clear()
    assert stats2.interval_lookup_requests == 0
    assert stats2.interval_remote_read_requests == 0
    assert stats2.interval_remote_ping_latency == 0


def test_zero_division_protection(stats_monitor):
    # Test that hit rates handle zero division gracefully
    stats = stats_monitor.get_stats_and_clear()
    assert stats.retrieve_hit_rate == 0
    assert stats.lookup_hit_rate == 0


def test_backend_specific_lookup_metrics(stats_monitor):
    stats_monitor.on_lookup_request(num_tokens=100)
    backend_hits = {"LocalCPUBackend": 60, "WekaGdsBackend": 40}
    stats_monitor.on_lookup_finished(num_hit_tokens=100, backend_hits=backend_hits)

    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_hit_tokens == 100
    assert stats.backend_lookup_hit_tokens == {
        "LocalCPUBackend": 60,
        "WekaGdsBackend": 40,
    }


def test_backend_specific_retrieve_metrics(stats_monitor):
    request_id = stats_monitor.on_retrieve_request(num_tokens=150)
    backend_tokens = {"LocalCPUBackend": 90, "LocalDiskBackend": 60}
    stats_monitor.on_retrieve_finished(
        request_id=request_id, retrieved_tokens=150, backend_tokens=backend_tokens
    )

    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_retrieve_retrieved_tokens == 150
    assert stats.backend_retrieve_retrieved_tokens == {
        "LocalCPUBackend": 90,
        "LocalDiskBackend": 60,
    }


def test_backend_specific_store_metrics(stats_monitor):
    # Test per-backend store metrics
    request_id = stats_monitor.on_store_request(num_tokens=200)
    # Single call with backend list
    backends_used = ["LocalCPUBackend", "LocalDiskBackend", "RemoteBackend"]
    stats_monitor.on_store_finished(
        request_id=request_id, num_tokens=200, backends=backends_used
    )

    stats = stats_monitor.get_stats_and_clear()
    assert (
        stats.interval_store_stored_tokens == 200
    )  # Total (not summed across backends)
    assert stats.backend_store_stored_tokens == {
        "LocalCPUBackend": 200,
        "LocalDiskBackend": 200,
        "RemoteBackend": 200,
    }


def test_backend_latency_metrics(stats_monitor):
    # Test per-backend latency tracking

    # Test retrieve latencies
    request_id = stats_monitor.on_retrieve_request(num_tokens=150)
    backend_tokens = {"LocalCPUBackend": 150}
    retrieve_latencies = {"LocalCPUBackend": 2.3, "LocalDiskBackend": 15.8}
    stats_monitor.on_retrieve_finished(
        request_id=request_id,
        retrieved_tokens=150,
        backend_tokens=backend_tokens,
        backend_latencies=retrieve_latencies,
    )

    # Test store latencies
    store_id = stats_monitor.on_store_request(num_tokens=200)
    backends_used = ["LocalCPUBackend", "RemoteBackend"]
    store_latencies = {"LocalCPUBackend": 3.1, "RemoteBackend": 25.6}
    stats_monitor.on_store_finished(
        request_id=store_id,
        num_tokens=200,
        backends=backends_used,
        backend_latencies=store_latencies,
    )

    stats = stats_monitor.get_stats_and_clear()

    # Check get latencies were recorded
    assert "LocalCPUBackend" in stats.backend_get_latencies
    assert "LocalDiskBackend" in stats.backend_get_latencies
    assert stats.backend_get_latencies["LocalCPUBackend"] == [2.3]
    assert stats.backend_get_latencies["LocalDiskBackend"] == [15.8]

    # Check put latencies were recorded
    assert "LocalCPUBackend" in stats.backend_put_latencies
    assert "RemoteBackend" in stats.backend_put_latencies
    assert stats.backend_put_latencies["LocalCPUBackend"] == [3.1]
    assert stats.backend_put_latencies["RemoteBackend"] == [25.6]


def test_weka_gds_metrics(stats_monitor):
    # Test Weka GDS read metrics
    stats_monitor.update_weka_gds_read_metrics(read_ops=5, read_bytes=1024000)
    stats_monitor.update_weka_gds_read_metrics(read_ops=3, read_bytes=512000)

    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_weka_gds_read_ops == 8
    assert stats.interval_weka_gds_read_bytes == 1536000
