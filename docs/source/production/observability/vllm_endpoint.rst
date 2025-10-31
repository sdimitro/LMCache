.. _observability_vllm_endpoint:

Metrics by vLLM API
==========================================

LMCache provides detailed metrics via a Prometheus endpoint, allowing for in-depth monitoring of cache performance and behavior.
This section outlines how to enable and configure observability from embedded vLLM ``/metrics`` API endpoint.


Quick Start Guide
-----

1) On vLLM/LMCache side
^^^^^^^^^^^^^^^^^^^^^^^

In v1, vLLM and LMCache run in separate processes, so you have to use multi‑process Prometheus.

The ``PROMETHEUS_MULTIPROC_DIR`` environment variable must be the same in both processes, as a IPC directory.

.. code-block:: bash

   PROMETHEUS_MULTIPROC_DIR=/tmp/lmcache_prometheus \
   #.. other environment variables \
   vllm serve $MODEL -port 8000 ...

Once the HTTP server is running, you can access the LMCache metrics at the ``/metrics`` endpoint.

.. code-block:: bash

   curl http://$<vllm-worker-ip>:8000/metrics | grep lmcache

   # Replace $IP with the IP address of a vLLM worker


And you will also find some ``.db`` files in the ``$PROMETHEUS_MULTIPROC_DIR`` directory.


2) Prometheus Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To scrape the LMCache metrics with a Prometheus server, add the following job to your ``prometheus.yml`` configuration,
or equivalent configuration to scrape the metrics endpoint:

.. code-block:: yaml

   scrape_configs:
     - job_name: 'lmcache'
       static_configs:
         - targets: ['<vllm-worker-ip>:8000']
       scrape_interval: 15s

Available Metrics
-----------------

LMCache exposes a variety of metrics to monitor its performance. The following table lists all available metrics organized by category:

.. list-table:: LMCache Metrics
   :header-rows: 1
   :widths: 30 15 55

   * - Metric Name
     - Type
     - Description
   * - **Core Request Metrics**
     - 
     - 
   * - ``lmcache:num_retrieve_requests``
     - Counter
     - Total number of retrieve requests
   * - ``lmcache:num_store_requests``
     - Counter
     - Total number of store requests
   * - ``lmcache:num_lookup_requests``
     - Counter
     - Total number of lookup requests
   * - ``lmcache:num_retrieve_requested_tokens``
     - Counter
     - Total number of tokens requested in retrieve operations
   * - ``lmcache:num_retrieve_retrieved_tokens``
     - Counter
     - Total number of tokens successfully retrieved
   * - ``lmcache:num_store_requested_tokens``
     - Counter
     - Total number of tokens requested to store
   * - ``lmcache:num_store_stored_tokens``
     - Counter
     - Total number of tokens actually stored
   * - ``lmcache:num_lookup_requested_tokens``
     - Counter
     - Total number of tokens requested in lookup operations
   * - ``lmcache:num_lookup_hit_tokens``
     - Counter
     - Total number of tokens hit in lookup operations
   * - ``lmcache:num_vllm_hit_tokens``
     - Counter
     - Number of hit tokens in vLLM
   * - **Per-Backend Token Metrics**
     - 
     - 
   * - ``lmcache:num_lookup_hit_tokens_by_backend``
     - Counter
     - Total number of tokens hit in lookup by backend (labeled by backend name)
   * - ``lmcache:num_retrieve_retrieved_tokens_by_backend``
     - Counter
     - Total number of tokens retrieved by backend (labeled by backend name)
   * - ``lmcache:num_store_stored_tokens_by_backend``
     - Counter
     - Total number of tokens stored by backend (labeled by backend name)
   * - **Per-Backend Latency Metrics**
     - 
     - 
   * - ``lmcache:backend_get_latency_ms``
     - Histogram
     - Latency of get/retrieve operations per backend in milliseconds (labeled by backend name)
   * - ``lmcache:backend_put_latency_ms``
     - Histogram
     - Latency of put/store operations per backend in milliseconds (labeled by backend name)
   * - **Hit Rate Metrics**
     - 
     - 
   * - ``lmcache:retrieve_hit_rate``
     - Gauge
     - The hit rate for retrieve requests
   * - ``lmcache:lookup_hit_rate``
     - Gauge
     - The hit rate for lookup requests
   * - **Cache Usage Metrics**
     - 
     - 
   * - ``lmcache:local_cache_usage``
     - Gauge
     - Local cache usage in bytes
   * - ``lmcache:remote_cache_usage``
     - Gauge
     - Remote cache usage in bytes
   * - ``lmcache:local_storage_usage``
     - Gauge
     - Local storage usage in bytes
   * - **Performance Metrics**
     - 
     - 
   * - ``lmcache:time_to_retrieve``
     - Histogram
     - Time taken to retrieve from the cache (seconds)
   * - ``lmcache:time_to_store``
     - Histogram
     - Time taken to store to the cache (seconds)
   * - ``lmcache:retrieve_speed``
     - Histogram
     - Retrieval speed (tokens per second)
   * - ``lmcache:store_speed``
     - Histogram
     - Storage speed (tokens per second)
   * - **Remote Backend Metrics**
     - 
     - 
   * - ``lmcache:num_remote_read_requests``
     - Counter
     - Total number of read requests to remote backends
   * - ``lmcache:num_remote_read_bytes``
     - Counter
     - Total number of bytes read from remote backends
   * - ``lmcache:num_remote_write_requests``
     - Counter
     - Total number of write requests to remote backends
   * - ``lmcache:num_remote_write_bytes``
     - Counter
     - Total number of bytes written to remote backends
   * - ``lmcache:remote_time_to_get``
     - Histogram
     - Time taken to get data from remote backends (milliseconds)
   * - ``lmcache:remote_time_to_put``
     - Histogram
     - Time taken to put data to remote backends (milliseconds)
   * - ``lmcache:remote_time_to_get_sync``
     - Histogram
     - Time taken to get data from remote backends synchronously (milliseconds)
   * - **Network Monitoring Metrics**
     - 
     - 
   * - ``lmcache:remote_ping_latency``
     - Gauge
     - Latest ping latency to remote backends (milliseconds)
   * - ``lmcache:remote_ping_errors``
     - Counter
     - Number of ping errors to remote backends
   * - ``lmcache:remote_ping_successes``
     - Counter
     - Number of ping successes to remote backends
   * - ``lmcache:remote_ping_error_code``
     - Gauge
     - Latest ping error code to remote backends
   * - **Local CPU Backend Metrics**
     - 
     - 
   * - ``lmcache:local_cpu_evict_count``
     - Counter
     - Total number of evictions in local CPU backend
   * - ``lmcache:local_cpu_evict_keys_count``
     - Counter
     - Total number of evicted keys in local CPU backend
   * - ``lmcache:local_cpu_evict_failed_count``
     - Counter
     - Total number of failed evictions in local CPU backend
   * - ``lmcache:local_cpu_hot_cache_count``
     - Gauge
     - The size of the hot cache
   * - ``lmcache:local_cpu_keys_in_request_count``
     - Gauge
     - The size of the keys in request
   * - **Weka GDS Metrics**
     - 
     - 
   * - ``lmcache:weka_gds_read_ops``
     - Counter
     - Total number of GDS read operations
   * - ``lmcache:weka_gds_read_bytes``
     - Counter
     - Total bytes read via GDS
   * - ``lmcache:weka_gds_errors_total``
     - Counter
     - Total Weka GDS errors by type (labeled by error_type: timeout, alloc_failures, threshold, io_failures)
   * - **Memory Management Metrics**
     - 
     - 
   * - ``lmcache:active_memory_objs_count``
     - Gauge
     - The number of active memory objects
   * - ``lmcache:pinned_memory_objs_count``
     - Gauge
     - The number of pinned memory objects


Per-Backend Metrics Examples
-----------------------------

The per-backend metrics allow you to break down performance by storage backend. Here are some example Prometheus queries:

**View lookup hits by backend:**

.. code-block:: promql

   # Total lookup hits per backend
   lmcache:num_lookup_hit_tokens_by_backend{backend="LocalCPUBackend"}
   lmcache:num_lookup_hit_tokens_by_backend{backend="WekaGdsBackend"}
   lmcache:num_lookup_hit_tokens_by_backend{backend="LocalDiskBackend"}

**View retrieve performance by backend:**

.. code-block:: promql

   # Rate of tokens retrieved per backend (tokens/sec)
   rate(lmcache:num_retrieve_retrieved_tokens_by_backend[5m])
   
   # Compare backends
   sum by (backend) (rate(lmcache:num_retrieve_retrieved_tokens_by_backend[5m]))

**View store distribution across backends:**

.. code-block:: promql

   # Total tokens stored per backend
   lmcache:num_store_stored_tokens_by_backend
   
   # Percentage of stores going to each backend
   lmcache:num_store_stored_tokens_by_backend / ignoring(backend) group_left sum(lmcache:num_store_stored_tokens_by_backend)

**Backend hit rate comparison:**

.. code-block:: promql

   # Which backend serves most lookups?
   topk(3, sum by (backend) (rate(lmcache:num_lookup_hit_tokens_by_backend[5m])))

**Backend latency analysis:**

.. code-block:: promql

   # p50 latency for each backend's get operations
   histogram_quantile(0.50, sum by (backend, le) (rate(lmcache:backend_get_latency_ms_bucket[5m])))
   
   # p95 latency for each backend's get operations
   histogram_quantile(0.95, sum by (backend, le) (rate(lmcache:backend_get_latency_ms_bucket[5m])))
   
   # p99 latency for WekaGdsBackend specifically
   histogram_quantile(0.99, rate(lmcache:backend_get_latency_ms_bucket{backend="WekaGdsBackend"}[5m]))
   
   # Identify slowest backend for store operations
   topk(1, histogram_quantile(0.99, sum by (backend, le) (rate(lmcache:backend_put_latency_ms_bucket[5m]))))

**Aggregate metrics (without backend label) are still available:**

.. code-block:: promql

   # Total tokens retrieved across all backends
   lmcache:num_retrieve_retrieved_tokens
   
   # Overall hit rate
   lmcache:retrieve_hit_rate

**Weka GDS performance metrics:**

.. code-block:: promql

   # GDS read throughput (GB/s)
   rate(lmcache:weka_gds_read_bytes[5m]) / 1024 / 1024 / 1024
   
   # GDS read operations per second
   rate(lmcache:weka_gds_read_ops[5m])
   
   # Average bytes per GDS read operation
   rate(lmcache:weka_gds_read_bytes[5m]) / rate(lmcache:weka_gds_read_ops[5m])

**Weka GDS error tracking:**

.. code-block:: promql

   # Total errors by type (stacked area chart)
   sum by (error_type) (rate(lmcache:weka_gds_errors_total[5m]))
   
   # Timeout errors per second
   rate(lmcache:weka_gds_errors_total{error_type="timeout"}[5m])
   
   # Allocation failures per second
   rate(lmcache:weka_gds_errors_total{error_type="alloc_failures"}[5m])
   
   # I/O failures per second
   rate(lmcache:weka_gds_errors_total{error_type="io_failures"}[5m])
   
   # Alert: High error rate
   sum(rate(lmcache:weka_gds_errors_total[5m])) > 1


