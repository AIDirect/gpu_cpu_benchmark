# gpu-benchmark

Benchmark script to measure GPU and CPU performance across environments.

## Benchmarks

**CPU:**

| Benchmark | Description |
|-----------|-------------|
| `cpu_matmul` | 4096x4096 matrix multiply (NumPy, multi-core BLAS) |
| `cpu_single_thread` | Monte Carlo pi estimation (pure Python, single-threaded) |
| `cpu_multi_core` | 2048x2048 SVD (NumPy, multi-core BLAS) |

**GPU (CUDA):**

| Benchmark | Description |
|-----------|-------------|
| `gpu_matmul` | 4096x4096 matrix multiply (PyTorch) |
| `gpu_matmul_large` | 8192x8192 matrix multiply (PyTorch) |
| `gpu_conv2d` | Conv2d 64x64x256x256, 128 output channels, kernel 3 |
| `gpu_memory_bandwidth` | 256 MB host-to-device and device-to-host transfer |

Each benchmark runs warmup iterations followed by timed repeats, reporting **median**, **mean**, and **std** of the measured times.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.10.

Install with a CUDA extra matching your GPU driver:

```bash
# CUDA 13.0
uv sync --extra cu130

# CUDA 12.8
uv sync --extra cu128

# CUDA 12.6
uv sync --extra cu126

# CUDA 12.4
uv sync --extra cu124

# CUDA 12.1
uv sync --extra cu121

# CUDA 11.8
uv sync --extra cu118

# CPU only (no CUDA)
uv sync --extra cpu
```

## Usage

```bash
# Run with defaults (5 repeats per benchmark)
uv run benchmark

# More repeats for higher accuracy
uv run benchmark --repeats 10

# Quick run
uv run benchmark --repeats 2
```

Results are printed to the console and saved as a timestamped JSON file (e.g., `benchmark_results_hostname_20260408T120000Z.json`).

## Output

```
============================================================
  GPU + CPU Benchmark
============================================================

  Host:   my-machine
  OS:     Linux 6.17.0-19-generic
  CPU:    x86_64
  Cores:  32
  GPU:    NVIDIA GeForce RTX 5090
  VRAM:   31.35 GB
  Python: 3.10.19  NumPy: 2.2.6  PyTorch: 2.11.0+cu130

------------------------------------------------------------
  CPU Benchmarks
------------------------------------------------------------
  Running cpu_matmul... 0.0866s (mean=0.0873s, std=0.0035s)  (1586.34 GFLOPS)
  Running cpu_single_thread... 0.4068s (mean=0.4085s, std=0.0052s)  (12,291,041 samples/s)
  Running cpu_multi_core... 33.8978s (mean=34.3025s, std=13.9892s)

------------------------------------------------------------
  GPU Benchmarks
------------------------------------------------------------
  Running gpu_matmul... 0.0021s (mean=0.0021s, std=0.0s)  (66412.38 GFLOPS)
  Running gpu_matmul_large... 0.0162s (mean=0.016s, std=0.0003s)  (67828.18 GFLOPS)
  Running gpu_conv2d... 0.0107s (mean=0.0107s, std=0.0s)
  Running gpu_memory_bandwidth... H2D 44.98 GB/s (mean=0.0055s, std=0.0s)  D2H 4.09 GB/s (mean=0.0615s, std=0.0023s)

============================================================
  Results saved to benchmark_results_my-machine_20260408T073625Z.json
============================================================
```
