import argparse
import json
import math
import platform
import random
import socket
import time
from datetime import datetime, timezone

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_REPEATS = 5


def timed(fn, warmup=2, repeats=None):
    """Run *fn* with warmup iterations, then return (median, mean, std) in seconds."""
    if repeats is None:
        repeats = DEFAULT_REPEATS
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return median, mean, std


def gflops_matmul(n, seconds):
    """GFLOPS for an (n x n) @ (n x n) matrix multiply."""
    return 2 * n**3 / seconds / 1e9


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def collect_system_info():
    info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": None,
        "cpu_cores_logical": None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": None,
        "gpu_vram_gb": None,
    }
    try:
        import os
        info["cpu_cores_physical"] = os.cpu_count()  # logical; psutil would give physical
        info["cpu_cores_logical"] = os.cpu_count()
    except Exception:
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
    return info


# ---------------------------------------------------------------------------
# CPU benchmarks
# ---------------------------------------------------------------------------

def cpu_matmul(n=4096):
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    def run():
        np.dot(a, b)

    median, mean, std = timed(run)
    return {
        "name": f"CPU matrix multiply ({n}x{n})",
        "time_s": round(median, 4),
        "time_mean_s": round(mean, 4),
        "time_std_s": round(std, 4),
        "gflops": round(gflops_matmul(n, median), 2),
    }


def cpu_single_thread(n_samples=5_000_000):
    """Approximate pi via Monte Carlo (pure Python, single-threaded)."""
    def run():
        inside = 0
        for _ in range(n_samples):
            x = random.random()
            y = random.random()
            if x * x + y * y <= 1.0:
                inside += 1
        return 4.0 * inside / n_samples

    median, mean, std = timed(run, warmup=1)
    return {
        "name": f"CPU single-thread Monte Carlo pi ({n_samples:,} samples)",
        "time_s": round(median, 4),
        "time_mean_s": round(mean, 4),
        "time_std_s": round(std, 4),
        "samples_per_sec": round(n_samples / median, 0),
    }


def cpu_multi_core():
    """NumPy SVD on a large matrix — exercises multiple cores via BLAS."""
    n = 2048
    a = np.random.randn(n, n).astype(np.float32)

    def run():
        np.linalg.svd(a, full_matrices=False)

    median, mean, std = timed(run, warmup=1)
    return {
        "name": f"CPU SVD ({n}x{n}, multi-core BLAS)",
        "time_s": round(median, 4),
        "time_mean_s": round(mean, 4),
        "time_std_s": round(std, 4),
    }


# ---------------------------------------------------------------------------
# GPU benchmarks
# ---------------------------------------------------------------------------

def gpu_matmul(n=4096):
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)

    def run():
        torch.mm(a, b)
        torch.cuda.synchronize()

    median, mean, std = timed(run)
    return {
        "name": f"GPU matrix multiply ({n}x{n})",
        "time_s": round(median, 4),
        "time_mean_s": round(mean, 4),
        "time_std_s": round(std, 4),
        "gflops": round(gflops_matmul(n, median), 2),
    }


def gpu_matmul_large(n=8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)

    def run():
        torch.mm(a, b)
        torch.cuda.synchronize()

    median, mean, std = timed(run)
    return {
        "name": f"GPU matrix multiply ({n}x{n})",
        "time_s": round(median, 4),
        "time_mean_s": round(mean, 4),
        "time_std_s": round(std, 4),
        "gflops": round(gflops_matmul(n, median), 2),
    }


def gpu_conv2d():
    """Simulate a CNN-style 2D convolution on GPU."""
    batch, channels, h, w = 64, 64, 256, 256
    out_channels, kernel = 128, 3
    x = torch.randn(batch, channels, h, w, device="cuda", dtype=torch.float32)
    conv = torch.nn.Conv2d(channels, out_channels, kernel, padding=1).cuda()

    def run():
        conv(x)
        torch.cuda.synchronize()

    median, mean, std = timed(run)
    return {
        "name": f"GPU Conv2d ({batch}x{channels}x{h}x{w}, k={kernel})",
        "time_s": round(median, 4),
        "time_mean_s": round(mean, 4),
        "time_std_s": round(std, 4),
    }


def gpu_memory_bandwidth():
    """Measure host-to-device and device-to-host transfer speed."""
    size_mb = 256
    n_elements = size_mb * 1024 * 1024 // 4  # float32
    host_tensor = torch.randn(n_elements, dtype=torch.float32, pin_memory=True)

    # Host → Device
    def h2d():
        d = host_tensor.to("cuda", non_blocking=False)
        torch.cuda.synchronize()

    h2d_median, h2d_mean, h2d_std = timed(h2d)
    h2d_gbps = size_mb / 1024 / h2d_median

    # Device → Host
    device_tensor = host_tensor.to("cuda")
    torch.cuda.synchronize()

    def d2h():
        _ = device_tensor.to("cpu", non_blocking=False)
        torch.cuda.synchronize()

    d2h_median, d2h_mean, d2h_std = timed(d2h)
    d2h_gbps = size_mb / 1024 / d2h_median

    return {
        "name": f"GPU memory transfer ({size_mb} MB)",
        "h2d_time_s": round(h2d_median, 4),
        "h2d_time_mean_s": round(h2d_mean, 4),
        "h2d_time_std_s": round(h2d_std, 4),
        "h2d_gbps": round(h2d_gbps, 2),
        "d2h_time_s": round(d2h_median, 4),
        "d2h_time_mean_s": round(d2h_mean, 4),
        "d2h_time_std_s": round(d2h_std, 4),
        "d2h_gbps": round(d2h_gbps, 2),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmarks():
    print("=" * 60)
    print("  GPU + CPU Benchmark")
    print("=" * 60)

    info = collect_system_info()
    print(f"\n  Host:   {info['hostname']}")
    print(f"  OS:     {info['os']}")
    print(f"  CPU:    {info['cpu']}")
    print(f"  Cores:  {info['cpu_cores_logical']}")
    print(f"  GPU:    {info['gpu'] or 'N/A'}")
    if info["gpu_vram_gb"]:
        print(f"  VRAM:   {info['gpu_vram_gb']} GB")
    print(f"  Python: {info['python']}  NumPy: {info['numpy']}  PyTorch: {info['torch']}")
    print()

    results = []

    # --- CPU ---
    print("-" * 60)
    print("  CPU Benchmarks")
    print("-" * 60)

    for bench_fn in [cpu_matmul, cpu_single_thread, cpu_multi_core]:
        print(f"  Running {bench_fn.__name__}...", end=" ", flush=True)
        r = bench_fn()
        results.append(r)
        summary = f"{r['time_s']}s (mean={r['time_mean_s']}s, std={r['time_std_s']}s)"
        if "gflops" in r:
            summary += f"  ({r['gflops']} GFLOPS)"
        if "samples_per_sec" in r:
            summary += f"  ({r['samples_per_sec']:,.0f} samples/s)"
        print(summary)

    # --- GPU ---
    if torch.cuda.is_available():
        print()
        print("-" * 60)
        print("  GPU Benchmarks")
        print("-" * 60)

        for bench_fn in [gpu_matmul, gpu_matmul_large, gpu_conv2d, gpu_memory_bandwidth]:
            print(f"  Running {bench_fn.__name__}...", end=" ", flush=True)
            r = bench_fn()
            results.append(r)
            if "h2d_gbps" in r:
                summary = (f"H2D {r['h2d_gbps']} GB/s (mean={r['h2d_time_mean_s']}s, std={r['h2d_time_std_s']}s)  "
                           f"D2H {r['d2h_gbps']} GB/s (mean={r['d2h_time_mean_s']}s, std={r['d2h_time_std_s']}s)")
            else:
                summary = f"{r['time_s']}s (mean={r['time_mean_s']}s, std={r['time_std_s']}s)"
                if "gflops" in r:
                    summary += f"  ({r['gflops']} GFLOPS)"
            print(summary)
    else:
        print("\n  [GPU benchmarks skipped — no CUDA device available]")

    # --- Save JSON ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hostname = info["hostname"].replace(" ", "_")
    filename = f"benchmark_results_{hostname}_{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "system": info,
        "benchmarks": results,
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 60)
    print(f"  Results saved to {filename}")
    print("=" * 60)


def main():
    global DEFAULT_REPEATS
    parser = argparse.ArgumentParser(description="GPU + CPU Benchmark")
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"Number of timed repeats per benchmark (default: {DEFAULT_REPEATS})",
    )
    args = parser.parse_args()
    DEFAULT_REPEATS = args.repeats
    run_benchmarks()


if __name__ == "__main__":
    main()
