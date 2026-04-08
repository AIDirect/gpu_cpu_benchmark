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

def timed(fn, warmup=2, repeats=5):
    """Run *fn* with warmup iterations, then return the median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]


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
        info["gpu_vram_gb"] = round(props.total_mem / 1024**3, 2)
    return info


# ---------------------------------------------------------------------------
# CPU benchmarks
# ---------------------------------------------------------------------------

def cpu_matmul(n=4096):
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    def run():
        np.dot(a, b)

    sec = timed(run)
    return {
        "name": f"CPU matrix multiply ({n}x{n})",
        "time_s": round(sec, 4),
        "gflops": round(gflops_matmul(n, sec), 2),
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

    sec = timed(run, warmup=1, repeats=3)
    return {
        "name": f"CPU single-thread Monte Carlo pi ({n_samples:,} samples)",
        "time_s": round(sec, 4),
        "samples_per_sec": round(n_samples / sec, 0),
    }


def cpu_multi_core():
    """NumPy SVD on a large matrix — exercises multiple cores via BLAS."""
    n = 2048
    a = np.random.randn(n, n).astype(np.float32)

    def run():
        np.linalg.svd(a, full_matrices=False)

    sec = timed(run, warmup=1, repeats=3)
    return {
        "name": f"CPU SVD ({n}x{n}, multi-core BLAS)",
        "time_s": round(sec, 4),
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

    sec = timed(run)
    return {
        "name": f"GPU matrix multiply ({n}x{n})",
        "time_s": round(sec, 4),
        "gflops": round(gflops_matmul(n, sec), 2),
    }


def gpu_matmul_large(n=8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)

    def run():
        torch.mm(a, b)
        torch.cuda.synchronize()

    sec = timed(run)
    return {
        "name": f"GPU matrix multiply ({n}x{n})",
        "time_s": round(sec, 4),
        "gflops": round(gflops_matmul(n, sec), 2),
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

    sec = timed(run)
    return {
        "name": f"GPU Conv2d ({batch}x{channels}x{h}x{w}, k={kernel})",
        "time_s": round(sec, 4),
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

    h2d_sec = timed(h2d)
    h2d_gbps = size_mb / 1024 / h2d_sec

    # Device → Host
    device_tensor = host_tensor.to("cuda")
    torch.cuda.synchronize()

    def d2h():
        _ = device_tensor.to("cpu", non_blocking=False)
        torch.cuda.synchronize()

    d2h_sec = timed(d2h)
    d2h_gbps = size_mb / 1024 / d2h_sec

    return {
        "name": f"GPU memory transfer ({size_mb} MB)",
        "h2d_time_s": round(h2d_sec, 4),
        "h2d_gbps": round(h2d_gbps, 2),
        "d2h_time_s": round(d2h_sec, 4),
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
        summary = f"{r['time_s']}s"
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
            summary = f"{r['time_s']}s" if "time_s" in r else ""
            if "gflops" in r:
                summary += f"  ({r['gflops']} GFLOPS)"
            if "h2d_gbps" in r:
                summary = f"H2D {r['h2d_gbps']} GB/s  D2H {r['d2h_gbps']} GB/s"
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
    run_benchmarks()


if __name__ == "__main__":
    main()
