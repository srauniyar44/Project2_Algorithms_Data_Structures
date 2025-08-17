"""
HPC Data Structure Optimization Demo: AoS vs SoA (NumPy)

Run:
  python hpc_aos_soa_demo.py
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs("output", exist_ok=True)
    rng = np.random.default_rng(42)
    N = 500_000
    x = rng.standard_normal(N, dtype=np.float64)
    y = rng.standard_normal(N, dtype=np.float64)
    z = rng.standard_normal(N, dtype=np.float64)
    m = rng.random(N, dtype=np.float64) * 10.0 + 0.1
    aos_data = list(zip(x.tolist(), y.tolist(), z.tolist(), m.tolist()))

    def bench_aos(scale=1.001, thresh=2.0):
        t0 = time.perf_counter()
        ke_sum = 0.0
        for (xi, yi, zi, mi) in aos_data:
            ke_sum += 0.5 * mi * (xi*xi + yi*yi + zi*zi)
        t1 = time.perf_counter()
        ke_time = t1 - t0

        t0 = time.perf_counter()
        scaled = [(xi*scale, yi*scale, zi*scale, mi) for (xi, yi, zi, mi) in aos_data]
        t1 = time.perf_counter()
        scale_time = t1 - t0

        t0 = time.perf_counter()
        count = 0
        mass_sum = 0.0
        for (xi, yi, zi, mi) in scaled:
            if abs(xi) + abs(yi) + abs(zi) > thresh:
                count += 1
                mass_sum += mi
        t1 = time.perf_counter()
        filter_time = t1 - t0
        return ke_time, scale_time, filter_time

    def bench_soa(scale=1.001, thresh=2.0):
        nonlocal x, y, z, m
        t0 = time.perf_counter()
        ke_sum = 0.5 * np.sum(m * (x*x + y*y + z*z))
        t1 = time.perf_counter()
        ke_time = t1 - t0

        t0 = time.perf_counter()
        x *= scale; y *= scale; z *= scale
        t1 = time.perf_counter()
        scale_time = t1 - t0

        t0 = time.perf_counter()
        mask = (np.abs(x) + np.abs(y) + np.abs(z)) > thresh
        count = int(np.sum(mask))
        mass_sum = float(np.sum(m[mask]))
        t1 = time.perf_counter()
        filter_time = t1 - t0
        return ke_time, scale_time, filter_time

    _ = bench_aos(); _ = bench_soa()
    a_ke, a_scale, a_filter = bench_aos()
    s_ke, s_scale, s_filter = bench_soa()

    results = pd.DataFrame({
        "Workload": ["KE sum", "Scale update", "Filter+aggregate"],
        "AoS_time_s": [a_ke, a_scale, a_filter],
        "SoA_time_s": [s_ke, s_scale, s_filter],
    })
    results["Speedup (AoS/SoA)"] = results["AoS_time_s"] / results["SoA_time_s"]
    print(results)

    # Plots
    plt.figure(figsize=(7,5))
    idx = np.arange(len(results))
    width = 0.35
    plt.bar(idx - width/2, results["AoS_time_s"], width, label="AoS")
    plt.bar(idx + width/2, results["SoA_time_s"], width, label="SoA")
    plt.xticks(idx, results["Workload"], rotation=0)
    plt.ylabel("Time (seconds)")
    plt.title("AoS vs SoA Benchmark Times")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/aos_vs_soa_times.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.bar(results["Workload"], results["Speedup (AoS/SoA)"])
    plt.ylabel("Speedup (AoS / SoA)")
    plt.title("Speedup from SoA (higher is better)")
    plt.tight_layout()
    plt.savefig("output/aos_soa_speedup.png", dpi=160)
    plt.close()

    results.to_csv("output/benchmark_results.csv", index=False)

if __name__ == "__main__":
    main()
