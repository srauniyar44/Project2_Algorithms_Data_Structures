# HPC Data Structure Optimization: AoS vs SoA

This package demonstrates transforming Array-of-Structs (AoS) into Structure-of-Arrays (SoA)
and measuring the impact on runtime for common numeric workloads.

## Contents
- `hpc_aos_soa_demo.py` — runnable Python benchmark
- `benchmark_results.csv` — timing results
- `aos_vs_soa_times.png`, `aos_soa_speedup.png` — plots
- `HPC_Optimization_Report_AoS_vs_SoA.docx` — APA 7 report
- `HPC_Optimization_Presentation.pptx` — slides

## Run
```bash
pip install numpy pandas matplotlib
python hpc_aos_soa_demo.py
```
Results will be written to `./output`.

## Notes
- Increase N in the script if you want larger, clearer speedups (hardware permitting).
