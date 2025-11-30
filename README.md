## How to run?

make run

## Results

Machine info: Google Clound, C4D, 4 vCPUs (4 physical cores), 32 GB, AMD Turin


```bash
Benchmark: Array Size = 1048576 (u64)
Metrics: CPU Cycles (RDTSC)
-------------------------------------------------------------------------
Scalar (No Unroll)        | Avg Cycles:    1009690 | Min Cycles:     879363 | CPE: 0.9629
Scalar (Unroll 8)         | Avg Cycles:     863740 | Min Cycles:     733617 | CPE: 0.8237
AVX2 SIMD                 | Avg Cycles:     716090 | Min Cycles:     618759 | CPE: 0.6829
AVX2 SIMD (Unroll 8)      | Avg Cycles:     650483 | Min Cycles:     591786 | CPE: 0.6203
AVX2 SIMD (Unroll 4)      | Avg Cycles:     654454 | Min Cycles:     591570 | CPE: 0.6241

Benchmark: Array Size = 1048576 (u64)
Metrics: CPU Cycles (RDTSC)
-------------------------------------------------------------------------
Scalar (No Unroll)        | Avg Cycles:    1008841 | Min Cycles:     911061 | CPE: 0.9621
Scalar (Unroll 8)         | Avg Cycles:     837662 | Min Cycles:     707319 | CPE: 0.7989
AVX2 SIMD                 | Avg Cycles:     710542 | Min Cycles:     628236 | CPE: 0.6776
AVX2 SIMD (Unroll 8)      | Avg Cycles:     630058 | Min Cycles:     590868 | CPE: 0.6009
AVX2 SIMD (Unroll 4)      | Avg Cycles:     632950 | Min Cycles:     589005 | CPE: 0.6036
```