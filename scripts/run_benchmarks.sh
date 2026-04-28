#!/usr/bin/env bash
set -e

ITERATIONS=${1:-200}

make clean
make

mkdir -p results

echo "N,iterations,cpu_ms,gpu_kernel_ms,gpu_total_ms,max_error,speedup_kernel,speedup_total,gpu_gflops,gpu_bandwidth_gbs,validation" > results/benchmark.csv

for N in 256 512 1024 2048
do
    echo "========================================"
    echo "Running N=$N iterations=$ITERATIONS"
    echo "========================================"

    ./build/heat "$N" "$ITERATIONS" | tee "results/run_${N}.txt"

    grep '^CSV,' "results/run_${N}.txt" | sed 's/^CSV,//' >> results/benchmark.csv
done

echo "Benchmark results saved to results/benchmark.csv"
