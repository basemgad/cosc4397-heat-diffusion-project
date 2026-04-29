#!/usr/bin/env bash
set -e

ITERATIONS=${1:-500}

make clean
make

mkdir -p results

echo "variant,n,iterations,block_x,block_y,cpu_ms,gpu_kernel_ms,gpu_total_ms,max_error,validation,speedup_kernel,speedup_total,gpu_gflops,gpu_bandwidth_gbs" > results/benchmark.csv

for N in 256 512 1024 2048
do
    echo "Running N=$N iterations=$ITERATIONS"
    ./build/heat "$N" "$ITERATIONS" all "" csv | tee "results/run_${N}.txt"
    grep '^CSV,' "results/run_${N}.txt" | sed 's/^CSV,//' >> results/benchmark.csv
done

echo "Saved results to results/benchmark.csv"