#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__global__ void jacobiKernel(const float* current, float* next, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (row >= n - 1 || col >= n - 1) {
        return;
    }

    int idx = row * n + col;

    next[idx] = 0.25f * (
        current[idx - n] +
        current[idx + n] +
        current[idx - 1] +
        current[idx + 1]
    );
}

void initializeGrid(std::vector<float>& grid, int n) {
    std::fill(grid.begin(), grid.end(), 0.0f);

    // Hot top boundary.
    for (int col = 0; col < n; ++col) {
        grid[col] = 100.0f;
    }

    // Cold left and right boundaries.
    for (int row = 0; row < n; ++row) {
        grid[row * n] = 0.0f;
        grid[row * n + (n - 1)] = 0.0f;
    }
}

std::vector<float> runCpuJacobi(const std::vector<float>& initial, int n, int iterations) {
    std::vector<float> current = initial;
    std::vector<float> next = initial;

    for (int iter = 0; iter < iterations; ++iter) {
        for (int row = 1; row < n - 1; ++row) {
            for (int col = 1; col < n - 1; ++col) {
                int idx = row * n + col;

                next[idx] = 0.25f * (
                    current[idx - n] +
                    current[idx + n] +
                    current[idx - 1] +
                    current[idx + 1]
                );
            }
        }

        std::swap(current, next);
    }

    return current;
}

std::vector<float> runGpuJacobi(
    const std::vector<float>& initial,
    int n,
    int iterations,
    float& kernelMs,
    double& totalMs
) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    auto totalStart = std::chrono::high_resolution_clock::now();

    float* dCurrent = nullptr;
    float* dNext = nullptr;

    CHECK_CUDA(cudaMalloc(&dCurrent, bytes));
    CHECK_CUDA(cudaMalloc(&dNext, bytes));

    CHECK_CUDA(cudaMemcpy(dCurrent, initial.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dNext, initial.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (n - 2 + block.x - 1) / block.x,
        (n - 2 + block.y - 1) / block.y
    );

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    CHECK_CUDA(cudaEventRecord(startEvent));

    for (int iter = 0; iter < iterations; ++iter) {
        jacobiKernel<<<grid, block>>>(dCurrent, dNext, n);
        CHECK_CUDA(cudaGetLastError());
        std::swap(dCurrent, dNext);
    }

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, startEvent, stopEvent));

    std::vector<float> output(static_cast<size_t>(n) * n);
    CHECK_CUDA(cudaMemcpy(output.data(), dCurrent, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    CHECK_CUDA(cudaFree(dCurrent));
    CHECK_CUDA(cudaFree(dNext));

    auto totalEnd = std::chrono::high_resolution_clock::now();
    totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return output;
}

double maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    double maxError = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        maxError = std::max(maxError, diff);
    }

    return maxError;
}

int main(int argc, char** argv) {
    int n = 512;
    int iterations = 500;

    if (argc >= 2) {
        n = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        iterations = std::atoi(argv[2]);
    }

    if (n < 3) {
        std::cerr << "Grid size must be at least 3." << std::endl;
        return 1;
    }

    std::vector<float> initial(static_cast<size_t>(n) * n);
    initializeGrid(initial, n);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuOutput = runCpuJacobi(initial, n, iterations);
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    float gpuKernelMs = 0.0f;
    double gpuTotalMs = 0.0;

    std::vector<float> gpuOutput = runGpuJacobi(
        initial,
        n,
        iterations,
        gpuKernelMs,
        gpuTotalMs
    );

    double error = maxAbsError(cpuOutput, gpuOutput);
    bool passed = error < 1e-3;

    double updates = static_cast<double>(n - 2) * static_cast<double>(n - 2) * iterations;
    double gpuSeconds = gpuKernelMs / 1000.0;
    double estimatedGflops = (updates * 4.0) / (gpuSeconds * 1e9);
    double estimatedBandwidth = (updates * 20.0) / (gpuSeconds * 1e9);

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Grid size: " << n << " x " << n << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "CPU time: " << cpuMs << " ms" << std::endl;
    std::cout << "GPU kernel time: " << gpuKernelMs << " ms" << std::endl;
    std::cout << "GPU total time: " << gpuTotalMs << " ms" << std::endl;
    std::cout << "Max absolute error: " << error << std::endl;
    std::cout << "Validation: " << (passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "GPU speedup using kernel time: " << cpuMs / gpuKernelMs << "x" << std::endl;
    std::cout << "GPU speedup using total time: " << cpuMs / gpuTotalMs << "x" << std::endl;
    std::cout << "Estimated GPU GFLOP/s: " << estimatedGflops << std::endl;
    std::cout << "Estimated GPU bandwidth: " << estimatedBandwidth << " GB/s" << std::endl;

    std::cout << "CSV,"
              << n << ","
              << iterations << ","
              << cpuMs << ","
              << gpuKernelMs << ","
              << gpuTotalMs << ","
              << error << ","
              << cpuMs / gpuKernelMs << ","
              << cpuMs / gpuTotalMs << ","
              << estimatedGflops << ","
              << estimatedBandwidth << ","
              << (passed ? "PASS" : "FAIL")
              << std::endl;

    return passed ? 0 : 1;
}
