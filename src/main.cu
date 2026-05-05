#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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

struct GpuResult {
    std::string name;
    std::vector<float> output;
    double kernelMs;
    double totalMs;
    int blockX;
    int blockY;
};

__global__ void jacobiGlobalKernel(const float* current, float* next, int n) {
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

__forceinline__ __device__ float readCell(
    const float* __restrict__ grid,
    int n,
    int row,
    int col
) {
    if (row >= 0 && row < n && col >= 0 && col < n) {
        return grid[row * n + col];
    }

    return 0.0f;
}

template <int BLOCK_X, int BLOCK_Y>
__global__ void jacobiSharedTiledKernel(
    const float* __restrict__ current,
    float* __restrict__ next,
    int n
) {
    __shared__ float tile[BLOCK_Y + 2][BLOCK_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * BLOCK_X + tx + 1;
    int row = blockIdx.y * BLOCK_Y + ty + 1;

    tile[ty + 1][tx + 1] = readCell(current, n, row, col);

    if (tx == 0) {
        tile[ty + 1][0] = readCell(current, n, row, col - 1);
    }

    if (tx == BLOCK_X - 1) {
        tile[ty + 1][BLOCK_X + 1] = readCell(current, n, row, col + 1);
    }

    if (ty == 0) {
        tile[0][tx + 1] = readCell(current, n, row - 1, col);
    }

    if (ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][tx + 1] = readCell(current, n, row + 1, col);
    }

    __syncthreads();

    if (row >= n - 1 || col >= n - 1) {
        return;
    }

    float top = tile[ty][tx + 1];
    float bottom = tile[ty + 2][tx + 1];
    float left = tile[ty + 1][tx];
    float right = tile[ty + 1][tx + 2];

    next[row * n + col] = 0.25f * (top + bottom + left + right);
}

__global__ void jacobiSharedKernel(const float* current, float* next, int n) {
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sharedWidth = blockDim.x + 2;
    int sharedHeight = blockDim.y + 2;

    int blockStartCol = blockIdx.x * blockDim.x;
    int blockStartRow = blockIdx.y * blockDim.y;

    for (int sy = ty; sy < sharedHeight; sy += blockDim.y) {
        for (int sx = tx; sx < sharedWidth; sx += blockDim.x) {
            int globalRow = blockStartRow + sy;
            int globalCol = blockStartCol + sx;

            tile[sy * sharedWidth + sx] = readCell(current, n, globalRow, globalCol);
        }
    }

    __syncthreads();

    int col = blockStartCol + tx + 1;
    int row = blockStartRow + ty + 1;

    if (row >= n - 1 || col >= n - 1) {
        return;
    }

    int sharedRow = ty + 1;
    int sharedCol = tx + 1;

    float top = tile[(sharedRow - 1) * sharedWidth + sharedCol];
    float bottom = tile[(sharedRow + 1) * sharedWidth + sharedCol];
    float left = tile[sharedRow * sharedWidth + (sharedCol - 1)];
    float right = tile[sharedRow * sharedWidth + (sharedCol + 1)];

    next[row * n + col] = 0.25f * (top + bottom + left + right);
}

void initializeGrid(std::vector<float>& grid, int n) {
    std::fill(grid.begin(), grid.end(), 0.0f);

    for (int col = 0; col < n; ++col) {
        grid[col] = 100.0f;
    }

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

GpuResult runGpuGlobal(
    const std::vector<float>& initial,
    int n,
    int iterations,
    int blockX,
    int blockY,
    const std::string& name
) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    auto totalStart = std::chrono::high_resolution_clock::now();

    float* dCurrent = nullptr;
    float* dNext = nullptr;

    CHECK_CUDA(cudaMalloc(&dCurrent, bytes));
    CHECK_CUDA(cudaMalloc(&dNext, bytes));

    CHECK_CUDA(cudaMemcpy(dCurrent, initial.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dNext, initial.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(blockX, blockY);
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
        jacobiGlobalKernel<<<grid, block>>>(dCurrent, dNext, n);
        CHECK_CUDA(cudaGetLastError());
        std::swap(dCurrent, dNext);
    }

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, startEvent, stopEvent));

    std::vector<float> output(static_cast<size_t>(n) * n);
    CHECK_CUDA(cudaMemcpy(output.data(), dCurrent, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    CHECK_CUDA(cudaFree(dCurrent));
    CHECK_CUDA(cudaFree(dNext));

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return {name, output, static_cast<double>(kernelMs), totalMs, blockX, blockY};
}

GpuResult runGpuShared(
    const std::vector<float>& initial,
    int n,
    int iterations,
    int blockX,
    int blockY,
    const std::string& name
) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    auto totalStart = std::chrono::high_resolution_clock::now();

    float* dCurrent = nullptr;
    float* dNext = nullptr;

    CHECK_CUDA(cudaMalloc(&dCurrent, bytes));
    CHECK_CUDA(cudaMalloc(&dNext, bytes));

    CHECK_CUDA(cudaMemcpy(dCurrent, initial.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dNext, initial.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(blockX, blockY);
    dim3 grid(
        (n - 2 + block.x - 1) / block.x,
        (n - 2 + block.y - 1) / block.y
    );

    size_t sharedBytes = static_cast<size_t>(blockX + 2) * (blockY + 2) * sizeof(float);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    CHECK_CUDA(cudaEventRecord(startEvent));

    for (int iter = 0; iter < iterations; ++iter) {
        if (blockX == 32 && blockY == 16) {
            jacobiSharedTiledKernel<32, 16><<<grid, block>>>(dCurrent, dNext, n);
        } else if (blockX == 16 && blockY == 16) {
            jacobiSharedTiledKernel<16, 16><<<grid, block>>>(dCurrent, dNext, n);
        } else if (blockX == 32 && blockY == 8) {
            jacobiSharedTiledKernel<32, 8><<<grid, block>>>(dCurrent, dNext, n);
        } else {
            jacobiSharedKernel<<<grid, block, sharedBytes>>>(dCurrent, dNext, n);
        }

        CHECK_CUDA(cudaGetLastError());
        std::swap(dCurrent, dNext);
    }

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, startEvent, stopEvent));

    std::vector<float> output(static_cast<size_t>(n) * n);
    CHECK_CUDA(cudaMemcpy(output.data(), dCurrent, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    CHECK_CUDA(cudaFree(dCurrent));
    CHECK_CUDA(cudaFree(dNext));

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return {name, output, static_cast<double>(kernelMs), totalMs, blockX, blockY};
}

double maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    double maxError = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        maxError = std::max(maxError, diff);
    }

    return maxError;
}

float clamp01(float x) {
    if (x < 0.0f) {
        return 0.0f;
    }

    if (x > 1.0f) {
        return 1.0f;
    }

    return x;
}

void writePPM(const std::vector<float>& grid, int n, const std::string& filename) {
    std::ofstream out(filename);

    if (!out) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    out << "P3\n";
    out << n << " " << n << "\n";
    out << 255 << "\n";

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float temp = grid[row * n + col];
            float t = clamp01(temp / 100.0f);

            t = std::pow(t, 0.35f);

            int r = static_cast<int>(255.0f * clamp01(1.5f * t - 0.3f));
            int g = static_cast<int>(255.0f * clamp01(1.5f - std::fabs(3.0f * t - 1.5f)));
            int b = static_cast<int>(255.0f * clamp01(1.0f - 1.5f * t));

            out << r << " " << g << " " << b << " ";
        }

        out << "\n";
    }
}

void printGpuResult(
    const GpuResult& result,
    double cpuMs,
    const std::vector<float>& cpuOutput,
    int n,
    int iterations
) {
    double error = maxAbsError(cpuOutput, result.output);
    bool passed = error < 1e-3;

    double updates = static_cast<double>(n - 2) * static_cast<double>(n - 2) * iterations;
    double kernelSeconds = result.kernelMs / 1000.0;

    double gflops = (updates * 4.0) / (kernelSeconds * 1e9);
    double bandwidth = (updates * 20.0) / (kernelSeconds * 1e9);

    double speedupKernel = cpuMs / result.kernelMs;
    double speedupTotal = cpuMs / result.totalMs;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Variant: " << result.name << std::endl;
    std::cout << "Block size: " << result.blockX << " x " << result.blockY << std::endl;
    std::cout << "CPU time: " << cpuMs << " ms" << std::endl;
    std::cout << "GPU kernel time: " << result.kernelMs << " ms" << std::endl;
    std::cout << "GPU total time: " << result.totalMs << " ms" << std::endl;
    std::cout << "Max absolute error: " << error << std::endl;
    std::cout << "Validation: " << (passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "GPU speedup using kernel time: " << speedupKernel << "x" << std::endl;
    std::cout << "GPU speedup using total time: " << speedupTotal << "x" << std::endl;
    std::cout << "Estimated GPU GFLOP/s: " << gflops << std::endl;
    std::cout << "Estimated GPU bandwidth: " << bandwidth << " GB/s" << std::endl;

    std::cout << "CSV,"
              << result.name << ","
              << n << ","
              << iterations << ","
              << result.blockX << ","
              << result.blockY << ","
              << cpuMs << ","
              << result.kernelMs << ","
              << result.totalMs << ","
              << error << ","
              << (passed ? "PASS" : "FAIL") << ","
              << speedupKernel << ","
              << speedupTotal << ","
              << gflops << ","
              << bandwidth
              << std::endl;
}
void printCsvLine(
    const GpuResult& result,
    double cpuMs,
    const std::vector<float>& cpuOutput,
    int n,
    int iterations
) {
    double error = maxAbsError(cpuOutput, result.output);
    bool passed = error < 1e-3;

    double updates = static_cast<double>(n - 2) * static_cast<double>(n - 2) * iterations;
    double kernelSeconds = result.kernelMs / 1000.0;
    double gflops = (updates * 4.0) / (kernelSeconds * 1e9);
    double bandwidth = (updates * 20.0) / (kernelSeconds * 1e9);

    std::cout << "CSV,"
              << result.name << ","
              << n << ","
              << iterations << ","
              << result.blockX << ","
              << result.blockY << ","
              << cpuMs << ","
              << result.kernelMs << ","
              << result.totalMs << ","
              << error << ","
              << (passed ? "PASS" : "FAIL") << ","
              << cpuMs / result.kernelMs << ","
              << cpuMs / result.totalMs << ","
              << gflops << ","
              << bandwidth
              << std::endl;
}

void runAllVariants(
    const std::vector<float>& initial,
    const std::vector<float>& cpuOutput,
    double cpuMs,
    int n,
    int iterations,
    const std::string& ppmFilename,
    bool csvOutput
) {
    std::vector<GpuResult> results;

    results.push_back(runGpuGlobal(initial, n, iterations, 32, 16, "global_32x16"));
    results.push_back(runGpuShared(initial, n, iterations, 32, 16, "shared_32x16"));

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Heat Diffusion Jacobi Stencil\n";
    std::cout << "Grid: " << n << " x " << n << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "CPU reference time: " << cpuMs << " ms\n\n";

    std::cout << std::left
              << std::setw(16) << "Variant"
              << std::setw(13) << "Kernel ms"
              << std::setw(13) << "Total ms"
              << std::setw(10) << "Speedup"
              << std::setw(11) << "Error"
              << "Result\n";

    std::cout << std::string(70, '-') << "\n";

    const GpuResult* best = &results[0];

    for (const GpuResult& result : results) {
        double error = maxAbsError(cpuOutput, result.output);
        bool passed = error < 1e-3;
        double speedup = cpuMs / result.kernelMs;

        std::cout << std::left
                  << std::setw(16) << result.name
                  << std::setw(13) << result.kernelMs
                  << std::setw(13) << result.totalMs
                  << std::setw(10) << speedup
                  << std::setw(11) << error
                  << (passed ? "PASS" : "FAIL")
                  << "\n";

        if (result.kernelMs < best->kernelMs) {
            best = &result;
        }

        if (csvOutput) {
            printCsvLine(result, cpuMs, cpuOutput, n, iterations);
        }
    }

    std::cout << "\nBest variant: " << best->name << "\n";
    std::cout << "Best kernel time: " << best->kernelMs << " ms\n";

    if (!ppmFilename.empty()) {
        writePPM(best->output, n, ppmFilename);
        std::cout << "Visualization: " << ppmFilename << "\n";
    }
}

int main(int argc, char** argv) {
    int n = 256;
    int iterations = 2000;
    std::string mode = "all";
    std::string ppmFilename = "results/heat.ppm";
    bool csvOutput = false;
    if (argc >= 2) {
        n = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        iterations = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        mode = argv[3];
    }

    if (argc >= 5) {
        ppmFilename = argv[4];
    }

    if (argc >= 6) {
        csvOutput = std::string(argv[5]) == "csv";
    }

    if (n < 3) {
        std::cerr << "Grid size must be at least 3." << std::endl;
        return 1;
    }

    if (iterations < 0) {
        std::cerr << "Iterations cannot be negative." << std::endl;
        return 1;
    }

    CHECK_CUDA(cudaFree(0));

    std::vector<float> initial(static_cast<size_t>(n) * n);
    initializeGrid(initial, n);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuOutput = runCpuJacobi(initial, n, iterations);
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    if (mode == "cpu") {
        std::cout << "Grid size: " << n << " x " << n << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "CPU time: " << cpuMs << " ms" << std::endl;

        if (!ppmFilename.empty()) {
            writePPM(cpuOutput, n, ppmFilename);
            std::cout << "Wrote visualization to: " << ppmFilename << std::endl;
        }

        return 0;
    }

    if (mode == "all") {
        runAllVariants(initial, cpuOutput, cpuMs, n, iterations, ppmFilename, csvOutput);
        return 0;
    }

    GpuResult result;

    if (mode == "global16") {
        result = runGpuGlobal(initial, n, iterations, 16, 16, "global_16x16");
    } else if (mode == "global32x8") {
        result = runGpuGlobal(initial, n, iterations, 32, 8, "global_32x8");
    } else if (mode == "global32x16") {
        result = runGpuGlobal(initial, n, iterations, 32, 16, "global_32x16");
    } else if (mode == "shared16") {
        result = runGpuShared(initial, n, iterations, 16, 16, "shared_16x16");
    } else if (mode == "shared32x8") {
        result = runGpuShared(initial, n, iterations, 32, 8, "shared_32x8");
    } else if (mode == "shared32x16") {
        result = runGpuShared(initial, n, iterations, 32, 16, "shared_32x16");
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        std::cerr << "Valid modes: all, cpu, global16, global32x8, global32x16, shared16, shared32x8, shared32x16" << std::endl;
        return 1;
    }

    std::cout << "Grid size: " << n << " x " << n << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::endl;

    printGpuResult(result, cpuMs, cpuOutput, n, iterations);

    if (!ppmFilename.empty()) {
        writePPM(result.output, n, ppmFilename);
        std::cout << "Wrote visualization to: " << ppmFilename << std::endl;
    }

    return 0;
}

