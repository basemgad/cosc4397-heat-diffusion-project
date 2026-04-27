#include <iostream>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello from CUDA kernel!\n");
}

int main() {
    std::cout << "Starting CUDA test..." << std::endl;

    helloKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "CUDA test finished." << std::endl;
    return 0;
}
