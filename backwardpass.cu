#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 1024
#define OUTPUT_SIZE 512
#define TILE 256

__global__ void dense_backward(
    const float* x,
    const float* dY, // gradient de sortie
    float* dW        // gradient poids
) {
    __shared__ float x_tile[TILE];

    int out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= OUTPUT_SIZE) return;

    for (int t = 0; t < INPUT_SIZE; t += TILE) {
        if (t + threadIdx.x < INPUT_SIZE)
            x_tile[threadIdx.x] = x[t + threadIdx.x];
        else
            x_tile[threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++)
            dW[(t + i) * OUTPUT_SIZE + out] = x_tile[i] * dY[out];

        __syncthreads();
    }
}
