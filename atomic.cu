#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_atomic(const float* input, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, input[idx]);
    }
}

int main() {
    const int N = 1024;
    float h_input[N], h_result = 0.0f;
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_result;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    reduce_atomic<<<(N+255)/256, 256>>>(d_input, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum = %.1f\n", h_result);

    cudaFree(d_input);
    cudaFree(d_result);
}
