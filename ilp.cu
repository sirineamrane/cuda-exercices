#include <cuda_runtime.h>
#include <iostream>

__global__ void sum_kernel_ilp(const float* __restrict__ a, const float* __restrict__ b, float* c, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // 4 éléments par thread
    if(idx + 3 < N) {
        // on loaad 4 éléments à l’avance (ILP)
        float a0 = a[idx];
        float b0 = b[idx];
        float a1 = a[idx+1];
        float b1 = b[idx+1];
        float a2 = a[idx+2];
        float b2 = b[idx+2];
        float a3 = a[idx+3];
        float b3 = b[idx+3];

        // on compute pendant que d'autres loads se font en pipeline
        c[idx]   = a0 + b0;
        c[idx+1] = a1 + b1;
        c[idx+2] = a2 + b2;
        c[idx+3] = a3 + b3;
    }
}

int main() {
    const int N = 1 << 20;
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for(int i=0;i<N;i++){ h_a[i]=i; h_b[i]=2*i; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    cudaMemcpy(d_a,h_a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*sizeof(float),cudaMemcpyHostToDevice);

    int block = 256;
    int grid = ((N+3)/4 + block - 1)/block;
    sum_kernel_ilp<<<grid, block>>>(d_a,d_b,d_c,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c,d_c,N*sizeof(float),cudaMemcpyDeviceToHost);

    std::cout << "c[0] = " << h_c[0] << " c[N-1] = " << h_c[N-1] << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}
