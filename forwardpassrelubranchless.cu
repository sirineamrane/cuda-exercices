#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024
#define TILE 32

__global__ void dense_forward_branchless(float* x, float* W, float* y, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;

    float sum = 0.0f;

    for(int i=0;i<n;i++){
        // mul + sum branchless
        sum = fmaf(x[i], W[i*n + tid], sum);
    }

    // reLU branchless
    y[tid] = fmaxf(0.0f, sum);
}

int main() {
    float *x, *W, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&W, N*N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // init
    for(int i=0;i<N;i++){
        x[i] = i*0.01f;
        for(int j=0;j<N;j++) W[i*N+j] = 0.01f*(i+j+1);
    }

    dense_forward_branchless<<<(N+TILE-1)/TILE,TILE>>>(x,W,y,N);
    cudaDeviceSynchronize();

    printf("y[0]=%f\n", y[0]);

    cudaFree(x); cudaFree(W); cudaFree(y);
}
