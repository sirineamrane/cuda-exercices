#include <cuda_runtime.h>
#include <stdio.h>
#define N 32
#define THREADS 32

__global__ void spill_demo(float* out){
    int tid = threadIdx.x;
    float a1,a2,a3,a4,a5,a6,a7,a8,a9,a10;
    float b1,b2,b3,b4,b5,b6,b7,b8,b9,b10;
    float c1,c2,c3,c4,c5,c6,c7,c8,c9,c10;

    float sum = 0.0f;
    for(int i=0;i<10;i++){
        sum += tid + i; // juste pour utiliser les variables locales
    }

    out[tid] = sum;
}

int main(){
    float* out;
    cudaMallocManaged(&out, THREADS*sizeof(float));
    spill_demo<<<1,THREADS>>>(out);
    cudaDeviceSynchronize();

    for(int i=0;i<THREADS;i++) printf("out[%d]=%f\n", i, out[i]);
    cudaFree(out);
}
