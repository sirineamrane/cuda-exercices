#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 1024
#define OUTPUT_SIZE 512
#define TILE 256

__global__ void dense_forward(
    const float* x,
    const float* W,
    float* y
) {
    __shared__ float x_tile[TILE]; // on partage x à tous lesneurones et on le charge une seule fois par block

    int out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= OUTPUT_SIZE) return;

    float sum = 0.0f;

    for (int t = 0; t < INPUT_SIZE; t += TILE) {

        // on charge une part de x dans shared memory
        if (t + threadIdx.x < INPUT_SIZE)
            x_tile[threadIdx.x] = x[t + threadIdx.x];
        else
            x_tile[threadIdx.x] = 0.0f;

        __syncthreads(); 

        // on calcule partiellement
        for (int i = 0; i < TILE; i++) {
            sum += x_tile[i] * W[(t + i) * OUTPUT_SIZE + out]; // c une variable locale = elle go dans registres (mémoire ultra rapide)
        }

        __syncthreads(); 
    }

    // calcul final 
    y[out] = sum; //chaque thread écrit sa propre case = 0 conflit (pas besoin d'atomic)
}

int main() {
    float *x, *W, *y;
    float *dx, *dW, *dy;

    x = (float*)malloc(INPUT_SIZE * sizeof(float));
    W = (float*)malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
    y = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < INPUT_SIZE; i++) x[i] = 1.0f;
    for (int i = 0; i < INPUT_SIZE * OUTPUT_SIZE; i++) W[i] = 0.01f;

    cudaMalloc(&dx, INPUT_SIZE * sizeof(float));
    cudaMalloc(&dW, INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&dy, OUTPUT_SIZE * sizeof(float));

    cudaMemcpy(dx, x, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W, INPUT_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dense_forward<<<(OUTPUT_SIZE + TILE - 1)/TILE, TILE>>>(dx, dW, dy);

    cudaMemcpy(y, dy, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("y[0] = %f\n", y[0]);

    cudaFree(dx); cudaFree(dW); cudaFree(dy);
    free(x); free(W); free(y);
}

