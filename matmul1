#include <stdio.h>
#include <cuda_runtime.h>

#define N 4          // taille matrice NxN
#define TILE_SIZE 2  // taille du tile (block shared memory)

__global__ void matMulTiled(float *A, float *B, float *C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = threadIdx.y + blockIdx.y * TILE_SIZE;
    int col = threadIdx.x + blockIdx.x * TILE_SIZE;

    float sum = 0.0f;

    // parcours des tiles
    for (int t = 0; t < (n + TILE_SIZE - 1)/TILE_SIZE; t++) {
        // pour charger un tile dans shared memory
        if (row < n && t*TILE_SIZE + threadIdx.x < n)
            tileA[threadIdx.y][threadIdx.x] = A[row*n + t*TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t*TILE_SIZE + threadIdx.y < n && col < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // attendre que tous les threads aient chargé les tiles

        // calcul du produit partiel
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    // écrire le résultat
    if (row < n && col < n)
        C[row*n + col] = sum;
}

int main() {
    float h_A[N*N] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float h_B[N*N] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    float h_C[N*N];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE -1)/TILE_SIZE, (N + TILE_SIZE -1)/TILE_SIZE);

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // afficher le résultat
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%6.1f ", h_C[i*N + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
