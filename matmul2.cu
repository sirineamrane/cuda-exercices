#include <stdio.h>
#include <cuda_runtime.h>

#define N 32       // taille matrice NxN
#define TILE 16    // tile 16x16 → 256 threads par bloc

// kernel GPU multiplication de matrices avec tiling
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {

    // mémoire partagée pour une sous-matrice (tile)
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    // indice global calculé a partir de bloc & thread
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {

        // indices locaux dans la matrice
        int tiledCol = t * TILE + threadIdx.x;
        int tiledRow = t * TILE + threadIdx.y;

        // chargement sécurisé en mémoire partagée
        A_tile[threadIdx.y][threadIdx.x] =
            (row < n && tiledCol < n) ? A[row * n + tiledCol] : 0.0f;

        B_tile[threadIdx.y][threadIdx.x] =
            (tiledRow < n && col < n) ? B[tiledRow * n + col] : 0.0f;

        __syncthreads();  // on attend le chargement complet

        // produit partiel sur le tile
        for (int k = 0; k < TILE; k++) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();  // avant de passer au tile suivant
    }

    // écriture (si dans la matrice)
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    float h_A[N*N], h_B[N*N], h_C[N*N];

    // on remplit avec des valeurs simples
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    // configuration des blocs et de la grille
    dim3 dimBlock(TILE, TILE); 
    dim3 dimGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // lancement du kernel GPU
    matmul_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // on récupère  lerésultat
    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // affichage de la matrice résultat (C)
    printf("Matrice C :\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%5.1f ", h_C[i*N + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
}
