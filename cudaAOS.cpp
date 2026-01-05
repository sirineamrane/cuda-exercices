#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Définition d'un point 2D
struct Point {
    float x;
    float y;
};

// Kernel CUDA : ajoute 1 à x et y de chaque point
__global__ void addAOS(Point* points, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // calcul de l'indice global du thread
    if(idx < N) {
        points[idx].x += 1.0f;
        points[idx].y += 1.0f;
    }
}

int main() {
    const int N = 8;
    Point h_points[N];

    // initialisation des points côté CPU
    for(int i = 0; i < N; i++) {
        h_points[i].x = i;
        h_points[i].y = i*10;
    }

    // allocation memoire GPU
    Point* d_points;
    cudaMalloc(&d_points, N * sizeof(Point));

    // copie des donnees vers le GPU
    cudaMemcpy(d_points, h_points, N * sizeof(Point), cudaMemcpyHostToDevice);

    // lancement du kernel 
    addAOS<<<1, N>>>(d_points, N);
    cudaDeviceSynchronize();

    // copie du résultat vers le CPU
    cudaMemcpy(h_points, d_points, N * sizeof(Point), cudaMemcpyDeviceToHost);

    // affichage
    for(int i = 0; i < N; i++) {
        cout << "Point " << i << ": x=" << h_points[i].x << ", y=" << h_points[i].y << endl;
    }

    // libération mémoire GPU
    cudaFree(d_points);
    return 0;
}
