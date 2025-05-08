#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *result, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float *a, float *b, float *c, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < p) {
        float sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += a[row * m + k] * b[k * p + col];
        }
        c[row * p + col] = sum;
    }
}

int main() {
    int n, m, p;

    // ---------------- VECTOR ADDITION ----------------
    printf("Enter size of vectors: ");
    scanf("%d", &n);

    // Allocate host memory
    float *h_vecA = (float *)malloc(n * sizeof(float));
    float *h_vecB = (float *)malloc(n * sizeof(float));
    float *h_vecResult = (float *)malloc(n * sizeof(float));

    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_vecA[i] = i + 1;
        h_vecB[i] = (i + 1) * 2;
    }

    // Allocate device memory
    float *d_vecA, *d_vecB, *d_vecResult;
    cudaMalloc((void **)&d_vecA, n * sizeof(float));
    cudaMalloc((void **)&d_vecB, n * sizeof(float));
    cudaMalloc((void **)&d_vecResult, n * sizeof(float));

    // Copy vectors to device
    cudaMemcpy(d_vecA, h_vecA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecB, h_vecB, n * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_vecA, d_vecB, d_vecResult, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_vecResult, d_vecResult, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nVector Addition Result:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_vecResult[i]);
    }
    printf("\n");

    // Cleanup vector memory
    cudaFree(d_vecA);
    cudaFree(d_vecB);
    cudaFree(d_vecResult);
    free(h_vecA);
    free(h_vecB);
    free(h_vecResult);

    // ---------------- MATRIX MULTIPLICATION ----------------
    printf("\nEnter rows and columns of Matrix A (n m): ");
    scanf("%d %d", &n, &m);
    printf("Enter columns of Matrix B (p): ");
    scanf("%d", &p);

    size_t sizeA = n * m * sizeof(float);
    size_t sizeB = m * p * sizeof(float);
    size_t sizeC = n * p * sizeof(float);

    float *h_matA = (float *)malloc(sizeA);
    float *h_matB = (float *)malloc(sizeB);
    float *h_matC = (float *)malloc(sizeC);

    // Initialize matrices
    for (int i = 0; i < n * m; i++) h_matA[i] = 1.0;
    for (int i = 0; i < m * p; i++) h_matB[i] = 2.0;

    float *d_matA, *d_matB, *d_matC;
    cudaMalloc((void **)&d_matA, sizeA);
    cudaMalloc((void **)&d_matB, sizeB);
    cudaMalloc((void **)&d_matC, sizeC);

    cudaMemcpy(d_matA, h_matA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, sizeB, cudaMemcpyHostToDevice);

    // Grid and block configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + 15) / 16, (n + 15) / 16);
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_matA, d_matB, d_matC, n, m, p);
    cudaDeviceSynchronize();

    cudaMemcpy(h_matC, d_matC, sizeC, cudaMemcpyDeviceToHost);

    printf("\nMatrix Multiplication Result (A x B):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.1f ", h_matC[i * p + j]);
        }
        printf("\n");
    }

    // Cleanup matrix memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
    free(h_matA);
    free(h_matB);
    free(h_matC);

    return 0;
}










































/*nvcc vector_matrix_combined.cu -o vector_matrix_combined
./vector_matrix_combined



Advantages of CUDA:

Runs operations in parallel on GPU, making it much faster than CPU for large data.

Reduces time complexity for vector/matrix tasks using multiple threads.

How to launch a kernel:

Use the syntax: kernel<<<numBlocks, threadsPerBlock>>>(args)

For example: vectorAdd<<<gridSize, blockSize>>>(a, b, result, n);

How to optimize performance:

Use appropriate block and grid sizes.

Ensure memory coalescing and shared memory (for advanced optimization).

Avoid unnecessary memory transfers between host and device.

Enter size of vectors: 5
Enter rows and columns of Matrix A (n m): 2 3
Enter columns of Matrix B (p): 2
Vector Addition:
[1+2, 2+4, 3+6, 4+8, 5+10] → [3, 6, 9, 12, 15]
Matrix Multiplication:
Matrix A (2x3) × Matrix B (3x2) = Matrix C (2x2)

C[0][0] = 1×2 + 1×2 + 1×2 = 6
C[0][1] = 1×2 + 1×2 + 1×2 = 6
C[1][0] = 1×2 + 1×2 + 1×2 = 6
C[1][1] = 1×2 + 1×2 + 1×2 = 6


Matrix A (2 rows × 3 columns):

1   2   3
4   5   6

Matrix B (3 rows × 2 columns):

7   8
9   10
11  12




*/
