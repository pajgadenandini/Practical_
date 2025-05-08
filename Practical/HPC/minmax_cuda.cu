#include <iostream>
#include <cuda.h>
#include <limits.h>

__global__ void reduce_sum(int *input, int *output, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    temp[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            temp[tid] += temp[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = temp[0];
}

__global__ void reduce_min(int *input, int *output, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    temp[tid] = (i < n) ? input[i] : INT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            temp[tid] = min(temp[tid], temp[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = temp[0];
}

__global__ void reduce_max(int *input, int *output, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    temp[tid] = (i < n) ? input[i] : INT_MIN;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            temp[tid] = max(temp[tid], temp[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = temp[0];
}

int reduce_final(int *d_input, int n, void (*kernel)(int*, int*, int), int identity, const char* label, const char* op) {
    int *d_output;
    int block_size = 128;
    int grid_size = (n + block_size - 1) / block_size;

    cudaMalloc(&d_output, grid_size * sizeof(int));
    kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, n);

    int *h_output = new int[grid_size];
    cudaMemcpy(h_output, d_output, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    int result = identity;
    for (int i = 0; i < grid_size; i++) {
        if (kernel == reduce_sum)
            result += h_output[i];
        else if (kernel == reduce_min)
            result = std::min(result, h_output[i]);
        else if (kernel == reduce_max)
            result = std::max(result, h_output[i]);
    }

    std::cout << label << ": " << result << std::endl;
    delete[] h_output;
    cudaFree(d_output);
    return result;
}

int main() {
    int n;
    std::cout << "Enter number of elements: ";
    std::cin >> n;

    int *h_input = new int[n];
    std::cout << "Enter " << n << " integers:\n";
    for (int i = 0; i < n; i++)
        std::cin >> h_input[i];

    int *d_input;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int sum = reduce_final(d_input, n, reduce_sum, 0, "Sum", "sum");
    reduce_final(d_input, n, reduce_min, INT_MAX, "Minimum", "min");
    reduce_final(d_input, n, reduce_max, INT_MIN, "Maximum", "max");

    double avg = static_cast<double>(sum) / n;
    std::cout << "Average: " << avg << std::endl;

    cudaFree(d_input);
    delete[] h_input;

    return 0;
}












































//save the code->nano reduction.cu
//compile -> nvcc reduction.cu -o reduction
//run -> ./reduction
//ex -> Enter number of elements: 5
Enter 5 integers:
4 9 1 7 3
