% % writefile find_repeats.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

        // --- KERNELS FOR EXCLUSIVE SCAN ---

        __global__ void up_sweep_kernel(int *data, int n, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1 << (step + 1);
    int idx = (i + 1) * offset - 1;

    if (idx < n)
    {
        data[idx] += data[idx - (1 << step)];
    }
}

__global__ void down_sweep_kernel(int *data, int n, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1 << (step + 1);
    int idx = (i + 1) * offset - 1;

    if (idx < n)
    {
        int t = data[idx - (1 << step)];
        data[idx - (1 << step)] = data[idx];
        data[idx] += t;
    }
}

void exclusive_scan(int *d_data, int n)
{
    int m = 1;
    while (m < n)
        m *= 2;

    int threadsPerBlock = 256;

    for (int step = 0; (1 << (step + 1)) <= m; step++)
    {
        int active_threads = m / (1 << (step + 1));
        int blocks = (active_threads + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > 0)
        {
            up_sweep_kernel<<<blocks, threadsPerBlock>>>(d_data, m, step);
        }
        cudaDeviceSynchronize();
    }

    int zero = 0;
    cudaMemcpy(&d_data[m - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);

    for (int step = (int)log2((float)m) - 1; step >= 0; step--)
    {
        int active_threads = m / (1 << (step + 1));
        int blocks = (active_threads + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > 0)
        {
            down_sweep_kernel<<<blocks, threadsPerBlock>>>(d_data, m, step);
        }
        cudaDeviceSynchronize();
    }
}

// --- KERNELS FOR FIND REPEATS ---

__global__ void mark_repeats_kernel(const int *A, int *F, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1)
    {
        F[i] = (A[i] == A[i + 1]) ? 1 : 0;
    }
    else if (i == n - 1)
    {
        F[i] = 0;
    }
}

__global__ void scatter_results_kernel(const int *F, const int *S, int *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && F[i] == 1)
    {
        output[S[i]] = i;
    }
}

int find_repeats(int *d_A, int *d_output, int n)
{
    int m = 1;
    while (m < n)
        m *= 2;

    int *d_F, *d_S;
    cudaMalloc(&d_F, m * sizeof(int));
    cudaMalloc(&d_S, m * sizeof(int));
    cudaMemset(d_F, 0, m * sizeof(int));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    mark_repeats_kernel<<<blocks, threads>>>(d_A, d_F, n);
    cudaDeviceSynchronize();

    cudaMemcpy(d_S, d_F, m * sizeof(int), cudaMemcpyDeviceToDevice);
    exclusive_scan(d_S, m);

    scatter_results_kernel<<<blocks, threads>>>(d_F, d_S, d_output, n);
    cudaDeviceSynchronize();

    int last_f, last_s;
    cudaMemcpy(&last_f, &d_F[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_s, &d_S[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

    int count = last_f + last_s;

    cudaFree(d_F);
    cudaFree(d_S);
    return count;
}

void printArray(const std::string &label, const std::vector<int> &arr, int n = -1)
{
    if (n == -1)
        n = arr.size();
    std::cout << label << ": { ";
    for (int i = 0; i < n; i++)
    {
        std::cout << arr[i] << (i == n - 1 ? "" : ", ");
    }
    std::cout << " }" << std::endl;
}

int main()
{
    std::vector<int> h_A = {1, 2, 2, 1, 1, 1, 3, 5, 3, 3};
    int n = h_A.size();

    std::cout << "--- CUDA Find Repeats Test ---" << std::endl;
    printArray("Input A", h_A);

    int *d_A, *d_output;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_A, h_A.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int result_count = find_repeats(d_A, d_output, n);

    std::vector<int> h_output(result_count);
    cudaMemcpy(h_output.data(), d_output, result_count * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Detected repeats: " << result_count << std::endl;
    printArray("Output Indices", h_output);

    cudaFree(d_A);
    cudaFree(d_output);

    return 0;
}