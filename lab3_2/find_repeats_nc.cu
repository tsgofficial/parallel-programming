#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t _e = (call);                                 \
        if (_e != cudaSuccess)                                   \
        {                                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ void upsweep_kernel(int *d, int stride, int active)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active)
        return; /* блок хэмжээний дугаарлалтын хилийн шалгалт */

    int right = (idx + 1) * (stride * 2) - 1;
    int left = right - stride;
    d[right] += d[left];
}

__global__ void downsweep_kernel(int *d, int stride, int active)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active)
        return;

    int right = (idx + 1) * (stride * 2) - 1;
    int left = right - stride;

    int tmp = d[left];
    d[left] = d[right];
    d[right] += tmp;
}

void exclusive_scan(int *d_data, int n)
{
    int rounded = 1;
    while (rounded < n)
        rounded <<= 1;

    int *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, rounded * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_buf, d_data, n * sizeof(int), cudaMemcpyDeviceToDevice));
    if (rounded > n)
        CUDA_CHECK(cudaMemset(d_buf + n, 0, (rounded - n) * sizeof(int)));

    for (int stride = 1; stride < rounded; stride <<= 1)
    {
        int active = rounded / (stride * 2); /* энэ түвшний идэвхтэй thread */
        int blocks = (active + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_buf, stride, active);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_buf + rounded - 1, 0, sizeof(int)));

    for (int stride = rounded >> 1; stride >= 1; stride >>= 1)
    {
        int active = rounded / (stride * 2);
        int blocks = (active + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_buf, stride, active);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_data, d_buf, n * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_buf));
}

__global__ void mark_kernel(const int *input, int *flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1)
        return;
    flags[i] = (input[i] == input[i + 1]) ? 1 : 0;
}

__global__ void gather_kernel(const int *flags, const int *scan,
                              int *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1)
        return;
    if (flags[i])
        output[scan[i]] = i;
}

int find_repeats(const int *d_input, int n, int **d_output)
{
    int blocks = (n - 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int *d_flags;
    CUDA_CHECK(cudaMalloc(&d_flags, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flags, 0, n * sizeof(int))); /* flags[n-1] = 0 */
    mark_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_flags, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_scan;
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_scan, d_flags, n * sizeof(int), cudaMemcpyDeviceToDevice));
    exclusive_scan(d_scan, n);

    int last_scan, last_flag;
    CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_flag, d_flags + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int count = last_scan + last_flag;

    CUDA_CHECK(cudaMalloc(d_output, (count > 0 ? count : 1) * sizeof(int)));
    if (count > 0)
    {
        gather_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_flags, d_scan, *d_output, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_scan));
    return count;
}

int main(void)
{
    printf("══════════════════════════════════════════\n");
    printf(" А. Exclusive Prefix Scan\n");
    printf("══════════════════════════════════════════\n");
    {
        int h[] = {1, 4, 6, 8, 2};
        int n = 5;
        int *d;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice));

        exclusive_scan(d, n);

        int res[5];
        CUDA_CHECK(cudaMemcpy(res, d, n * sizeof(int), cudaMemcpyDeviceToHost));

        printf("  Input    : {1, 4, 6, 8, 2}\n");
        printf("  Output   : {");
        for (int i = 0; i < n; i++)
            printf(i ? ", %d" : "%d", res[i]);
        printf("}\n");
        printf("  Expected : {0, 1, 5, 11, 19}\n\n");
        CUDA_CHECK(cudaFree(d));
    }

    printf("══════════════════════════════════════════\n");
    printf(" Б. Find Repeats\n");
    printf("══════════════════════════════════════════\n");
    {
        int h[] = {1, 2, 2, 1, 1, 1, 3, 5, 3, 3};
        int n = 10;

        int *d_in;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_in, h, n * sizeof(int), cudaMemcpyHostToDevice));

        int *d_out = nullptr;
        int count = find_repeats(d_in, n, &d_out);

        printf("  Input    : {1,2,2,1,1,1,3,5,3,3}\n");
        printf("  Count    : %d\n", count);

        int *h_out = (int *)malloc(count * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_out, d_out, count * sizeof(int), cudaMemcpyDeviceToHost));

        printf("  Output   : {");
        for (int i = 0; i < count; i++)
            printf(i ? ", %d" : "%d", h_out[i]);
        printf("}\n");
        printf("  Expected : {1, 3, 4, 8}\n");

        free(h_out);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    return 0;
}
