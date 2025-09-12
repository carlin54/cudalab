#include "../include/cudalab/library.cuh"

namespace {
    template<typename T, int BLOCK>
    __global__ void __add_vectors(const T *__restrict__ a,
                                  const T *__restrict__ b,
                                  T *__restrict__ out,
                                  unsigned n) {
        int i = blockIdx.x * BLOCK + threadIdx.x;
        int stride = BLOCK * gridDim.x;
        for (; i < n; i += stride) {
            out[i] = a[i] + b[i];
        }
    }

    template<typename T, int BLOCK>
    __global__ void __multiply_vectors(const T *__restrict__ a,
                                  const T *__restrict__ b,
                                  T *__restrict__ out,
                                  unsigned n) {
        int i = blockIdx.x * BLOCK + threadIdx.x;
        int stride = BLOCK * gridDim.x;
        for (; i < n; i += stride) {
            out[i] = a[i] * b[i];
        }
    }

    template<typename T>
    __global__ void __min_vector(const T *input, size_t n, T *min_out);

    template<>
    __global__ void __min_vector<float>(const float *input,
                                        size_t n,
                                        float *min_out) {
        extern __shared__ float shared_min[];
        float local_min = CUDART_INF_F;
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;

        // Grid-stride loop to process input
        for (; idx < n; idx += stride) {
            local_min = fminf(local_min, input[idx]);
        }
        shared_min[threadIdx.x] = local_min;

        __syncthreads();

        // Reduction with bounds checking
        for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
            if (threadIdx.x < s) {  // Only threads in lower half participate
                shared_min[threadIdx.x] = fminf(shared_min[threadIdx.x], shared_min[threadIdx.x + s]);
            }
            __syncthreads();
        }

        // Write block result to global memory
        if (threadIdx.x == 0) {
            min_out[blockIdx.x] = shared_min[0];  // Each block writes to its own location
        }
    }

    template<typename T>
    __global__ void __max_vector(const T *input, size_t n, T *min_out);

    template<>
    __global__ void __max_vector<float>(const float *input,
                                        size_t n,
                                        float *max_out) {
        extern __shared__ float shared_max[];
        float local_max = -CUDART_INF_F;
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;

        for (; idx < n; idx += stride) {
            local_max = fmaxf(local_max, input[idx]);
        }
        shared_max[threadIdx.x] = local_max;

        __syncthreads();

        // Reduction with bounds checking
        for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
            if (threadIdx.x < s) {
                shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            max_out[blockIdx.x] = shared_max[0];
        }
    }
}

namespace cudalab {

    template<typename T>
    cudaError_t add_vectors(const T* a, const T* b, T* out, int n) {
        if (n <= 0) return cudaSuccess;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        __add_vectors<T, 256><<<blocks, threads>>>(a, b, out, n);
        return cudaGetLastError();
    }
    template cudaError_t add_vectors<float>(const float*, const float*, float*, int);
    template<typename T>
    cudaError_t multiply_vectors(const T* a, const T* b, T* out, int n) {
        if (n <= 0) return cudaSuccess;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        __add_vectors<T, 256><<<blocks, threads>>>(a, b, out, n);
        return cudaGetLastError();
    }
    template cudaError_t multiply_vectors<float>(const float*, const float*, float*, int);

    template<typename T>
    cudaError_t min_vector(const T* a, size_t n, T* out) {
        if (n <= 0) return cudaSuccess;
        unsigned int threads = 256;
        unsigned int blocks = 3;  // Use only one block
        size_t shared_mem_size = threads * sizeof(T);
        __min_vector<T><<<blocks, threads, shared_mem_size>>>(a, n, out);
        return cudaGetLastError();
    }
    template cudaError_t min_vector<float>(const float*, size_t, float*);

    template<typename T>
    cudaError_t max_vector(const T* a, size_t n, T* out) {
        // TODO: Update max and min to support variable block sizes
        if (n <= 0) return cudaSuccess;
        unsigned int threads = 256;
        unsigned int blocks = 1;
        size_t shared_mem_size = threads * sizeof(T);
        __max_vector<T><<<blocks, threads, shared_mem_size>>>(a, n, out);  // ← FIXED: Now calls __max_vector
        return cudaGetLastError();
    }
    template cudaError_t max_vector<float>(const float*, size_t, float*);

}
