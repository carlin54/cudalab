#include "../include/cudalab/library.cuh"

namespace {
    template<typename T, int BLOCK>
    __global__ void _add_vectors(const T *__restrict__ a,
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
    __global__ void _multiply_vectors(const T *__restrict__ a,
                                  const T *__restrict__ b,
                                  size_t n,
                                  T *__restrict__ out) {
        int i = blockIdx.x * BLOCK + threadIdx.x;
        int stride = BLOCK * gridDim.x;
        for (; i < n; i += stride) {
            out[i] = a[i] * b[i];
        }
    }

    template<typename T>
    __global__ void _min_vector(const T *input, size_t n, T *min_out);

    template<>
    __global__ void _min_vector<float>(const float *input,
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
    __global__ void _max_vector(const T *input, size_t n, T *min_out);

    template<>
    __global__ void _max_vector<float>(const float *input,
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

    template<typename T>
    __global__ void _sum_vector(const T* input, size_t n, T* sum) {
        extern __shared__ T shared_sum[];
        T local_sum = 0;
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;

        for (; idx < n; idx += stride) {
            local_sum += input[idx];
        }

        shared_sum[threadIdx.x] = local_sum;
        __syncthreads();

        for (unsigned s = blockDim.x / 2; s > 0; s /= 2) {
            if (threadIdx.x < s) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) sum[blockIdx.x] = shared_sum[0];
    }


    template<typename T>
    __global__ void _dot_product(const T* a, const T* b, size_t n, T* product) {
        extern __shared__ T shared_product[];
        T local_sum = 0;
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;

        for (; idx < n; idx += stride) {
            local_sum += a[idx] * b[idx];
        }
        shared_product[threadIdx.x] = local_sum;
        __syncthreads();

        for (unsigned s = blockDim.x / 2; s > 0; s /= 2) {
            if (threadIdx.x < s) {
                shared_product[threadIdx.x] += shared_product[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            product[blockIdx.x] = shared_product[0];
        }
    }
}

namespace cudalab {

    template<typename T>
    cudaError_t add_vectors(const T* a, const T* b, T* out, int n) {
        if (n <= 0) return cudaSuccess;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        _add_vectors<T, 256><<<blocks, threads>>>(a, b, out, n);
        return cudaGetLastError();
    }
    template cudaError_t add_vectors<float>(const float*, const float*, float*, int);

    template<typename T>
    cudaError_t min_vector(const T* a, size_t n, T* out) {
        if (n <= 0) return cudaSuccess;
        unsigned int threads = 256;
        unsigned int blocks = 1;
        size_t shared_mem_size = threads * sizeof(T);
        _min_vector<T><<<blocks, threads, shared_mem_size>>>(a, n, out);
        return cudaGetLastError();
    }
    template cudaError_t min_vector<float>(const float*, size_t, float*);

    template<typename T>
    cudaError_t max_vector(const T* a, size_t n, T* out) {
        if (n <= 0) return cudaSuccess;
        unsigned int threads = 256;
        unsigned int blocks = 1;
        size_t shared_mem_size = threads * sizeof(T);
        _max_vector<T><<<blocks, threads, shared_mem_size>>>(a, n, out);
        return cudaGetLastError();
    }
    template cudaError_t max_vector<float>(const float*, size_t, float*);

    template<typename T>
    cudaError_t multiply_vectors(const T* a, const T* b, size_t n, T* out) {
        if (n == 0) return cudaSuccess;
        unsigned threads = 256;
        unsigned blocks  = (static_cast<unsigned>(n) + threads - 1) / threads;
        _multiply_vectors<T, 256><<<blocks, threads>>>(a, b, n, out);
        return cudaGetLastError();
    }
    template cudaError_t multiply_vectors<float>(const float*, const float*, size_t, float*);

    template<typename T>
    cudaError_t dot_product(const T* a, const T* b, size_t n, T* out) {
        if (n == 0) return cudaSuccess;
        unsigned threads = 256;
        unsigned blocks  = 1;
        size_t shared_mem_size = threads * sizeof(T);
        _dot_product<T><<<blocks, threads, shared_mem_size>>>(a, b, n, out);
        return cudaGetLastError();
    }
    template cudaError_t dot_product<float>(const float*, const float*, size_t, float*);

    template<typename T>
    cudaError_t sum_vector(const T* input, size_t n, T* out) {
        if (n == 0) return cudaSuccess;
        unsigned threads = 1;
        unsigned blocks  = 1;
        size_t shared_mem_size = threads * sizeof(T);
        _sum_vector<T><<<blocks, threads, shared_mem_size>>>(input, n, out);
        return cudaGetLastError();
    }
    template cudaError_t sum_vector<float>(const float*, size_t, float*);

}
