#ifndef CUDALAB_LIBRARY_CUH
#define CUDALAB_LIBRARY_CUH
#include <limits>
#include <cuda_runtime.h>
#include <math_constants.h>
namespace cudalab {
    template<typename T>
    cudaError_t add_vectors(const T* a, const T* b, T* out, int n);
    extern template cudaError_t add_vectors<float>(const float*, const float*, float*, int);

    template<typename T>
    cudaError_t multiply_vectors(const T* a, const T* b, T* out, int n);
    extern template cudaError_t multiply_vectors<float>(const float*, const float*, float*, int);

    template<typename T>
    cudaError_t sum_vectors(const T* a, const T* b, T* out, int n);
    extern template cudaError_t sum_vectors<float>(const float*, const float*, float*, int);

    template<typename T>
    cudaError_t dot_product(const T* a, const T* b, T* out);
    extern template cudaError_t dot_product<float>(const float*, const float*, float*, int);

    template<typename T>
    cudaError_t min_vector(const T* vec, size_t n, T* out);
    extern template cudaError_t min_vector<float>(const float*, std::size_t, float*);

    template<typename T>
    cudaError_t max_vector(const T* vec, size_t n, T* out);
    extern template cudaError_t max_vector<float>(const float*, std::size_t, float*);

}

#endif //CUDALAB_LIBRARY_CUH
