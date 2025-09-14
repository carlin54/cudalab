#include <limits>
#include <random>
#include <algorithm>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/cudalab/library.cuh"

TEST(CudaLab, AddVectors) {
    const int n = 1024;
    float *da{}, *db{}, *dc{};
    ASSERT_EQ(cudaMalloc(&da, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&db, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dc, n * sizeof(float)), cudaSuccess);

    std::vector<float> ha(n, 1.0f), hb(n, 2.0f), hc(n, 0.0f);

    ASSERT_EQ(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudalab::add_vectors<float>(da, db, dc, n), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(hc[i], 3.0f);
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

using MinVectorParams = std::tuple<size_t, int, size_t, bool, bool, bool>;
class MinVectorTest : public ::testing::TestWithParam<MinVectorParams> {};

TEST_P(MinVectorTest, ComputesMinCorrectly) {
    const auto [ n, shuffle_n, min_count,
                should_generate_vector_randomly,
                should_generate_min_value_randomly,
                should_populate_min_value_randomly] = GetParam();

    std::vector<float> vector_host(n, std::numeric_limits<float>::max());

    // Random floats in [-n + 1, n - 1]
    std::mt19937 rng(42);
    const float lo = -static_cast<float>(n) + 1.0f;
    const float hi = static_cast<float>(n) - 1.0f;
    std::uniform_real_distribution<float> min_dist(lo, hi);

    // Populate the vector
    for (size_t i = 0; i < n; ++i) {
        vector_host[i] = should_generate_vector_randomly
                         ? min_dist(rng)
                         : static_cast<float>(i);
    }

    const float min_value = should_generate_min_value_randomly ? (lo - 1.0f)
                                                               : std::numeric_limits<float>::lowest();

    // Min index
    std::uniform_int_distribution<unsigned> idx_dist(0, n - 1);
    for (size_t i = 0; i < min_count; ++i) {
        unsigned min_idx = should_populate_min_value_randomly
                           ? idx_dist(rng)
                           : (i % n);
        vector_host[min_idx] = min_value;
    }

    // Rotate the vector
    std::rotate(vector_host.begin(), vector_host.begin() + static_cast<long>(shuffle_n % n), vector_host.end());
    std::vector<float> result_host(1, std::numeric_limits<float>::max());
    ASSERT_LE(min_value, *std::min_element(vector_host.begin(), vector_host.end()));

    float *vector_dev{};
    float *result_dev{};
    ASSERT_EQ(cudaMalloc(&vector_dev, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&result_dev, sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(vector_dev, vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_dev, result_host.data(), sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudalab::min_vector<float>(vector_dev, n, result_dev), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_host.data(), result_dev, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_FLOAT_EQ(result_host[0], min_value);

    ASSERT_EQ(cudaFree(vector_dev), cudaSuccess);
    ASSERT_EQ(cudaFree(result_dev), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
        AllPermutations,
        MinVectorTest,
        ::testing::Combine(
                ::testing::Values(
                       1, 2, 4, 8, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 6, 1 << 10, 1 << 16, 1 << 20,
                       (1 << 2)-1, (1 << 3)-1, (1 << 4)-1, (1 << 6)-1, (1 << 10)-1, (1 << 16)-1, (1 << 20)-1
                        ),       // n
                ::testing::Values(0, 1 << 10, 1 << 16, 1 << 20),    // shuffle_n
                ::testing::Values(1, 1 << 10, 1 << 16, 1 << 20),    // min_count
                ::testing::Values(false, true),                     // should_generate_vector_randomly
                ::testing::Values(false, true),                     // should_generate_min_value_randomly
                ::testing::Values(false, true)                      // should_populate_min_value_randomly
        ));


using MaxVectorParams = std::tuple<size_t, int, size_t, bool, bool, bool>;
class MaxVectorTest : public ::testing::TestWithParam<MaxVectorParams> {};

TEST_P(MaxVectorTest, ComputesMaxCorrectly) {
    const auto [ n, shuffle_n, max_count,
            should_generate_vector_randomly,
            should_generate_min_value_randomly,
            should_populate_min_value_randomly] = GetParam();

    std::vector<float> vector_host(n, std::numeric_limits<float>::max());

    // Random floats in [-n + 1, n - 1]
    std::mt19937 rng(42);
    const float lo = -static_cast<float>(n) + 1.0f;
    const float hi = static_cast<float>(n) - 1.0f;
    std::uniform_real_distribution<float> min_dist(lo, hi);

    // Populate the vector
    for (size_t i = 0; i < n; ++i) {
        vector_host[i] = should_generate_vector_randomly
                         ? min_dist(rng)
                         : static_cast<float>(i);
    }

    const float max_value = should_generate_min_value_randomly ? (hi + 1.0f)
                                                               : std::numeric_limits<float>::max();

    // Min index
    std::uniform_int_distribution<unsigned> idx_dist(0, n - 1);
    for (size_t i = 0; i < max_count; ++i) {
        unsigned max_idx = should_populate_min_value_randomly
                           ? idx_dist(rng)
                           : (i % n);
        vector_host[max_idx] = max_value;
    }

    // Rotate the vector
    std::rotate(vector_host.begin(), vector_host.begin() + static_cast<long>(shuffle_n % n), vector_host.end());
    std::vector<float> result_host(1, std::numeric_limits<float>::max());
    ASSERT_LE(max_value, *std::max_element(vector_host.begin(), vector_host.end()));

    float *vector_dev{};
    float *result_dev{};
    ASSERT_EQ(cudaMalloc(&vector_dev, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&result_dev, sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(vector_dev, vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_dev, result_host.data(), sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudalab::max_vector<float>(vector_dev, n, result_dev), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_host.data(), result_dev, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_FLOAT_EQ(result_host[0], max_value);

    ASSERT_EQ(cudaFree(vector_dev), cudaSuccess);
    ASSERT_EQ(cudaFree(result_dev), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
        AllPermutations,
        MaxVectorTest,
        ::testing::Combine(
                ::testing::Values(
                        1, 2, 4, 8, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 6, 1 << 10, 1 << 16, 1 << 20,
                        (1 << 2)-1, (1 << 3)-1, (1 << 4)-1, (1 << 6)-1, (1 << 10)-1, (1 << 16)-1, (1 << 20)-1
                ),       // n
                ::testing::Values(0, 1 << 10, 1 << 16, 1 << 20),    // shuffle_n
                ::testing::Values(1, 1 << 10, 1 << 16, 1 << 20),    // min_count
                ::testing::Values(false, true),                     // should_generate_vector_randomly
                ::testing::Values(false, true),                     // should_generate_min_value_randomly
                ::testing::Values(false, true)                      // should_populate_min_value_randomly
        ));

using MultiplyVectorParams = std::tuple<size_t>;
class MultiplyVectorTest : public ::testing::TestWithParam<MultiplyVectorParams> {};

TEST_P(MultiplyVectorTest, ComputesMinCorrectly) {
    const auto [n] = GetParam();

    std::vector<float> a_vector_host(n, 0);
    std::vector<float> b_vector_host(n, 0);
    std::vector<float> actual_c_vector_host(n, 0);
    std::vector<float> expected_c_vector_host(n, 0);

    // Random floats in [-n + 1, n - 1]
    std::mt19937 rng(42);
    const float lo = -static_cast<float>(n) + 1.0f;
    const float hi = static_cast<float>(n) - 1.0f;
    std::uniform_real_distribution<float> dist(lo, hi);

    // Populate the vector
    for (size_t i = 0; i < n; ++i) {
        a_vector_host[i] = dist(rng);
        b_vector_host[i] = dist(rng);
        expected_c_vector_host[i] = a_vector_host[i] * b_vector_host[i];
        actual_c_vector_host[i] = a_vector_host[i] * b_vector_host[i];
    }

    float* a_vector_dev{};
    float* b_vector_dev{};
    float* actual_c_vector_dev{};
    ASSERT_EQ(cudaMalloc(&a_vector_dev, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&b_vector_dev, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&actual_c_vector_dev, n * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(a_vector_dev, a_vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(b_vector_dev, b_vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_c_vector_dev, actual_c_vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudalab::multiply_vectors<float>(a_vector_dev, b_vector_dev, n, actual_c_vector_dev), cudaSuccess);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_c_vector_host.data(), actual_c_vector_dev, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < actual_c_vector_host.size(); i++) {
        EXPECT_FLOAT_EQ(actual_c_vector_host[i], expected_c_vector_host[i]);
    }

    ASSERT_EQ(cudaFree(a_vector_dev), cudaSuccess);
    ASSERT_EQ(cudaFree(b_vector_dev), cudaSuccess);
    ASSERT_EQ(cudaFree(actual_c_vector_dev), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
        AllPermutations,
        MultiplyVectorTest,
        ::testing::Combine(
                ::testing::Values(
                        1, 2, 4, 8, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 6, 1 << 10, 1 << 16, 1 << 20, 1 << 24,
                        (1 << 2)-1, (1 << 3)-1, (1 << 4)-1, (1 << 6)-1, (1 << 10)-1, (1 << 16)-1, (1 << 20)-1,
                        (1 << 24)-1
                )
        ));

using SumVectorParams = std::tuple<size_t>;
class SumVectorTest : public ::testing::TestWithParam<SumVectorParams> {};

TEST_P(SumVectorTest, ComputesSumCorrectly) {
    const auto [n] = GetParam();

    // Random floats in [-n + 1, n - 1]
    std::mt19937 rng(42);
    const float lo = -static_cast<float>(n) + 1.0f;
    const float hi = static_cast<float>(n) - 1.0f;
    std::uniform_real_distribution<float> dist(lo, hi);

    // Populate the vector
    std::vector<float> vector_host(n, 0);
    float actual_sum_host = std::numeric_limits<float>::lowest();
    float expected_sum_host = 0.0;
    for (size_t i = 0; i < n; ++i) {
        vector_host[i] = dist(rng);
        expected_sum_host += vector_host[i];
    }

    float* vector_dev{};
    float* actual_sum_dev{};
    ASSERT_EQ(cudaMalloc(&vector_dev, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&actual_sum_dev, sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(vector_dev, vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_sum_dev, &actual_sum_host, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudalab::sum_vector<float>(vector_dev, n, actual_sum_dev), cudaSuccess);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_NE(actual_sum_host, expected_sum_host);
    ASSERT_EQ(cudaMemcpy(&actual_sum_host, actual_sum_dev, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FLOAT_EQ(actual_sum_host, expected_sum_host);

    ASSERT_EQ(cudaFree(vector_dev), cudaSuccess);
    ASSERT_EQ(cudaFree(actual_sum_dev), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
        AllPermutations,
        SumVectorTest,
        ::testing::Combine(
                ::testing::Values(
                        1, 2, 4, 8, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 6, 1 << 10, 1 << 16, 1 << 20, 1 << 24,
                        (1 << 2)-1, (1 << 3)-1, (1 << 4)-1, (1 << 6)-1, (1 << 10)-1, (1 << 16)-1, (1 << 20)-1,
                        (1 << 24)-1
                )
        ));

using DotProductVectorParams = std::tuple<size_t>;
class DotProductVectorTest : public ::testing::TestWithParam<SumVectorParams> {};

TEST_P(DotProductVectorTest, ComputesDotProductCorrectly) {
    const auto [n] = GetParam();

    std::vector<float> a_vector_host(n, 0);
    std::vector<float> b_vector_host(n, 0);

    // Random floats in [-n + 1, n - 1]
    std::mt19937 rng(42);
    const float lo = -static_cast<float>(n) + 1.0f;
    const float hi = static_cast<float>(n) - 1.0f;
    std::uniform_real_distribution<float> dist(lo, hi);

    // Populate the vector
    float expected_product_host = 0.0f;
    float actual_product_host = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < n; ++i) {
        a_vector_host[i] = dist(rng);
        b_vector_host[i] = dist(rng);
        expected_product_host += a_vector_host[i] * b_vector_host[i];
    }

    float* a_vector_dev{};
    float* b_vector_dev{};
    float* actual_product_dev{};
    ASSERT_EQ(cudaMalloc(&a_vector_dev, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&b_vector_dev, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&actual_product_dev, sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(a_vector_dev, a_vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(b_vector_dev, b_vector_host.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_product_dev, &actual_product_host, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudalab::dot_product<float>(a_vector_dev, b_vector_dev, n, actual_product_dev), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_NE(actual_product_host, expected_product_host);
    ASSERT_EQ(cudaMemcpy(&actual_product_host, actual_product_dev, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    const float abs_eps = 1e-6f;
    const float rel_per_elem = 2e-7f;
    const float tol = abs_eps + rel_per_elem * static_cast<float>(n) * std::fabs(expected_product_host);
    EXPECT_NEAR(actual_product_host, expected_product_host, tol);

    ASSERT_EQ(cudaFree(a_vector_dev), cudaSuccess);
    ASSERT_EQ(cudaFree(b_vector_dev), cudaSuccess);
    ASSERT_EQ(cudaFree(actual_product_dev), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
        AllPermutations,
        DotProductVectorTest,
        ::testing::Combine(
                ::testing::Values(
                        1, 2, 4, 8, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 6, 1 << 10, 1 << 16, 1 << 20, 1 << 24,
                        (1 << 2)-1, (1 << 3)-1, (1 << 4)-1, (1 << 6)-1, (1 << 10)-1, (1 << 16)-1, (1 << 20)-1,
                        (1 << 24)-1
                )
        ));
