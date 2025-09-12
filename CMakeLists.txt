cmake_minimum_required(VERSION 3.27)
project(cudalab LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)

# ----- Debug info -----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -g -lineinfo")

find_package(CUDAToolkit REQUIRED)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

add_library(cudalab STATIC src/library.cu)
target_include_directories(cudalab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cudalab PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- GoogleTest ----
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# ---- Tests ----
enable_testing()
add_executable(cudalab_tests tests/test_library.cpp)
target_link_libraries(cudalab_tests PRIVATE cudalab CUDA::cudart GTest::gtest_main Threads::Threads)

include(GoogleTest)
gtest_discover_tests(cudalab_tests)
