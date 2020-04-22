project(coalescing LANGUAGES CXX CUDA)

# 3.8+ for CUDA
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting CMAKE_BUILD_TYPE=Release")
endif()

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

include_directories(PUBLIC SYSTEM include)

# Add line info to binaries to help with profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

add_executable(main main.cu)
