project(sgemm LANGUAGES CXX CUDA)

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

add_executable(sgemm-cpu cpu.cpp)

add_executable(0-pageable-basic 0_pageable_basic.cu)
target_link_libraries(0-pageable-basic nvToolsExt)

add_executable(1-pinned-basic 1_pinned_basic.cu)
target_link_libraries(1-pinned-basic nvToolsExt)

add_executable(2-pinned-tiled 2_pinned_tiled.cu)
target_link_libraries(2-pinned-tiled nvToolsExt)

add_executable(3-pinned-joint 3_pinned_joint.cu)
target_link_libraries(3-pinned-joint nvToolsExt)

add_executable(4-pinned-joint-wall 4_pinned_joint_wall.cu)
target_link_libraries(4-pinned-joint-wall nvToolsExt)

add_executable(5-pinned-joint-overlap 5_pinned_joint_overlap.cu)
target_link_libraries(5-pinned-joint-overlap nvToolsExt)