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

add_executable(1-1-pinned-basic 1_1_pinned_basic.cu)
target_link_libraries(1-1-pinned-basic nvToolsExt)

add_executable(1-2-pinned-tiled 1_2_pinned_tiled.cu)
target_link_libraries(1-2-pinned-tiled nvToolsExt)

add_executable(1-3-pinned-joint 1_3_pinned_joint.cu)
target_link_libraries(1-3-pinned-joint nvToolsExt)

add_executable(2-1-pageable-basic 2_1_pageable_basic.cu)
target_link_libraries(2-1-pageable-basic nvToolsExt)

add_executable(2-2-pinned-basic 2_2_pinned_basic.cu)
target_link_libraries(2-2-pinned-basic nvToolsExt)

add_executable(2-3-pinned-tiled 2_3_pinned_tiled.cu)
target_link_libraries(2-3-pinned-tiled nvToolsExt)

add_executable(2-4-pinned-tiled-overlap 2_4_pinned_tiled_overlap.cu)
target_link_libraries(2-4-pinned-tiled-overlap nvToolsExt)

add_executable(2-5-pinned-joint 2_5_pinned_joint.cu)
target_link_libraries(2-5-pinned-joint nvToolsExt)

add_executable(2-6-pinned-joint-overlap 2_6_pinned_joint_overlap.cu)
target_link_libraries(2-6-pinned-joint-overlap nvToolsExt)