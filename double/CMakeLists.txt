cmake_minimum_required(VERSION 3.16)

# https://stackoverflow.com/questions/27320527/cuda-compilation-of-examples%5B/url%5D
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

project( double LANGUAGES CXX CUDA )

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable( double double.cu )
set_property(TARGET double 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

