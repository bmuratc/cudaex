cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

project( rgb2grayscale LANGUAGES CXX CUDA )

find_package( OpenCV REQUIRED )

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( rgb2grayscale rgb2grayscale.cu )
set_property(TARGET rgb2grayscale
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries( rgb2grayscale ${OpenCV_LIBS} )

