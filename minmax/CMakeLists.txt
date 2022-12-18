cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

project( minmax LANGUAGES CXX CUDA )

find_package( OpenCV REQUIRED )

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( minmax minmax.cu )
set_property(TARGET minmax
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries( minmax ${OpenCV_LIBS} )

