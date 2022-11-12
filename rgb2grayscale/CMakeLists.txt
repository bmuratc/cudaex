cmake_minimum_required(VERSION 3.16)
project( rgb2grayscale LANGUAGES CXX CUDA )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( rgb2grayscale rgb2grayscale.cu )
set_property(TARGET rgb2grayscale
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries( rgb2grayscale ${OpenCV_LIBS} )

