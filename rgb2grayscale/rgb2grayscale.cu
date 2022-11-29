#include <stdio.h>
#include <time.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        exit(code);
    }
}

__device__ int globalIdx_X() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ int globalIdx_Y() { return blockIdx.y * blockDim.y + threadIdx.y; }

__global__ void rgba_to_grey(unsigned char *d_grey, uchar4 *d_rgba,
                             const std::size_t rows, const std::size_t cols) {
    unsigned int idx = globalIdx_X();
    unsigned int idy = globalIdx_Y();

    if (idx >= rows || idy >= cols) return;

    uchar4 p = d_rgba[idx * cols + idy];
    d_grey[idx * cols + idy] =
        (unsigned char)(0.299f * p.x + 0.587f * p.y + 0.114f * p.z);
}

using namespace cv;

// pre kernel processing
// kernel processing
// post kernel processing

int main(int argc, char *argv[]) {
    clock_t start_t, end_t;
    double total_t;

    if (argc != 3) {
        std::cout << "usage: rgb2grayscale <infile> <outfile>" << std::endl;
        return 1;
    }

    std::string in_img_name = std::string(argv[1]);
    std::cout << "input image : " << in_img_name << std::endl;
    std::string out_img_name = std::string(argv[2]);
    std::cout << "output image : " << out_img_name << std::endl;

    cv::Mat img = cv::imread(in_img_name);
    if (img.empty()) {
        std::cout << "Error in reading image : " << in_img_name << std::endl;
        return 1;
    }
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, cv::COLOR_RGBA2BGRA);

    std::size_t image_rows = img_rgba.rows;
    std::size_t image_cols = img_rgba.cols;
    const std::size_t image_dim_size = image_rows * image_cols;

    cv::Mat img_grey(image_rows, image_cols, CV_8UC1);

    uchar4 *h_rgba = (uchar4 *)img_rgba.ptr<unsigned char>(0);
    unsigned char *h_grey = img_grey.ptr<unsigned char>(0);

    uchar4 *d_rgba;
    unsigned char *d_grey;

    cudaMalloc((void **)&d_rgba, sizeof(uchar4) * image_dim_size);
    cudaMalloc((void **)&d_grey, sizeof(unsigned char) * image_dim_size);
    cudaMemset((void *)d_grey, 0, sizeof(unsigned char) * image_dim_size);
    cudaMemcpy((void *)d_rgba, (void *)h_rgba, sizeof(uchar4) * image_dim_size,
               cudaMemcpyHostToDevice);

    const std::size_t BLOCK_WIDTH = 16;
    const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    unsigned int grid_x = (unsigned int)(image_rows / BLOCK_WIDTH);
    unsigned int grid_y = (unsigned int)(image_cols / BLOCK_WIDTH);
    const dim3 gridSize(grid_x, grid_y, 1);

    total_t = 0;
    start_t = clock();
    rgba_to_grey<<<gridSize, blockSize>>>(d_grey, d_rgba, image_rows,
                                          image_cols);
    end_t = clock();
    total_t += (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Total time GPU OP: %f\n", total_t / 10);

    cudaMemcpy(h_grey, d_grey, sizeof(unsigned char) * image_dim_size,
               cudaMemcpyDeviceToHost);

    cv::imwrite(out_img_name.c_str(), img_grey);

    // cv::imshow("grey", img_grey);
    // cv::imshow("rgb", img);
    // cv::waitKey(0);

    cudaFree(d_rgba);
    cudaFree(d_grey);

    gpuErrchk(cudaPeekAtLastError());

    return 0;
}
