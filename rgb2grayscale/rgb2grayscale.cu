#include <stdio.h>
#include <time.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

__global__ void rgba_to_grey(unsigned char *d_grey, uchar4 *d_rgba,
                             const std::size_t rows, const std::size_t cols) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx >= rows || idy >= cols) return;

    uchar4 p = d_rgba[idx * cols + idy];
    d_grey[idx * cols + idy] =
        (unsigned char)(0.299f * p.x + 0.587f * p.y + 0.114f * p.z);
}

using namespace cv;

// parameter parsing
// pre kernel processing
// kernel processing
// post kernel processing
int main(int argc, char *argv[]) {
    clock_t start_t, end_t;
    double total_t;

    std::string inimg = std::string(argv[1]);
    std::string outimg = std::string(argv[2]);
    cv::Mat img = cv::imread(inimg);

    cv::Mat img_rgba, img_grey;
    cv::cvtColor(img, img_rgba, cv::COLOR_RGBA2BGRA);
    img_grey.create(img_rgba.rows, img_rgba.cols, CV_8UC1);

    uchar4 *h_rgba = (uchar4 *)img_rgba.ptr<unsigned char>(0);
    unsigned char *h_grey = img_grey.ptr<unsigned char>(0);

    uchar4 *d_rgba;
    unsigned char *d_grey;

    unsigned int np = img_rgba.rows * img_rgba.cols;
    cudaMalloc((void **)&d_rgba, sizeof(uchar4) * np);
    cudaMalloc((void **)&d_grey, sizeof(unsigned char) * np);
    cudaMemset((void **)d_grey, 0, sizeof(unsigned char) * np);
    cudaMemcpy((void **)d_rgba, (void **)h_rgba, sizeof(uchar4) * np,
               cudaMemcpyHostToDevice);

    std::size_t rows = img_rgba.rows;
    std::size_t cols = img_rgba.cols;

    const std::size_t BLOCK_WIDTH = 32;

    const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    unsigned int grid_x = (unsigned int)(rows / BLOCK_WIDTH + 1);
    unsigned int grid_y = (unsigned int)(cols / BLOCK_WIDTH + 1);
    const dim3 grid_size(grid_x, grid_y, 1);

    start_t = clock();
    rgba_to_grey<<<grid_size, block_size>>>(d_grey, d_rgba, rows, cols);
    end_t = clock();

    cudaMemcpy(h_grey, d_grey, sizeof(unsigned char) * np,
               cudaMemcpyDeviceToHost);

    cv::imwrite(outimg.c_str(), img_grey);

    // cv::imshow("rgb", img_rgba);
    // cv::imshow("grey", img_grey);
    // cv::waitKey(0);

    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Total time taken by GPU+CPU: %f\n", total_t);

    cudaFree(d_rgba);
    cudaFree(d_grey);

    return 0;
}
