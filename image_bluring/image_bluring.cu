#include <stdio.h>
#include <time.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "../util/params.h"

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
__global__ void bluring(const uchar4 *d_rgba, uchar4 *d_blur,
                        const unsigned int cols, const unsigned int rows,
                        const float *d_gaussian_blur,
                        const unsigned int gBlur_cols) {
    // unsigned int idx = globalIdx_X();
    // unsigned int idy = globalIdx_Y();

    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= cols || idy >= rows) return;

    //    copy to SMEM
    //    __sycnThread();
    //    __device olarak kernel disinda tanimlanabilir mi?
    const int middle = gBlur_cols / 2;

    float blurx = 0.0;
    float blury = 0.0;
    float blurz = 0.0;

    for (int i = -middle; i <= middle; ++i) {
        for (int j = -middle; j <= middle; ++j) {
            const unsigned int x = max(0, min(cols - 1, idx + j));
            const unsigned int y = max(0, min(rows - 1, idy + i));

            const float w = d_gaussian_blur[(j + gBlur_cols / 2) +
                                            (i + gBlur_cols / 2) * gBlur_cols];
            blurx += w * d_rgba[x + y * cols].x;
            blury += w * d_rgba[x + y * cols].y;
            blurz += w * d_rgba[x + y * cols].z;
        }
    }
    d_blur[idx + idy * cols].x = static_cast<unsigned char>(blurx);
    d_blur[idx + idy * cols].y = static_cast<unsigned char>(blury);
    d_blur[idx + idy * cols].z = static_cast<unsigned char>(blurz);
    d_blur[idx + idy * cols].w =
        static_cast<unsigned char>(d_rgba[idx + idy * cols].w);
}

struct GaussianBlur {
    GaussianBlur()
        : data{{1. / 16, 2. / 16, 1. / 16},
               {2. / 16, 4. / 16, 2. / 16},
               {1. / 16, 2. / 16, 1. / 16}},
          rows{3},
          cols{3} {};
    float data[3][3];
    std::size_t rows;
    std::size_t cols;
};

int main(int argc, char *argv[]) {
    clock_t start_t, end_t;
    double total_t;

    enum class FilterType { GAUSSION_BLUR };
    std::unordered_map<std::string, FilterType> available_filters{
        {"gaussian_blur", FilterType::GAUSSION_BLUR}};

    program_param params(argc, argv);
    params("-i,--infile", "", "Input image (Mandatory)")(
        "-o,--outfile", "", "Output image (Mandatory)")("--help", "Usage")(
        "-f,--filter", "gaussian_blur", "Selected filter");

    switch (params.parse()) {
        case program_param::NOK: {
            return 1;
        } break;
        case program_param::HELP: {
            params.print_usage();
            return 0;
        } break;
    }

    std::string in_img_name = params.get_param<std::string>("-i");
    std::cout << "input image : " << in_img_name << std::endl;
    std::string out_img_name = params.get_param<std::string>("-o");
    std::cout << "output image : " << out_img_name << std::endl;
    std::string filter_name = params.get_param<std::string>("-f");
    std::cout << "Selecte filter : " << filter_name << std::endl;

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

    cv::Mat img_blur(image_rows, image_cols, CV_8UC4);

    uchar4 *h_rgba = img_rgba.ptr<uchar4>(0);
    uchar4 *h_blur = img_blur.ptr<uchar4>(0);

    uchar4 *d_rgba;
    uchar4 *d_blur;

    float *d_gaussian_blur;

    cudaMalloc((void **)&d_rgba, sizeof(uchar4) * image_dim_size);
    cudaMalloc((void **)&d_blur, sizeof(uchar4) * image_dim_size);
    cudaMemset((void *)d_blur, 0, sizeof(uchar4) * image_dim_size);
    cudaMemcpy((void *)d_rgba, (void *)h_rgba, sizeof(uchar4) * image_dim_size,
               cudaMemcpyHostToDevice);

    const std::size_t BLOCK_WIDTH = 16;
    const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    unsigned int grid_x = (unsigned int)(image_cols / BLOCK_WIDTH);
    unsigned int grid_y = (unsigned int)(image_rows / BLOCK_WIDTH);
    const dim3 gridSize(grid_x, grid_y, 1);

    GaussianBlur gBlur;

    cudaMalloc((void **)&d_gaussian_blur,
               sizeof(float) * gBlur.rows * gBlur.cols);

    cudaMemcpy((void *)d_gaussian_blur, (void *)gBlur.data,
               sizeof(float) * gBlur.rows * gBlur.cols, cudaMemcpyHostToDevice);

    total_t = 0;
    start_t = clock();
    bluring<<<gridSize, blockSize>>>(d_rgba, d_blur, image_cols, image_rows,
                                     d_gaussian_blur, gBlur.cols);
    end_t = clock();
    total_t += (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Total time GPU OP: %f\n", total_t / 10);

    cudaMemcpy(h_blur, d_blur, sizeof(uchar4) * image_dim_size,
               cudaMemcpyDeviceToHost);

    cv::Mat img_blur_out;
    cv::cvtColor(img_blur, img_blur_out, cv::COLOR_BGRA2RGBA);
    cv::imwrite(out_img_name.c_str(), img_blur_out);

    // cv::imshow("grey", img_blur);
    // cv::imshow("rgb", img);
    // cv::waitKey(0);

    cudaFree(d_rgba);
    cudaFree(d_blur);

    gpuErrchk(cudaPeekAtLastError());

    return 0;
}
