#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
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

//

__global__ void minmax(float *d_input, float *d_output, const std::size_t len) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = d_input[idx];
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] =
                fminf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) d_output[blockIdx.x] = sdata[threadIdx.x];
}

struct Container {
    float *h_input;
    float *h_output;
    float *d_input;
    float *d_output;

    unsigned int h_input_dim;
    unsigned int h_output_dim;
    unsigned int d_input_dim;
    unsigned int d_output_dim;

    unsigned int input_length;
    unsigned int output_length;

    float min_element{std::numeric_limits<float>::max()};
    void allocate() {}
};

void preprocessing(Container &data) {
    data.h_input_dim = 1;
    data.input_length = 1 << 16;
    data.output_length = data.input_length;
    data.h_input = (float *)malloc(data.input_length * sizeof(float));
    data.h_output = (float *)malloc(data.output_length * sizeof(float));

    srand(time(0));

    auto frand = []() { return rand() / static_cast<float>(RAND_MAX); };

    std::generate(data.h_input, data.h_input + data.input_length, frand);

    data.min_element =
        *std::min_element(data.h_input, data.h_input + data.input_length);

    cudaMalloc((void **)&data.d_input, data.input_length * sizeof(float));
    cudaMalloc((void **)&data.d_output, data.input_length * sizeof(float));

    cudaMemset((void *)data.d_output, 0., data.input_length * sizeof(float));
    cudaMemcpy((void *)data.d_output, (void *)data.h_input,
               sizeof(float) * data.input_length, cudaMemcpyHostToDevice);
}

void postprocessing(const Container &data) {
    cudaMemcpy(data.h_output, data.d_output, sizeof(float) * data.output_length,
               cudaMemcpyDeviceToHost);

    std::cout << "Output Data" << std::endl;
    for (int i = 0; i < data.input_length; i++) {
        if (i % 16 == 0) std::cout << std::endl;
        std::cout << std::setw(8) << std::setprecision(6)
                  << *(data.h_output + i) << "\t";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Calculated Min : " << *(data.h_output) << std::endl;
    std::cout << "Expected   Min : " << data.min_element << std::endl;
    if (std::fabs(data.min_element - *(data.h_output)) < DBL_EPSILON) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    free(data.h_input);
    free(data.h_output);
    cudaFree(data.d_input);
    cudaFree(data.d_output);
}

int main(int argc, char *argv[]) {
    Container dataContainer;

    // prepration of dataContainer
    preprocessing(dataContainer);
    // std::cout << "Input Data" << std::endl;
    // for (int i = 0; i < dataContainer.input_length; i++) {
    //     if (i % 16 == 0) std::cout << std::endl;
    //     std::cout << std::setw(8) << std::setprecision(6)
    //               << *(dataContainer.h_input + i) << "\t";
    // }
    // std::cout << std::endl;

    std::cout << "Input Data Prepared" << std::endl;

    clock_t start_t, end_t;
    double total_t;

    // preparation of kernel call
    const std::size_t BLOCK_WIDTH = 1 << 7;
    const dim3 blockSize(BLOCK_WIDTH, 1, 1);

    int input_length = dataContainer.input_length;
    while (true) {
        unsigned int GRID_WIDTH = ceil((float)input_length / BLOCK_WIDTH);
        std::cout << "input_length: " << input_length << std::endl;
        std::cout << "GRID_WIDTH: " << GRID_WIDTH << std::endl;
        const dim3 gridSize(GRID_WIDTH, 1, 1);
        total_t = 0;
        start_t = clock();
        minmax<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(
            dataContainer.d_output, dataContainer.d_output, input_length);
        end_t = clock();
        total_t += (double)(end_t - start_t) / CLOCKS_PER_SEC;

        input_length = GRID_WIDTH;
        if (GRID_WIDTH == 1) break;
    }
    dataContainer.input_length = input_length;

    postprocessing(dataContainer);

    printf("\nTotal time GPU OP: %f\n", total_t / 10);

    gpuErrchk(cudaPeekAtLastError());

    return 0;
}
