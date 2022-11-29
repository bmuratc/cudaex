#include <stdio.h>

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        exit(code);
    }
}

__device__ int globalIdx() { return blockIdx.x * blockDim.x + threadIdx.x; }

__global__ void double_it(float *d_out, float *d_in) {
    int id = globalIdx();
    d_out[id] = d_in[id] + d_in[id];
}

int main(int argc, char **argv) {
    const int ARRAY_SIZE = 1 << 16;
    const int BLOCK_SIZE = 32 << 2;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    float *d_in;
    float *d_out;

    cudaMalloc((void **)&d_in, ARRAY_BYTES);
    cudaMalloc((void **)&d_out, ARRAY_BYTES);
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    dim3 gridSize = (ARRAY_SIZE >= BLOCK_SIZE ? ARRAY_SIZE / BLOCK_SIZE : 1);
    dim3 blockSize = (ARRAY_SIZE >= BLOCK_SIZE ? BLOCK_SIZE : ARRAY_SIZE);

    double_it<<<gridSize, blockSize>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (h_out[i] != i << 1) {
            printf("Failed!!!");
            return 1;
        }
    }

    printf("Successful!!!\n");

    cudaFree(d_in);
    cudaFree(d_out);

    gpuErrchk(cudaPeekAtLastError());

    return 0;
}
