#ifndef SHA256_CUDA_H
#define SHA256_CUDA_H

#include <stdint.h>
#include <cuda_runtime.h>

#define SHA256_BLOCK_SIZE 64  // 512 bits
#define SHA256_DIGEST_SIZE 32  // 256 bits

// Function to compute SHA-256 on the GPU
__device__ void sha256_gpu(const unsigned char* data, size_t len, unsigned char* hash);

#endif
