#ifndef RIPEMD160_CUDA_H
#define RIPEMD160_CUDA_H

#include <stdint.h>
#include <cuda_runtime.h>

#define RIPEMD160_BLOCK_SIZE 64  // 512 bits
#define RIPEMD160_DIGEST_SIZE 20  // 160 bits

// Function to compute RIPEMD-160 on the GPU
__device__ void ripemd160_gpu(const uint8_t* msg, uint32_t msg_len, uint8_t* hash);

#endif
