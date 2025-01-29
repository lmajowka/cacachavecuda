#ifndef GPU_CONSTANTS_H
#define GPU_CONSTANTS_H

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 44800
#define SIZE_CUDA_STACK 32768

#define SIZE_HASH160 20        // Each Hash160 is 20 bytes
#define SIZE_PRIV_KEY 32 	   // Length of the private key that is generated from input seed (in bytes)
#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)
#define COUNT_CUDA_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

//Contains the first element index for each chunk
//Pre-computed to save one multiplication
inline __constant__ int CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536*0,  65536*1,  65536*2,  65536*3,
  65536*4,  65536*5,  65536*6,  65536*7,
  65536*8,  65536*9,  65536*10, 65536*11,
  65536*12, 65536*13, 65536*14, 65536*15,
};

struct CUDAStream {
    cudaStream_t stream;
    unsigned char *d_private_key;
    unsigned char *d_bitcoin_address;
    int *d_match_found;
    bool in_use;
};

#endif // GPU_CONSTANTS_H