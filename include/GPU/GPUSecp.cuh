#ifndef GPUSECP
#define GPUSECP

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

//CUDA-specific parameters that determine occupancy and thread-count
//Please read more about them in CUDA docs and adjust according to your GPU specs
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 44800    // Aumentando para 4x mais blocos - RTX 3060 pode suportar muito mais
#define SIZE_CUDA_STACK 32768    //GPU stack size in bytes that will be allocated to each thread - has complex functionality - please read cuda docs about this

//---------------------------------------------------------------------------------------------------------------------------
// Don't edit configuration below this line
//---------------------------------------------------------------------------------------------------------------------------

#define SIZE_HASH160 20        // Each Hash160 is 20 bytes
#define SIZE_PRIV_KEY 32       // Length of the private key that is generated from input seed (in bytes)
#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)
#define COUNT_CUDA_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

// Declarações das funções device
__device__ unsigned short reverseBytes16(unsigned short val);
__device__ unsigned long reverseBytes64(unsigned long val);
__device__ void _PointMultiSecp256k1(uint64_t* rx, uint64_t* ry, uint16_t* privKey, uint8_t* gTableX, uint8_t* gTableY);

// Declaração do kernel
__global__ void computePublicKeyKernel(uint64_t* privateKey, uint8_t* gTableX, uint8_t* gTableY, uint64_t* outputX, uint64_t* outputY);

class GPUSecp
{
private:
    int gpuId;
    cudaDeviceProp deviceProp;
    
    // Buffers GPU
    uint8_t* gTableXGPU;
    uint8_t* gTableYGPU;
    uint64_t* outputBufferGPU;
    uint8_t* outputHashesGPU;
    uint8_t* outputPrivKeysGPU;
    uint64_t* inputHashBufferGPU;
    
    // Buffers CPU
    uint64_t* outputBufferCPU;
    uint8_t* outputHashesCPU;
    uint8_t* outputPrivKeysCPU;
    
    int numPositions;  // Novo: número de posições x no template
    
    void initializeGPU();
    void copyTargetHashToDevice(const uint8_t* targetHash);

public:
    GPUSecp(
        const uint8_t* gTableXCPU,
        const uint8_t* gTableYCPU,
        const uint8_t* targetHash160,
        int gpuId = 0
    );
    
    ~GPUSecp();
    
    void computePublicKey(uint64_t* privateKey, uint64_t* publicKeyX, uint64_t* publicKeyY);
    void searchForHash160Target(uint64_t startIndex, uint64_t numThreads, uint8_t* targetHash160);
    void setTemplateAndPositions(const uint8_t* template_bytes, const int* positions);
    void doFreeMemory();
};

#endif // GPUSECP