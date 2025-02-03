#ifndef GPUSECP
#define GPUSECP

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "GPUConstants.h"

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

__device__ void _PointMultiSecp256k1(uint64_t* pubX, uint64_t* pubY, uint16_t* privKey, uint8_t* gTableX, uint8_t* gTableY);

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

#endif // GPUSecpH
