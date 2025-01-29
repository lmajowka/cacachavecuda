#include "GPU/GPUSecp.h"
#include "GPU/GPUConstants.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPU/GPUMath.h"

using namespace std;

__constant__ uint8_t d_template[32];
__constant__ int d_positions[12];
__constant__ int d_numPositions;

// Função de verificação de erros CUDA
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        printf("cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

__device__ void _PointMultiSecp256k1(uint64_t* rx, uint64_t* ry, uint16_t* privKey, uint8_t* gTableX, uint8_t* gTableY) {
    int chunk = 0;
    uint64_t qz[5] = {1, 0, 0, 0, 0};

    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
      if (privKey[chunk] > 0) {
        int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
        memcpy(rx, gTableX + index, SIZE_GTABLE_POINT);
        memcpy(ry, gTableY + index, SIZE_GTABLE_POINT);
        chunk++;
        break;
      }
    }

    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
      if (privKey[chunk] > 0) {
        uint64_t gx[4];
        uint64_t gy[4];

        int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
        
        memcpy(gx, gTableX + index, SIZE_GTABLE_POINT);
        memcpy(gy, gTableY + index, SIZE_GTABLE_POINT);

        _PointAddSecp256k1(rx, ry, qz, gx, gy);
      }
    }

    _ModInv(qz);
    _ModMult(rx, qz);
    _ModMult(ry, qz);
}

__device__ uint16_t reverseBytes16(uint16_t value) {
    return (value >> 8) | (value << 8);
}

__global__ void computePublicKeyKernel(uint64_t* privateKey, uint8_t* gTableX, uint8_t* gTableY, uint64_t* outputX, uint64_t* outputY) {
    uint64_t qx[4], qy[4];
    uint16_t privKeyChunks[16];

    // Converter os bytes da chave privada de big-endian para little-endian
    uint16_t* privKeyShorts = (uint16_t*)privateKey;
    for (int i = 0; i < 16; i++) {
        privKeyChunks[15 - i] = reverseBytes16(privKeyShorts[i]);
    }

    _PointMultiSecp256k1(qx, qy, privKeyChunks, gTableX, gTableY);

    // Verificar se é necessário inverter a ordem dos blocos
    for (int i = 0; i < 4; i++) {
        outputX[i] = qx[3 - i]; // Inverter a posição dos blocos
        outputY[i] = qy[3 - i];
    }
}


void GPUSecp::computePublicKey(uint64_t* privateKey, uint64_t* publicKeyX, uint64_t* publicKeyY) {
    // Garantir que estamos usando a GPU correta
    cudaSetDevice(gpuId);

    // Alocar memória na GPU para a chave pública
    uint64_t* outputXGPU;
    uint64_t* outputYGPU;
    CudaSafeCall(cudaMalloc(&outputXGPU, 4 * sizeof(uint64_t)));
    CudaSafeCall(cudaMalloc(&outputYGPU, 4 * sizeof(uint64_t)));

    // Alocar e copiar a chave privada para a GPU
    uint64_t* privateKeyGPU;
    CudaSafeCall(cudaMalloc(&privateKeyGPU, 4 * sizeof(uint64_t)));
    CudaSafeCall(cudaMemcpy(privateKeyGPU, privateKey, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Executar o kernel
    computePublicKeyKernel<<<1, 1>>>(privateKeyGPU, gTableXGPU, gTableYGPU, outputXGPU, outputYGPU);
    CudaSafeCall(cudaGetLastError());
    CudaSafeCall(cudaDeviceSynchronize());

    // Copiar resultado de volta para a CPU
    CudaSafeCall(cudaMemcpy(publicKeyX, outputXGPU, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(publicKeyY, outputYGPU, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Liberar memória
    CudaSafeCall(cudaFree(privateKeyGPU));
    CudaSafeCall(cudaFree(outputXGPU));
    CudaSafeCall(cudaFree(outputYGPU));
}

GPUSecp::GPUSecp(
    const uint8_t* gTableXCPU,
    const uint8_t* gTableYCPU,
    const uint8_t* targetHash160,
    int gpuId
) : gpuId(gpuId) {
    
    // Selecionar a GPU específica
    cudaSetDevice(gpuId);
    
    // Obter propriedades do dispositivo
    cudaGetDeviceProperties(&deviceProp, gpuId);
    
    // Copiar tabelas G para a GPU
    CudaSafeCall(cudaMalloc(&gTableXGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
    CudaSafeCall(cudaMalloc(&gTableYGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
    CudaSafeCall(cudaMemcpy(gTableXGPU, gTableXCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gTableYGPU, gTableYCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));
}

GPUSecp::~GPUSecp() {
    if (gTableXGPU) cudaFree(gTableXGPU);
    if (gTableYGPU) cudaFree(gTableYGPU);
}