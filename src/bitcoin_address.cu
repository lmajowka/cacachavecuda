#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <secp256k1.h>
#include "sha256_cuda.cuh"
#include "ripemd160_cuda.cuh"
#include "GPU/GPUConstants.h"
#include "CPU/Int.h"
#include "CPU/Point.h"
#include "CPU/SECP256k1.h"
#include "GPU/GPUSecp.h"

// Variáveis globais para tabelas G na GPU
uint8_t *d_gTableX = nullptr;
uint8_t *d_gTableY = nullptr;

// Função para carregar as tabelas G
void loadGTable(uint8_t *gTableX, uint8_t *gTableY) {
    // Alocar memória temporária na CPU
    uint8_t *hostTableX = new uint8_t[NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT];
    uint8_t *hostTableY = new uint8_t[NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT];

    // Gerar tabelas na CPU
    Secp256K1 *secp = new Secp256K1();
    secp->Init();

    for (int i = 0; i < NUM_GTABLE_CHUNK; i++) {
        for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++) {
            int element = (i * NUM_GTABLE_VALUE) + j;
            Point p = secp->GTable[element];
            for (int b = 0; b < 32; b++) {
                hostTableX[(element * SIZE_GTABLE_POINT) + b] = p.x.GetByte64(b);
                hostTableY[(element * SIZE_GTABLE_POINT) + b] = p.y.GetByte64(b);
            }
        }
    }

    delete secp;

    cudaError_t err;
    err = cudaMemcpy(gTableX, hostTableX, 
                     NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT, 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Erro ao copiar gTableX para GPU: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(gTableY, hostTableY, 
                     NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT, 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Erro ao copiar gTableY para GPU: %s\n", cudaGetErrorString(err));
    }

    delete[] hostTableX;
    delete[] hostTableY;
}

void freeGPUTables() {
    if (d_gTableX) cudaFree(d_gTableX);
    if (d_gTableY) cudaFree(d_gTableY);
    d_gTableX = nullptr;
    d_gTableY = nullptr;
}

bool initGPUTables() {
    cudaError_t err;
    size_t tableSize = NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT;
    printf("Alocando %zu bytes para cada tabela...\n", tableSize);

    err = cudaMalloc(&d_gTableX, tableSize);
    if (err != cudaSuccess) {
        printf("Erro ao alocar memória para gTableX: %s\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc(&d_gTableY, tableSize);
    if (err != cudaSuccess) {
        printf("Erro ao alocar memória para gTableY: %s\n", cudaGetErrorString(err));
        cudaFree(d_gTableX);
        return false;
    }

    printf("Carregando tabelas...\n");
    loadGTable(d_gTableX, d_gTableY);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro após carregar tabelas: %s\n", cudaGetErrorString(err));
        freeGPUTables();
        return false;
    }

    printf("Tabelas carregadas com sucesso!\n");
    return true;
}

// Função device para incrementar a chave privada
__device__ void increment_private_key_gpu(unsigned char *private_key, uint64_t increment) {
    uint64_t carry = increment;
    for (int i = 31; i >= 0 && carry > 0; i--) {
        uint64_t sum = (uint64_t)private_key[i] + carry;
        private_key[i] = sum & 0xFF;
        carry = sum >> 8;
    }
}

// Kernel otimizado para processar múltiplas chaves em paralelo
__global__ void bitcoin_address_kernel(unsigned char* private_key, unsigned char* bitcoin_address, 
                                     const unsigned char* target_address, int* match_found,
                                     uint8_t* gTableX, uint8_t* gTableY) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (*match_found) return;

    // Buffer local para a chave privada
    unsigned char local_private_key[32];
    memcpy(local_private_key, private_key, 32);
    
    // Incrementa a chave baseado no thread ID
    increment_private_key_gpu(local_private_key, tid);

    // Buffers para as hashes
    unsigned char sha256_hash[SHA256_DIGEST_SIZE];
    unsigned char ripemd160_hash[RIPEMD160_DIGEST_SIZE];
    unsigned char public_key[33];  // Compressed public key format

    // Converter a chave privada para o formato correto
    uint16_t privKeyChunks[NUM_GTABLE_CHUNK] = {0};
    
    // Converter similar ao kernel host
    uint16_t* privKeyShorts = (uint16_t*)local_private_key;
    for (int i = 0; i < 16; i++) {
        // Reverse bytes e inverte a ordem dos chunks
        uint16_t value = privKeyShorts[i];
        value = ((value & 0xFF00) >> 8) | ((value & 0x00FF) << 8); // reverseBytes16
        privKeyChunks[15 - i] = value;
    }

    // Gerar public key usando as tabelas G
    uint64_t pubX[4], pubY[4];
    _PointMultiSecp256k1(pubX, pubY, privKeyChunks, gTableX, gTableY);

    // Converter para formato comprimido (33 bytes)
    public_key[0] = 0x02 | (pubY[0] & 1);
    for (int i = 0; i < 32; i++) {
        public_key[i+1] = ((unsigned char*)pubX)[31-i];
    }

    // Calcular hashes
    sha256_gpu(public_key, 33, sha256_hash);
    ripemd160_gpu(sha256_hash, SHA256_DIGEST_SIZE, ripemd160_hash);

    // Verificar match
    bool match = true;
    for (int i = 0; i < RIPEMD160_DIGEST_SIZE; i++) {
        if (ripemd160_hash[i] != target_address[i]) {
            match = false;
            break;
        }
    }

    // if (tid == 0) {
    //     printf("\n=== Primeiro Endereço ===\n");
    //     printf("Private Key: ");
    //     for (int i = 0; i < 32; i++) {
    //         printf("%02x", local_private_key[i]);
    //     }
        
    //     printf("\nPublic Key (compressed): ");
    //     for (int i = 0; i < 33; i++) {
    //         printf("%02x", public_key[i]);
    //     }

    //     printf("\nHash RIPEMD160: ");
    //     for (int i = 0; i < RIPEMD160_DIGEST_SIZE; i++) {
    //         printf("%02x", ripemd160_hash[i]);
    //     }
    //     printf("\n======================\n");
    // }

    if (match) {
        atomicExch(match_found, 1);
        memcpy(bitcoin_address, ripemd160_hash, RIPEMD160_DIGEST_SIZE);
        // Salvar a chave privada que gerou o match
        memcpy(private_key, local_private_key, 32);
    }
}

// Função auxiliar para converter hex string para bytes
bool hex_to_bytes(const char* hex_str, unsigned char* bytes, size_t length) {
    if (strlen(hex_str) != length * 2) return false;
    
    for (size_t i = 0; i < length; i++) {
        char hex_byte[3] = {hex_str[i*2], hex_str[i*2+1], 0};
        char* end_ptr;
        bytes[i] = (unsigned char)strtol(hex_byte, &end_ptr, 16);
        if (*end_ptr != 0) return false;
    }
    return true;
}

// Adicionar a função formatSpeed
const char* formatSpeed(double speed) {
    static char buffer[16];
    if (speed >= 1e9) {
        snprintf(buffer, sizeof(buffer), "%.2f Gkeys/s", speed / 1e9);
    } else if (speed >= 1e6) {
        snprintf(buffer, sizeof(buffer), "%.2f Mkeys/s", speed / 1e6);
    } else if (speed >= 1e3) {
        snprintf(buffer, sizeof(buffer), "%.2f Kkeys/s", speed / 1e3);
    } else {
        snprintf(buffer, sizeof(buffer), "%.2f keys/s", speed);
    }
    return buffer;
}

int main(int argc, char **argv) {
    int blockSize = 256;  // default
    int numBlocks = 2048; // default
    
    // Chave privada default
    unsigned char private_key[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
        0x83, 0x2E, 0xD7, 0x0F, 0x2B, 0x5C, 0x35, 0xEE
    };

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-block") == 0 && i + 1 < argc) {
            blockSize = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-grid") == 0 && i + 1 < argc) {
            numBlocks = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-private") == 0 && i + 1 < argc) {
            if (!hex_to_bytes(argv[i + 1], private_key, 32)) {
                printf("Erro: chave privada inválida. Use 64 caracteres hexadecimais\n");
                return 1;
            }
            i++;
        }
    }

    // Validar parâmetros
    if (blockSize <= 0 || blockSize > 1024) {
        printf("Erro: block size deve estar entre 1 e 1024\n");
        return 1;
    }
    if (numBlocks <= 0) {
        printf("Erro: grid size deve ser maior que 0\n");
        return 1;
    }

    printf("Configuração:\n");
    printf("Block Size: %d\n", blockSize);
    printf("Grid Size: %d\n", numBlocks);
    printf("Total Threads: %d\n", blockSize * numBlocks);
    printf("Private Key Inicial: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", private_key[i]);
    }
    printf("\n");

    printf("Inicializando tabelas G na GPU...\n");
    if (!initGPUTables()) {
        printf("Falha ao inicializar tabelas G. Abortando.\n");
        return 1;
    }

    unsigned char bitcoin_address[RIPEMD160_DIGEST_SIZE];

    unsigned char target_bitcoin_address[RIPEMD160_DIGEST_SIZE] = {
        0x20, 0xd4, 0x5a, 0x6a, 0x76, 0x25, 0x35, 0x70, 
        0x0c, 0xe9, 0xe0, 0xb2, 0x16, 0xe3, 0x19, 0x94, 
        0x33, 0x5d, 0xb8, 0xa5  
    };

    // Alocação de memória na GPU
    unsigned char *d_private_key;
    unsigned char *d_bitcoin_address;
    unsigned char *d_target_address;
    int *d_match_found;
    
    cudaMalloc(&d_private_key, 32);
    cudaMalloc(&d_bitcoin_address, RIPEMD160_DIGEST_SIZE);
    cudaMalloc(&d_target_address, RIPEMD160_DIGEST_SIZE);
    cudaMalloc(&d_match_found, sizeof(int));

    // Copiar dados para a GPU
    cudaMemcpy(d_target_address, target_bitcoin_address, RIPEMD160_DIGEST_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_private_key, private_key, 32, cudaMemcpyHostToDevice);

    const int TOTAL_THREADS = blockSize * numBlocks;
    
    // Variáveis para estatísticas
    int match_found_host = 0;
    clock_t start_time = clock();
    uint64_t addresses_processed = 0;
    int display_interval = 5;

    while (!match_found_host) {
        // Resetar flag de match
        cudaMemset(d_match_found, 0, sizeof(int));

        // Lançar kernel
        bitcoin_address_kernel<<<numBlocks, blockSize>>>(
            d_private_key, d_bitcoin_address, d_target_address, 
            d_match_found, d_gTableX, d_gTableY
        );

        // Verificar erros
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Erro no kernel: %s\n", cudaGetErrorString(err));
            break;
        }

        // Sincronizar e verificar resultado
        cudaDeviceSynchronize();
        cudaMemcpy(&match_found_host, d_match_found, sizeof(int), cudaMemcpyDeviceToHost);

        if (match_found_host) {
            // Recuperar a chave privada e o endereço encontrado
            cudaMemcpy(private_key, d_private_key, 32, cudaMemcpyDeviceToHost);
            cudaMemcpy(bitcoin_address, d_bitcoin_address, RIPEMD160_DIGEST_SIZE, cudaMemcpyDeviceToHost);
            
            printf("Chave encontrada: ");
            for (int i = 0; i < 32; i++) {
                printf("%02x", private_key[i]);
            }
            printf("\n");
            break;
        }

        // Incrementar a chave inicial pelo número total de threads
        uint64_t carry = TOTAL_THREADS;
        for (int i = 31; i >= 0 && carry > 0; i--) {
            uint64_t sum = (uint64_t)private_key[i] + carry;
            private_key[i] = sum & 0xFF;
            carry = sum >> 8;
        }

        // Atualizar a chave na GPU
        cudaMemcpy(d_private_key, private_key, 32, cudaMemcpyHostToDevice);

        addresses_processed += TOTAL_THREADS;

        // Estatísticas de performance
        clock_t current_time = clock();
        double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
        if (elapsed_time >= display_interval) {
            double keys_per_second = addresses_processed / elapsed_time;
            printf("\rVelocidade: %s", formatSpeed(keys_per_second));
            fflush(stdout);  // Força a atualização do output
            start_time = clock();
            addresses_processed = 0;
        }
    }

    // Cleanup
    cudaFree(d_private_key);
    cudaFree(d_bitcoin_address);
    cudaFree(d_target_address);
    cudaFree(d_match_found);
    freeGPUTables();

    return 0;
}