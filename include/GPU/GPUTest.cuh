#ifndef GPUTEST_CUH
#define GPUTEST_CUH

#include "GPUSHA512.cuh"
#include "GPUPBKDF2.cuh"
#include "GPUUtils.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace GPUTest {

void testSHA512() {
    printf("\n=== Teste SHA-512 ===\n");
    
    // Test vector from RFC 4231
    const char* input = "abc";
    const char* expected = 
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
        "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f";
    
    // Alocar memória
    uint8_t* d_input;
    uint8_t* d_output;
    uint8_t h_output[SHA512_DIGEST_SIZE];
    
    cudaMalloc(&d_input, 3);
    cudaMalloc(&d_output, SHA512_DIGEST_SIZE);
    cudaMemcpy(d_input, input, 3, cudaMemcpyHostToDevice);
    
    // Executar teste
    sha512_kernel<<<1, 1>>>(d_input, 3, d_output);
    
    // Copiar resultado
    cudaMemcpy(h_output, d_output, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost);
    
    // Converter para hex e comparar
    std::string result = bytesToHex(h_output, SHA512_DIGEST_SIZE);
    
    printf("Input: \"%s\"\n", input);
    printf("Esperado : %s\n", expected);
    printf("Resultado: %s\n", result.c_str());
    printf("Teste %s\n", result == expected ? "PASSOU" : "FALHOU");
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// HMAC-SHA-512 test kernel - Implementação de referência
__global__ void test_hmac_sha512_ref_kernel(uint8_t* output, uint8_t* debug_ipad, uint8_t* debug_opad, uint8_t* debug_inner) {
    // Test vector from RFC 4231 Test Case 1
    const uint8_t key[20] = {
        0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 
        0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 
        0x0b, 0x0b, 0x0b, 0x0b
    };
    const uint8_t data[8] = {0x48, 0x69, 0x20, 0x54, 0x68, 0x65, 0x72, 0x65}; // "Hi There"
    
    // Debug: mostrar a chave
    printf("Debug Reference Implementation:\n");
    printf("Key inicial: ");
    for(size_t i = 0; i < 20; i++) printf("%02x", key[i]);
    printf("\n");
    
    // 1. Preparar a chave
    uint8_t k[HMAC_BLOCK_SIZE] = {0};
    memcpy(k, key, 20);
    
    // Debug: mostrar a chave após o processamento
    printf("Key após processamento: ");
    for(size_t i = 0; i < HMAC_BLOCK_SIZE; i++) printf("%02x", k[i]);
    printf("\n");
    
    // 2. Criar k_ipad e k_opad
    uint8_t k_ipad[HMAC_BLOCK_SIZE], k_opad[HMAC_BLOCK_SIZE];
    
    // Debug: mostrar k_ipad e k_opad antes do XOR
    printf("k_ipad antes do XOR: ");
    for(int i = 0; i < 16; i++) printf("%02x ", k_ipad[i]);
    printf("\nk_opad antes do XOR: ");
    for(int i = 0; i < 16; i++) printf("%02x ", k_opad[i]);
    printf("\n");
    
    // Preencher com 0x36 e 0x5c
    memset(k_ipad, 0x36, HMAC_BLOCK_SIZE);
    memset(k_opad, 0x5c, HMAC_BLOCK_SIZE);
    
    // Debug: mostrar k_ipad e k_opad após o preenchimento
    printf("k_ipad após preenchimento: ");
    for(int i = 0; i < 16; i++) printf("%02x ", k_ipad[i]);
    printf("\nk_opad após preenchimento: ");
    for(int i = 0; i < 16; i++) printf("%02x ", k_opad[i]);
    printf("\n");
    
    // XOR com a chave
    for (int i = 0; i < HMAC_BLOCK_SIZE; i++) {
        k_ipad[i] ^= k[i];
        k_opad[i] ^= k[i];
    }
    
    // Debug: mostrar k_ipad e k_opad após o XOR
    printf("k_ipad após XOR: ");
    for(int i = 0; i < 16; i++) printf("%02x ", k_ipad[i]);
    printf("\nk_opad após XOR: ");
    for(int i = 0; i < 16; i++) printf("%02x ", k_opad[i]);
    printf("\n");
    
    // 3. Calcular hash interno
    SHA512_CTX ctx;
    uint8_t inner_hash[SHA512_DIGEST_SIZE];
    
    sha512_init(&ctx);
    sha512_update(&ctx, k_ipad, HMAC_BLOCK_SIZE);
    sha512_update(&ctx, data, 8);
    sha512_final(&ctx, inner_hash);
    
    // Debug: mostrar hash interno
    printf("Hash interno: ");
    for(int i = 0; i < SHA512_DIGEST_SIZE; i++) printf("%02x", inner_hash[i]);
    printf("\n");
    
    // 4. Calcular hash externo
    sha512_init(&ctx);
    sha512_update(&ctx, k_opad, HMAC_BLOCK_SIZE);
    sha512_update(&ctx, inner_hash, SHA512_DIGEST_SIZE);
    sha512_final(&ctx, output);
    
    // Copiar valores para debug
    memcpy(debug_ipad, k_ipad, HMAC_BLOCK_SIZE);
    memcpy(debug_opad, k_opad, HMAC_BLOCK_SIZE);
    memcpy(debug_inner, inner_hash, SHA512_DIGEST_SIZE);
}

// HMAC-SHA-512 test kernel - Nossa implementação
__global__ void test_hmac_sha512_kernel(uint8_t* output, uint8_t* debug_ipad, uint8_t* debug_opad, uint8_t* debug_inner) {
    // Test vector from RFC 4231 Test Case 1
    const uint8_t key[20] = {
        0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 
        0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 
        0x0b, 0x0b, 0x0b, 0x0b
    };
    const uint8_t data[8] = {0x48, 0x69, 0x20, 0x54, 0x68, 0x65, 0x72, 0x65}; // "Hi There"
    
    // Calcular HMAC usando nossa implementação
    HMAC_CTX ctx;
    ctx.debug = true;
    
    // 1. Inicializar HMAC
    hmac_sha512_init(&ctx, key, 20);
    
    // 2. Atualizar com os dados
    hmac_sha512_update(&ctx, data, 8);
    
    // 3. Finalizar
    hmac_sha512_final(&ctx, output);
    
    // Copiar valores para debug
    memcpy(debug_ipad, ctx.k_ipad, HMAC_BLOCK_SIZE);
    memcpy(debug_opad, ctx.k_opad, HMAC_BLOCK_SIZE);
}

void testHMAC() {
    uint8_t output[SHA512_DIGEST_SIZE];
    uint8_t output_ref[SHA512_DIGEST_SIZE];
    uint8_t debug_ipad[HMAC_BLOCK_SIZE];
    uint8_t debug_opad[HMAC_BLOCK_SIZE];
    uint8_t debug_inner[SHA512_DIGEST_SIZE];
    
    uint8_t* d_output;
    uint8_t* d_output_ref;
    uint8_t* d_debug_ipad;
    uint8_t* d_debug_opad;
    uint8_t* d_debug_inner;
    
    // Alocar memória na GPU
    cudaMalloc(&d_output, SHA512_DIGEST_SIZE);
    cudaMalloc(&d_output_ref, SHA512_DIGEST_SIZE);
    cudaMalloc(&d_debug_ipad, HMAC_BLOCK_SIZE);
    cudaMalloc(&d_debug_opad, HMAC_BLOCK_SIZE);
    cudaMalloc(&d_debug_inner, SHA512_DIGEST_SIZE);
    
    // Executar teste de referência
    test_hmac_sha512_ref_kernel<<<1, 1>>>(d_output_ref, d_debug_ipad, d_debug_opad, d_debug_inner);
    
    // Executar nossa implementação
    test_hmac_sha512_kernel<<<1, 1>>>(d_output, d_debug_ipad, d_debug_opad, d_debug_inner);
    
    // Copiar resultados de volta para CPU
    cudaMemcpy(output, d_output, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_ref, d_output_ref, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_ipad, d_debug_ipad, HMAC_BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_opad, d_debug_opad, HMAC_BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_inner, d_debug_inner, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost);
    
    // Valor esperado do RFC 4231 Test Case 1
    const uint8_t expected[SHA512_DIGEST_SIZE] = {
        0x87, 0xaa, 0x7c, 0xde, 0xa5, 0xef, 0x61, 0x9d,
        0x4f, 0xf0, 0xb4, 0x24, 0x1a, 0x1d, 0x6c, 0xb0,
        0x23, 0x79, 0xf4, 0xe2, 0xce, 0x4e, 0xc2, 0x78,
        0x7a, 0xd0, 0xb3, 0x05, 0x45, 0xe1, 0x7c, 0xde,
        0xda, 0xa8, 0x33, 0xb7, 0xd6, 0xb8, 0xa7, 0x02,
        0x03, 0x8b, 0x27, 0x4e, 0xae, 0xa3, 0xf4, 0xe4,
        0xbe, 0x9d, 0x91, 0x4e, 0xeb, 0x61, 0xf1, 0x70,
        0x2e, 0x69, 0x6c, 0x20, 0x3a, 0x12, 0x68, 0x54
    };
    
    // Debug
    printf("\n=== Nossa Implementação ===\n");
    printf("Input: \"Hi There\"\n");
    printf("Key: 0b0b0b... (20 bytes)\n");
    printf("Esperado : ");
    for(int i = 0; i < SHA512_DIGEST_SIZE; i++) printf("%02x", expected[i]);
    printf("\nResultado: ");
    for(int i = 0; i < SHA512_DIGEST_SIZE; i++) printf("%02x", output[i]);
    printf("\nTeste %s\n", memcmp(output, expected, SHA512_DIGEST_SIZE) == 0 ? "PASSOU" : "FALHOU");
    
    printf("\n=== Implementação de Referência ===\n");
    printf("Resultado: ");
    for(int i = 0; i < SHA512_DIGEST_SIZE; i++) printf("%02x", output_ref[i]);
    printf("\nTeste %s\n", memcmp(output_ref, expected, SHA512_DIGEST_SIZE) == 0 ? "PASSOU" : "FALHOU");
    
    // Debug HMAC
    printf("\nDebug HMAC:\n");
    printf("Key bytes (20): 0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b\n");
    printf("Data bytes (8): 48 69 20 54 68 65 72 65\n\n");
    
    printf("K_ipad primeiros 32 bytes:\n");
    for(int i = 0; i < 32; i++) {
        printf("%02x ", debug_ipad[i]);
        if((i+1) % 16 == 0) printf("\n");
    }
    
    printf("\nK_opad primeiros 32 bytes:\n");
    for(int i = 0; i < 32; i++) {
        printf("%02x ", debug_opad[i]);
        if((i+1) % 16 == 0) printf("\n");
    }
    
    printf("\nHash interno:\n");
    for(int i = 0; i < SHA512_DIGEST_SIZE; i++) printf("%02x", debug_inner[i]);
    printf("\n");
    
    // Liberar memória
    cudaFree(d_output);
    cudaFree(d_output_ref);
    cudaFree(d_debug_ipad);
    cudaFree(d_debug_opad);
    cudaFree(d_debug_inner);
}

} // namespace GPUTest

#endif // GPUTEST_CUH
