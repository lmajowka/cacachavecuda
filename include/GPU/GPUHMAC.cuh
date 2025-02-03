#ifndef GPU_HMAC_CUH
#define GPU_HMAC_CUH

#include "GPUSHA512.cuh"

#define BLOCK_SIZE 128  // SHA-512 block size in bytes

__global__ void hmac_sha512_kernel(const char* key, size_t key_len, const BYTE* data, size_t data_len, BYTE* output) {

    // Alinhar buffers para melhor performance e consistência
    __align__(16) BYTE k[BLOCK_SIZE] = {0};
    __align__(16) BYTE k_ipad[BLOCK_SIZE];
    __align__(16) BYTE k_opad[BLOCK_SIZE];
    __align__(16) BYTE inner_hash[SHA512_DIGEST_SIZE];
    __align__(16) BYTE initial_output[SHA512_DIGEST_SIZE]; 

    // Processar a chave
    if (key_len > BLOCK_SIZE) {
        SHA512_CTX ctx;
        sha512_init(&ctx);
        sha512_update(&ctx, (const BYTE*)key, key_len);
        sha512_final(&ctx, k);
        key_len = SHA512_DIGEST_SIZE;
    } else {
        memcpy(k, key, key_len);
        // Preencher o resto com zeros (já está zerado pela inicialização)
    }

    // Criar k_ipad e k_opad com o bloco inteiro
    for (int i = 0; i < BLOCK_SIZE; i++) {
        k_ipad[i] = k[i] ^ 0x36;
        k_opad[i] = k[i] ^ 0x5c;
    }

    // Hash interno
    SHA512_CTX ctx_in;
    sha512_init(&ctx_in);
    sha512_update(&ctx_in, k_ipad, BLOCK_SIZE);  // Usar bloco inteiro
    sha512_update(&ctx_in, data, data_len);
    sha512_final(&ctx_in, inner_hash);

    // Hash externo
    SHA512_CTX ctx_out;
    sha512_init(&ctx_out);
    sha512_update(&ctx_out, k_opad, BLOCK_SIZE);  // Usar bloco inteiro
    sha512_update(&ctx_out, inner_hash, SHA512_DIGEST_SIZE);
    sha512_final(&ctx_out, output);
    memcpy(initial_output, output, SHA512_DIGEST_SIZE);

}

#endif // GPU_HMAC_CUH