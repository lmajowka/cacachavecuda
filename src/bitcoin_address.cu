#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <secp256k1.h>
#include "sha256_cuda.h"
#include "ripemd160_cuda.h"

// Function to perform secp256k1 public key generation
void generate_public_key(unsigned char* private_key, unsigned char* public_key) {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    // Create the public key using secp256k1
    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, private_key)) {
        printf("Error: Public key generation failed.\n");
        secp256k1_context_destroy(ctx);
        return;
    }

    // Serialize the public key in uncompressed format (65 bytes)
    size_t pubkey_len = 33;
    secp256k1_ec_pubkey_serialize(ctx, public_key, &pubkey_len, &pubkey, SECP256K1_EC_COMPRESSED);

    secp256k1_context_destroy(ctx);
}


// Kernel to generate the Bitcoin address using SHA-256 and RIPEMD-160 on the GPU
__global__ void bitcoin_address_kernel(const unsigned char* public_key, unsigned char* bitcoin_address) {
    unsigned char sha256_hash[SHA256_DIGEST_SIZE];
    unsigned char ripemd160_hash[RIPEMD160_DIGEST_SIZE];

    // Step 1: Apply SHA-256 to the public key
    sha256_gpu(public_key, 33, sha256_hash);
    
    printf("Bitcoin address (SHA 256): ");
    for (int i = 0; i < SHA256_DIGEST_SIZE; i++) {
        printf("%02x", sha256_hash[i]);
    }
    printf("\n");

    // Step 2: Apply RIPEMD-160 to the SHA-256 hash
    ripemd160_gpu(sha256_hash, SHA256_DIGEST_SIZE, ripemd160_hash);

    // Copy the RIPEMD-160 hash to bitcoin_address (this is the Bitcoin address)
    memcpy(bitcoin_address, ripemd160_hash, RIPEMD160_DIGEST_SIZE);
}

int main() {
    // Step 1: Generate a random private key (32 bytes)
    unsigned char private_key[32] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
            0x83, 0x2E, 0xD7, 0x4F, 0x2B, 0x5E, 0x35, 0xEE
    };

    unsigned char public_key[65];  // Uncompressed public key will be 65 bytes

    // Step 2: Generate the public key from the private key using secp256k1
    generate_public_key(private_key, public_key);

    // printf("Public Key: ");
    // for (int i = 0; i < 65; i++) {
    //     printf("%02x", public_key[i]);
    // }
    // printf("\n");

    // Step 3: Allocate memory on the GPU for the public key and Bitcoin address
    unsigned char* d_public_key;
    unsigned char* d_bitcoin_address;
    unsigned char bitcoin_address[RIPEMD160_DIGEST_SIZE];

    cudaMalloc(&d_public_key, 65);
    cudaMalloc(&d_bitcoin_address, RIPEMD160_DIGEST_SIZE);

    // Copy public key to GPU
    cudaMemcpy(d_public_key, public_key, 65, cudaMemcpyHostToDevice);

    // Step 4: Launch kernel to compute Bitcoin address
    bitcoin_address_kernel<<<1, 1>>>(d_public_key, d_bitcoin_address);

    // Copy Bitcoin address result back to host
    cudaMemcpy(bitcoin_address, d_bitcoin_address, RIPEMD160_DIGEST_SIZE, cudaMemcpyDeviceToHost);

    // Step 5: Print the Bitcoin address (RIPEMD-160 hash of SHA-256 of the public key)
    printf("Bitcoin address (RIPEMD-160): ");
    for (int i = 0; i < RIPEMD160_DIGEST_SIZE; i++) {
        printf("%02x", bitcoin_address[i]);
    }
    printf("\n");

    // Free GPU memory
    cudaFree(d_public_key);
    cudaFree(d_bitcoin_address);

    return 0;
}
