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
    
    // printf("Bitcoin address (SHA 256): ");
    // for (int i = 0; i < SHA256_DIGEST_SIZE; i++) {
    //     printf("%02x", sha256_hash[i]);
    // }
    // printf("\n");

    // Step 2: Apply RIPEMD-160 to the SHA-256 hash
    ripemd160_gpu(sha256_hash, SHA256_DIGEST_SIZE, ripemd160_hash);

    // Copy the RIPEMD-160 hash to bitcoin_address (this is the Bitcoin address)
    memcpy(bitcoin_address, ripemd160_hash, RIPEMD160_DIGEST_SIZE);
}

// Function to increment the private key
void increment_private_key(unsigned char *private_key) {
    for (int i = 31; i >= 0; i--) {
        if (++private_key[i] != 0) break; // Stop incrementing if there is no overflow
    }
}

int main() {
    // Step 1: Initialize the private key (32 bytes)
    unsigned char private_key[32] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
            0x83, 0x2E, 0xD7, 0x4F, 0x2B, 0x4E, 0x35, 0xEE
    };

    unsigned char public_key[65];  // Uncompressed public key will be 65 bytes
    unsigned char bitcoin_address[RIPEMD160_DIGEST_SIZE];

    // Hardcoded target Bitcoin address
    unsigned char target_bitcoin_address[RIPEMD160_DIGEST_SIZE] = {
            0x20, 0xd4, 0x5a, 0x6a, 0x76, 0x25, 0x35, 0x70, 
            0x0c, 0xe9, 0xe0, 0xb2, 0x16, 0xe3, 0x19, 0x94, 
            0x33, 0x5d, 0xb8, 0xa5  
    };

    unsigned char* d_public_key;
    unsigned char* d_bitcoin_address;

    // Allocate memory on the GPU for public key and Bitcoin address
    cudaMalloc(&d_public_key, 65);
    cudaMalloc(&d_bitcoin_address, RIPEMD160_DIGEST_SIZE);

    // Variables to track time and address rate
    clock_t start_time = clock();
    int iterations = 0;
    int addresses_processed = 0;
    int display_interval = 5;  // Display every 5 seconds

    while (1) {
        // Step 2: Generate the public key from the private key using secp256k1
        generate_public_key(private_key, public_key);

        // Copy public key to GPU
        cudaMemcpy(d_public_key, public_key, 65, cudaMemcpyHostToDevice);

        // Step 3: Launch kernel to compute Bitcoin address
        bitcoin_address_kernel<<<1, 1>>>(d_public_key, d_bitcoin_address);

        // Copy Bitcoin address result back to host
        cudaMemcpy(bitcoin_address, d_bitcoin_address, RIPEMD160_DIGEST_SIZE, cudaMemcpyDeviceToHost);

        // Step 4: Compare Bitcoin address to the target address
        if (memcmp(bitcoin_address, target_bitcoin_address, RIPEMD160_DIGEST_SIZE) == 0) {
            printf("Matching Bitcoin address found!\n");
            printf("Private Key: ");
            for (int i = 0; i < 32; i++) {
                printf("%02x", private_key[i]);
            }
            printf("\n");
            break;
        }

        // Increment the private key
        increment_private_key(private_key);

        iterations++;
        addresses_processed++;

        // Check time every iteration
        clock_t current_time = clock();
        double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;

        // Display the address generation rate every 5 seconds
        if (elapsed_time >= display_interval) {
            double addresses_per_second = addresses_processed / elapsed_time;
            printf("Addresses per second: %.2f\n", addresses_per_second);

            // Reset tracking variables for the next interval
            start_time = clock();
            addresses_processed = 0;
        }
    }

    // Free GPU memory
    cudaFree(d_public_key);
    cudaFree(d_bitcoin_address);

    return 0;
}