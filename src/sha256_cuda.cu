#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define SHA256_BLOCK_SIZE 64  // 512 bits
#define SHA256_DIGEST_SIZE 32  // 256 bits

// SHA-256 constants (first 32 bits of the fractional parts of the cube roots of the first 64 primes)
__device__ const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Device function for bitwise rotation
__device__ __inline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 transformation step
__device__ void sha256_transform(uint32_t state[8], const unsigned char block[SHA256_BLOCK_SIZE]) {
    uint32_t a, b, c, d, e, f, g, h, i, t1, t2, m[64];

    // Prepare message schedule
    for (i = 0; i < 16; ++i) {
        m[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) | (block[i * 4 + 2] << 8) | block[i * 4 + 3];
    }
    for (; i < 64; ++i) {
        m[i] = (rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10)) + m[i - 7] +
               (rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3)) + m[i - 16];
    }

    // Initialize working variables to current state
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // Compression function main loop
    for (i = 0; i < 64; ++i) {
        t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + m[i];
        t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add the compressed chunk to the current hash value
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// SHA-256 padding and preprocessing
__device__ void sha256_init(uint32_t state[8]) {
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
}

// SHA-256 GPU implementation
__device__ void sha256_gpu(const unsigned char* data, size_t len, unsigned char* hash) {
    uint32_t state[8];
    unsigned char block[SHA256_BLOCK_SIZE];
    size_t i, bit_len = len * 8;

    sha256_init(state);

    // Process each 512-bit chunk
    while (len >= SHA256_BLOCK_SIZE) {
        memcpy(block, data, SHA256_BLOCK_SIZE);
        sha256_transform(state, block);
        data += SHA256_BLOCK_SIZE;
        len -= SHA256_BLOCK_SIZE;
    }

    // Padding
    memcpy(block, data, len);
    block[len] = 0x80;  // Append '1' bit
    if (len < SHA256_BLOCK_SIZE - 8) {
        memset(block + len + 1, 0, SHA256_BLOCK_SIZE - len - 9);
    } else {
        memset(block + len + 1, 0, SHA256_BLOCK_SIZE - len - 1);
        sha256_transform(state, block);
        memset(block, 0, SHA256_BLOCK_SIZE - 8);
    }

    // Append length in bits (big-endian)
    for (i = 0; i < 8; ++i) {
        block[SHA256_BLOCK_SIZE - 1 - i] = bit_len >> (i * 8);
    }
    sha256_transform(state, block);

    // Convert state to hash (big-endian)
    for (i = 0; i < 8; ++i) {
        hash[i * 4] = (state[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
        hash[i * 4 + 3] = state[i] & 0xff;
    }
}

// Test kernel to compute SHA-256 on GPU
__global__ void sha256_test_kernel(const unsigned char* data, size_t len, unsigned char* hash) {
    sha256_gpu(data, len, hash);
}

    