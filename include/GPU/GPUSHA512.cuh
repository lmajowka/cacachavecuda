#ifndef SHA512
#define SHA512

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cctype>

#define SHA512_DIGEST_SIZE 64
#define SHA512_BLOCK_SIZE 128

#define LL_ROTLEFT(a,b) (((a) << (b)) | ((a) >> (64-(b))))
#define LL_ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (64-(b))))

#define LL_CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define LL_MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define EP0(x) (LL_ROTRIGHT(x,28) ^ LL_ROTRIGHT(x,34) ^ LL_ROTRIGHT(x,39))
#define EP1(x) (LL_ROTRIGHT(x,14) ^ LL_ROTRIGHT(x,18) ^ LL_ROTRIGHT(x,41))
#define SIG0(x) (LL_ROTRIGHT(x,1) ^ LL_ROTRIGHT(x,8) ^ ((x)>>7))
#define SIG1(x) (LL_ROTRIGHT(x,19) ^ LL_ROTRIGHT(x,61) ^ ((x)>>6))

typedef unsigned char BYTE;
typedef uint32_t  WORD;
typedef uint64_t  QWORD;

typedef struct {
    BYTE data[128];
    QWORD datalen;
    QWORD bitlen;
    QWORD bitlen_high;  // Para números maiores que 64 bits
    QWORD state[8];
} SHA512_CTX;

__constant__ QWORD sha512_dev_k[80];
__device__ static const QWORD sha512_host_k[80] = {
  0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538,
  0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe,
  0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2, 0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235,
  0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
  0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5, 0x983e5152ee66dfab,
  0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
  0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed,
  0x53380d139d95b3df, 0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
  0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218,
  0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8, 0x19a4c116b8d2d0c8, 0x1e376c085141ab53,
  0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373,
  0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
  0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c,
  0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6,
  0x113f9804bef90dae, 0x1b710b35131c471b, 0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc,
  0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817
};

__host__ void initSHA512Constants() {
    cudaMemcpyToSymbol(sha512_dev_k, sha512_host_k, sizeof(sha512_host_k));
}

__device__ void sha512_init(SHA512_CTX* ctx);
__device__ void sha512_update(SHA512_CTX* ctx, const BYTE data[], size_t len);
__device__ void sha512_final(SHA512_CTX* ctx, BYTE hash[]);

__device__ void sha512_init(SHA512_CTX* ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->bitlen_high = 0;
    ctx->state[0] = 0x6a09e667f3bcc908ULL;
    ctx->state[1] = 0xbb67ae8584caa73bULL;
    ctx->state[2] = 0x3c6ef372fe94f82bULL;
    ctx->state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->state[4] = 0x510e527fade682d1ULL;
    ctx->state[5] = 0x9b05688c2b3e6c1fULL;
    ctx->state[6] = 0x1f83d9abfb41bd6bULL;
    ctx->state[7] = 0x5be0cd19137e2179ULL;
}

__device__ void sha512_transform(SHA512_CTX* ctx, const BYTE data[])
{
    QWORD a, b, c, d, e, f, g, h, i, t1, t2, m[80];

    // Montagem das palavras de 64 bits
    for (i = 0; i < 16; ++i) {
        int j = i * 8;
        m[i] = ((QWORD)data[j] << 56) | ((QWORD)data[j + 1] << 48) |
               ((QWORD)data[j + 2] << 40) | ((QWORD)data[j + 3] << 32) |
               ((QWORD)data[j + 4] << 24) | ((QWORD)data[j + 5] << 16) |
               ((QWORD)data[j + 6] << 8)  | (QWORD)data[j + 7];
    }

    // Expansão da mensagem
    for (; i < 80; ++i) {
        m[i] = SIG1(m[i-2]) + m[i-7] + SIG0(m[i-15]) + m[i-16];
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 80; ++i) {
        t1 = h + EP1(e) + LL_CH(e,f,g) + sha512_dev_k[i] + m[i];
        t2 = EP0(a) + LL_MAJ(a,b,c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

__device__ void sha512_update(SHA512_CTX* ctx, const BYTE data[], size_t len)
{
    for (QWORD i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == SHA512_BLOCK_SIZE) {
            sha512_transform(ctx, ctx->data);
            // Atualizar o comprimento antes de verificar overflow
            QWORD old_bitlen = ctx->bitlen;
            ctx->bitlen += SHA512_BLOCK_SIZE * 8;
            if (ctx->bitlen < old_bitlen) {
                ctx->bitlen_high++;
            }
            ctx->datalen = 0;
        }
    }
}

__device__ void sha512_final(SHA512_CTX* ctx, BYTE hash[])
{
    QWORD i = ctx->datalen;
    QWORD old_bitlen = ctx->bitlen;
    ctx->bitlen += ctx->datalen * 8;
    // Verifica overflow
    if (ctx->bitlen < old_bitlen)
        ctx->bitlen_high++;

    // Padding
    if (ctx->datalen < 112) {
        ctx->data[i++] = 0x80;
        while (i < 112)
            ctx->data[i++] = 0x00;
    }
    else {
        ctx->data[i++] = 0x80;
        while (i < 128)
            ctx->data[i++] = 0x00;
        sha512_transform(ctx, ctx->data);
        memset(ctx->data, 0, 112);
    }

    // Append length (128 bits)
    // Primeiro os 64 bits mais significativos (bitlen_high)
    ctx->data[112] = (ctx->bitlen_high >> 56) & 0xFF;
    ctx->data[113] = (ctx->bitlen_high >> 48) & 0xFF;
    ctx->data[114] = (ctx->bitlen_high >> 40) & 0xFF;
    ctx->data[115] = (ctx->bitlen_high >> 32) & 0xFF;
    ctx->data[116] = (ctx->bitlen_high >> 24) & 0xFF;
    ctx->data[117] = (ctx->bitlen_high >> 16) & 0xFF;
    ctx->data[118] = (ctx->bitlen_high >> 8) & 0xFF;
    ctx->data[119] = ctx->bitlen_high & 0xFF;
    
    // Depois os 64 bits menos significativos (bitlen)
    ctx->data[120] = (ctx->bitlen >> 56) & 0xFF;
    ctx->data[121] = (ctx->bitlen >> 48) & 0xFF;
    ctx->data[122] = (ctx->bitlen >> 40) & 0xFF;
    ctx->data[123] = (ctx->bitlen >> 32) & 0xFF;
    ctx->data[124] = (ctx->bitlen >> 24) & 0xFF;
    ctx->data[125] = (ctx->bitlen >> 16) & 0xFF;
    ctx->data[126] = (ctx->bitlen >> 8) & 0xFF;
    ctx->data[127] = ctx->bitlen & 0xFF;

    sha512_transform(ctx, ctx->data);

    // Copia o hash final em big-endian
    for (i = 0; i < 8; ++i) {
        hash[i*8]   = (ctx->state[i] >> 56) & 0xFF;
        hash[i*8+1] = (ctx->state[i] >> 48) & 0xFF;
        hash[i*8+2] = (ctx->state[i] >> 40) & 0xFF;
        hash[i*8+3] = (ctx->state[i] >> 32) & 0xFF;
        hash[i*8+4] = (ctx->state[i] >> 24) & 0xFF;
        hash[i*8+5] = (ctx->state[i] >> 16) & 0xFF;
        hash[i*8+6] = (ctx->state[i] >> 8) & 0xFF;
        hash[i*8+7] = ctx->state[i] & 0xFF;
    }
}

__global__ void sha512_kernel(const uint8_t* input, size_t len, uint8_t* output) {
    SHA512_CTX ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, input, len);
    sha512_final(&ctx, output);
}

#endif // SHA512