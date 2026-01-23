/*
 * sha256.cu Implementation of SHA256 Hashing    
 *
 * Date: 12 June 2019
 * Revision: 1
 * *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

 
/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
extern "C" {
#include "sha256.cuh"
}
/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/

typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} CUDA_SHA256_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
__constant__ WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DEFINITIONS ***********************/
__device__  __forceinline__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
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

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			cuda_sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[])
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		cuda_sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	cuda_sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

__global__ void kernel_sha256_hash(BYTE* indata, WORD inlen, BYTE* outdata, WORD n_batch)
{
	WORD thread = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread >= n_batch)
	{
		return;
	}
	BYTE* in = indata  + thread * inlen;
	BYTE* out = outdata  + thread * SHA256_BLOCK_SIZE;
	CUDA_SHA256_CTX ctx;
	cuda_sha256_init(&ctx);
	cuda_sha256_update(&ctx, in, inlen);
	cuda_sha256_final(&ctx, out);
}

extern "C"
{
void mcm_cuda_sha256_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch)
{
	BYTE *cuda_indata;
	BYTE *cuda_outdata;
	cudaMalloc(&cuda_indata, inlen * n_batch);
	cudaMalloc(&cuda_outdata, SHA256_BLOCK_SIZE * n_batch);
	cudaMemcpy(cuda_indata, in, inlen * n_batch, cudaMemcpyHostToDevice);

	WORD thread = 256;
	WORD block = (n_batch + thread - 1) / thread;

	kernel_sha256_hash << < block, thread >> > (cuda_indata, inlen, cuda_outdata, n_batch);
	cudaMemcpy(out, cuda_outdata, SHA256_BLOCK_SIZE * n_batch, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda sha256 hash: %s \n", cudaGetErrorString(error));
	}
	cudaFree(cuda_indata);
	cudaFree(cuda_outdata);
}
}

// Kernel CUDA che genera password e controlla l'hash
__global__ void kernel_brute_force(unsigned char* d_target_hash, char* d_charset, int charset_len, int pass_len, unsigned long long offset, volatile int* d_found, char* d_result) {
	// Calcolo indice globale del thread
	unsigned long long idx = offset + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

	// Se abbiamo già trovato la password, i thread successivi possono terminare subito (ottimizzazione)
	if (*d_found) return;

	// Ricostruiamo la password basandoci sull'indice (conversione da base-10 a base-N)
	char attempt[20]; // Buffer locale per la password (assicurarsi sia > max_test_len)
	unsigned long long temp = idx;

	for (int i = 0; i < pass_len; i++) {
		attempt[i] = d_charset[temp % charset_len];
		temp /= charset_len;
	}

	// Hash della password generata
	CUDA_SHA256_CTX ctx;
	cuda_sha256_init(&ctx);
	cuda_sha256_update(&ctx, (BYTE*)attempt, pass_len);

	BYTE hash[SHA256_BLOCK_SIZE];
	cuda_sha256_final(&ctx, hash);

	// Confronto con l'hash target
	bool match = true;
	for (int i = 0; i < SHA256_BLOCK_SIZE; i++) {
		if (hash[i] != d_target_hash[i]) {
			match = false;
			break;
		}
	}

	// Se trovata, salviamo il risultato
	if (match) {
		*d_found = 1;
		// Copiamo la password nel buffer di uscita globale
		for (int i = 0; i < pass_len; i++) {
			d_result[i] = attempt[i];
		}
		d_result[pass_len] = '\0'; // Terminatore stringa
	}
}

// Funzione Host (wrapper)
extern "C" int launchBruteForceCUDA(unsigned char* target_hash, char* charset, int charset_len, int min_len, int max_len, char* result_buffer) {

	unsigned char* d_target;
	char* d_charset;
	char* d_result;
	int* d_found;
	int h_found = 0;

	// Allocazione memoria Device
	cudaMalloc(&d_target, SHA256_BLOCK_SIZE);
	cudaMalloc(&d_charset, charset_len);
	cudaMalloc(&d_result, 100); // Buffer per la password trovata
	cudaMalloc(&d_found, sizeof(int));

	// Copia dati Host -> Device
	cudaMemcpy(d_target, target_hash, SHA256_BLOCK_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_charset, charset, charset_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);

	// Iteriamo per le diverse lunghezze (come fa la versione sequenziale)
	for (int len = min_len; len <= max_len; len++) {

		// Calcolo combinazioni totali: charset_len^len
		unsigned long long total_combinations = 1;
		for (int k = 0; k < len; k++) total_combinations *= charset_len;

		// Configurazione Kernel
		int blockSize = 256;
		// Calcoliamo i blocchi necessari. Attenzione: per len=5 i numeri sono grandi (~1.3 miliardi)
		unsigned long long numBlocks = (total_combinations + blockSize - 1) / blockSize;

		printf("Tentativo GPU lunghezza %d (Combinazioni: %llu)...\n", len, total_combinations);

		// Nota: Per numeri di blocchi molto alti, potresti dover spezzare il lancio in più kernel 
		// per evitare il timeout del driver (TDR), ma per questo esempio lanciamo tutto insieme.
		// Se numBlocks > 2147483647 (max int), va gestito un loop aggiuntivo.

		kernel_brute_force << < (unsigned int)numBlocks, blockSize >> > (d_target, d_charset, charset_len, len, 0, d_found, d_result);

		cudaDeviceSynchronize();

		// Controllo se trovata
		cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
		if (h_found) {
			cudaMemcpy(result_buffer, d_result, 100, cudaMemcpyDeviceToHost);
			break; // Usciamo dal loop delle lunghezze
		}
	}

	// Pulizia
	cudaFree(d_target);
	cudaFree(d_charset);
	cudaFree(d_result);
	cudaFree(d_found);

	return h_found;
}