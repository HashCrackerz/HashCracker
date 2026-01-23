/*
 * sha256.cuh
 */
#pragma once
#include "config.h"

 // Questo blocco assicura che C++ (kernel.cu) legga correttamente le funzioni C
#ifdef __cplusplus
extern "C" {
#endif

    void mcm_cuda_sha256_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch);

    // Aggiungiamo qui la dichiarazione della tua nuova funzione
    int launchBruteForceCUDA(unsigned char* target_hash, char* charset, int charset_len, int min_len, int max_len, char* result_buffer);

#ifdef __cplusplus
}
#endif