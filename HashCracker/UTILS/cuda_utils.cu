#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ void idxToString(unsigned long long idx, char* result, int len, char* charset, int charsetLen) {
    // Si riempie la stringa partendo dall'ultimo carattere (destra verso sinistra)
    for (int i = len - 1; i >= 0; i--) {
        // Il resto della divisione indica il carattere (rispetto al charset) 
        int charIndex = idx % charsetLen;

        result[i] = charset[charIndex];

        // Si passa alla posizione successiva (a sinistra)
        idx /= charsetLen;
    }
}

__device__ bool check_hash_match(const unsigned char* hash1, const unsigned char* hash2, int hashLen) {
    // Unroll del loop per massimizzare le prestazioni (opzionale, ma aiuta)
#pragma unroll
    for (int i = 0; i < hashLen; i++) {
        if (hash1[i] != hash2[i]) {
            return false; // Appena trovo un byte diverso, esco
        }
    }
    return true;
}