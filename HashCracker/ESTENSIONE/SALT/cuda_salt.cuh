#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../UTILS/cuda_utils.cuh"
#include "../../SHA256_CUDA _OPT/sha256_opt.cuh"
#include <stdio.h>
#include "../../SHA256_CUDA/sha256.cuh"
#include "../../SHA256_CUDA/config.h"

//TODO da mettere in un file a parte (.h o simile) per evitare duplicazione
#define MAX_CANDIDATE 10
#define MAX_CHARSET_LENGTH 67 
#define MAX_SALT_LENGTH 4

extern __constant__ BYTE d_target_hash[SHA256_DIGEST_LENGTH];
extern __constant__ char d_charSet[MAX_CHARSET_LENGTH];
extern __constant__ char d_salt[MAX_SALT_LENGTH];
extern __constant__ int d_salt_len;

__global__ void bruteForceKernel_salt(int len, char* d_result, int charSetLen, unsigned long long totalCombinations, bool* d_found);