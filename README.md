<p align="left">
  <img width="150" alt="logo" src="https://github.com/user-attachments/assets/73297594-3581-4afd-ad0b-39d4bc0e66bf" />
</p>

# HashCracker (CUDA)
## _Parallel SHA-256 Brute Force & Dictionary (salted) Password Cracker_

[🇬🇧 English](README.md) | [🇮🇹 Italiano](README-IT.md)


![alt text](https://img.shields.io/badge/Language-C++%20%7C%20CUDA-green)

![alt text](https://img.shields.io/badge/Platform-NVIDIA%20(CUDA)-blue) 

![alt text](https://img.shields.io/badge/Algorithm-SHA256-purple)

Project for the Accelerated Computing Systems course at the Master's degree in Computer Engineering, University of Bologna.
**Parallel** application for password cracking through Brute Force attack on SHA-256 hashes (including salted) and dictionary attack, 
with performance comparison between Sequential (CPU) and Parallel (GPU/CUDA) implementations.

## 📝 Description
The project implements a **password cracker** that supports different attack modes to reverse SHA-256 hashes. 
The main goal is to demonstrate the speedup achievable by moving from serial execution on CPU to 
massively parallel execution on GPU, analyzing different CUDA memory optimization strategies (Global vs Constant 
Memory) and computational resources.

## ⚙️ Features
- **Incremental Brute Force**: Dynamic password generation given a charset and length range (min-max).
- **Dictionary Attack**: Support for external wordlists.
- **Salt Support**: Handling of salted hashes (Brute Force and dictionary attack).
- **Multi-Platform**: Native CUDA code for **NVIDIA** and (semi) automatic porting script for **AMD** HIP.

## 📂 Project Structure
- `ASSETS/`: contains files used for cracking (charset and dictionary).
- `Sequenziale/`: **Sequential** reference implementation (uses [OpenSSL](https://openssl-library.org/)).
- `CUDA_NAIVE/`: **First GPU implementation** (global memory usage).
- `CUDAv1/`: Memory optimization (Constant Memory usage for charset and target).
- `CUDAv2/`: Kernel optimization (loop unrolling, register optimization for SHA-256).
- `UTILS/`: Support functions (file I/O, argument parsing).
- `SHA256_CUDA/`: **CUDA implementation for SHA256**, based on [mochimodev](https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha256.cu)'s implementation.
- `SHA256_CUDA_OPT/`: **Optimized CUDA implementation for SHA256** (used by CUDAv2).
- `ESTENSIONE/`: contains the implementation of the project extension, i.e., dictionary attack and hash cracking with salt (called from `kernel_estensione.cu`).
- `kernel_[project_version].cu`: file to run the corresponding version.
All CUDA[project_version] versions (executed by their respective kernel files) depend on 
`UTILS` and `SHA256_CUDA` files, except for CUDAv2 and extension which use `SHA256_CUDA_OPT` instead of `SHA256_CUDA`. \
_Note_: the dictionary used is a trimmed version to passwords of length 64 of [rockyou.txt](https://weakpass.com/wordlists/rockyou.txt). Our version is available in the `ASSETS` folder to be unzipped (due to GitHub limits).

## 🛠️ Requirements
- Hardware:
  - **NVIDIA GPU** (Compute Capability 5.0+)
- Software:
  - NVIDIA **CUDA Toolkit** (11.0+)
  - **OpenSSL** (for CPU implementation)
  - **C++ Compiler** (MSVC on Windows, GCC/Clang on Linux)
 
## 🚀 Compilation

### NVIDIA CUDA
Make sure OpenSSL libraries are linked correctly.
```powershell
nvcc -arch=sm_89 -rdc=true -O3 \
    kernel_naive.cu \
    CUDA_NAIVE/*.cu \
    SHA256_CUDA/*.cu \
    UTILS/*.cu UTILS/*.cpp \
    -o naive_cuda \
    -lssl -lcrypto -lcudadevrt -I.
```
_(change file names and dependencies based on the version to compile)_


## 💻 Usage
The program accepts command line parameters for maximum flexibility:
```cmd
./brute_force_cuda [<blockSize>] <hash_target> <min_len> <max_len> <file_charset> [<salt> <dictionary-yes/no> <dictionary_file>]
```
The `blockSize` must always be passed only in parallel GPU scripts (both CUDA and HIP). \
The dictionary (flag and file path) and salt must be passed only in extension scripts. \
Note: in the extension version `max_len` includes the salt length. 

Example:

Search for the password of the hash (corresponding to "qwerty") with length 6, using the standard charset:
```cmd
./brute_force_cuda 256 qwerty 1 6 ASSETS/CharSet.txt az No
```

## 📊 Performance Analysis
Tests were conducted on:
- sequential: Ryzen 9 9900X
- CUDA: NVIDIA RTX 4060 Laptop and partially Google Colab

### Technical Deep Dive: Analysis
The SHA-256 algorithm is heavily **Compute-Bound**. The v2 implementation heavily uses registers to maintain the hash 
state and avoid local/global memory latencies. Although the high number of registers (118) limits the number of active warps 
(low occupancy), the single thread execution speed increases drastically. In this scenario, maximizing IPC 
(Instructions Per Cycle) proved more effective than maximizing parallelism at the latency level (Occupancy).

Furthermore, the use of smaller block sizes (64/128 threads) led to better performance compared to the classic 256, 
thanks to better management of the Tail Effect (wave quantization) and lower scheduling overhead.

The extension implementation has essentially the same performance as v2 (since it uses practically the same code), 
with the addition that for dictionary attack, the time in case of hit is certainly lower than testing all combinations. 

## 👥 Authors
- [Andrea Vitale](https://github.com/WHYUBM)
- [Matteo Fontolan](https://github.com/itsjustwhitee)

## 📜 License
This project is distributed under the AGPL license. See the `LICENSE` file for details.
