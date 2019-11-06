#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// __sync_threads()
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "crt/device_functions.h"


// Mersenne Twister macros and parameters
#define hiBit(u)      ((u) & 0x80000000U)  /* mask all but highest   bit of u */
#define loBit(u)      ((u) & 0x00000001U)  /* mask all but lowest    bit of u */
#define loBits(u)     ((u) & 0x7FFFFFFFU)  /* mask     the highest   bit of u */
#define mixBits(u, v) (hiBit(u)|loBits(v)) /* move hi bit of u to hi bit of v */

#define N             (624)                /* length of state vector */
#define M             (397)                /* a period parameter */

#define PHP_MT_RAND_MAX ((long) (0x7FFFFFFF)) /* (1<<31) - 1 */

// Will define if the PHP generator will be used, change at compile time
#define PHPMtRand 0


typedef struct {
    uint32_t state[N];
    uint32_t left;
    uint32_t* next;
} MTState;

#define php_mt_rand_range php_mt_rand_range32


__device__ static inline uint32_t
php_twist(uint32_t m, uint32_t u, uint32_t v)
{
    return (m ^ (mixBits(u, v) >> 1) ^ ((uint32_t)(-(uint32_t)(loBit(u))) & 0x9908b0dfU));
}

__device__ static inline uint32_t
mt_twist(uint32_t m, uint32_t u, uint32_t v)
{
    return (m ^ (mixBits(u, v) >> 1) ^ ((uint32_t)(-(uint32_t)(loBit(v))) & 0x9908b0dfU));
}



__device__ void mtInitialize(uint32_t seed, MTState* mtInfo)
{
    /* Initialize generator state with seed
       See Knuth TAOCP Vol 2, 3rd Ed, p.106 for multiplier.
       In previous versions, most significant bits (MSBs) of the seed affect
       only MSBs of the state array.  Modified 9 Jan 2002 by Makoto Matsumoto. */

    register uint32_t* s = mtInfo->state;
    register uint32_t* r = mtInfo->state;
    register int i = 1;

    *s++ = seed & 0xffffffffU;
    for (; i < N; ++i) {
        *s++ = (1812433253U * (*r ^ (*r >> 30)) + i) & 0xffffffffU;
        r++;
    }
}


__device__ void mtReload(MTState* mtInfo)
{
    /* Generate N new values in state
       Made clearer and faster by Matthew Bellew (matthew.bellew@home.com) */

    register uint32_t* p = mtInfo->state;
    register int i;
    register uint32_t(*twist)(uint32_t, uint32_t, uint32_t) =
        (PHPMtRand) ? php_twist : mt_twist;


    for (i = N - M; i--; ++p)
        *p = twist(p[M], p[0], p[1]);
    for (i = M; --i; ++p)
        *p = twist(p[M - N], p[0], p[1]);
    *p = twist(p[M - N], p[0], mtInfo->state[0]);
    mtInfo->left = N;
    mtInfo->next = mtInfo->state;
}


__device__ void php_mt_srand(uint32_t seed, MTState* mtInfo) {
    mtInitialize(seed, mtInfo);
    mtInfo->left = 0;

    return;
}


__device__ uint32_t mt_rand(MTState* mtInfo)
{
    /* Pull a 32-bit integer from the generator state
       Every other access function simply transforms the numbers extracted here */

    register uint32_t s1;

    if (mtInfo->left == 0)
        mtReload(mtInfo);

    --(mtInfo->left);
    s1 = *mtInfo->next++;

    s1 ^= (s1 >> 11);
    s1 ^= (s1 << 7) & 0x9d2c5680U;
    s1 ^= (s1 << 15) & 0xefc60000U;
    s1 ^= (s1 >> 18);
    return s1;
}


__device__ uint32_t php_mt_rand(MTState* mtInfo)
{
    return mt_rand(mtInfo) >> 1;
}


#define UNEXPECTED(...) __VA_ARGS__
__device__ uint32_t php_mt_rand_range_32(MTState* mtInfo, uint32_t min, uint32_t max)
{
    uint32_t result = mt_rand(mtInfo);
    uint32_t umax = max - min;

    /* Special case where no modulus is required */
    if (UNEXPECTED(umax == UINT32_MAX)) {
        return result;
    }

    /* Increment the max so the range is inclusive of max */
    umax++;

    /* Powers of two are not biased */
    if ((umax & (umax - 1)) == 0) {
        return result & (umax - 1);
    }

    /* Ceiling under which UINT32_MAX % max == 0 */
    uint32_t limit = UINT32_MAX - (UINT32_MAX % umax) - 1;

    /* Discard numbers over the limit to avoid modulo bias */
    while (UNEXPECTED(result > limit)) {
        result = php_mt_rand(mtInfo);
    }

    return result % umax + min;
}

#define RAND_RANGE_BADSCALING(__n, __min, __max, __tmax) \
	(__n) = (__min) + (long) ((double) ( (double) (__max) - (__min) + 1.0) * ((__n) / ((__tmax) + 1.0)))

__device__ uint32_t php_mt_rand_range_php(MTState* mtInfo, uint32_t min, uint32_t max)
{
    int64_t n = (int64_t)php_mt_rand(mtInfo) >> 1;
    RAND_RANGE_BADSCALING(n, min, max, PHP_MT_RAND_MAX);
    return n;
}

// 用来调试发现最大的<<<gridDim, blockDim>>>
// __device__ uint64_t biggest_start;
// __device__ unsigned int bid, tid;

__global__ void crack(uint32_t* global_data, uint32_t* seq, int seqlen){
    uint32_t step = gridDim.x * blockDim.x;
    uint64_t start = blockDim.x * blockIdx.x + threadIdx.x;
    // if (start > biggest_start) {
    //     biggest_start = start; bid = blockIdx.x; tid = threadIdx.x;
    // }
    for(uint64_t i=start; i<0xffffffffUL; i+=step){
        if (global_data[0]) {
            break;
        };
        if (i%0x1000000 == 0) {
                printf("Progress: %08X/FFFFFFFF\n", (uint32_t)i);
                // printf("Biggest Start: %p (%x->%x), step: %x\n", biggest_start, bid, tid, step);
        }
        MTState info;
        php_mt_srand(i, &info);
        int j = 0;
        uint32_t n = 0;
        uint32_t t = 0;
        do{
            if (j >= seqlen) {
                global_data[0] = true;
                printf("\nFound Seed: 0x%08X\n", i);
                __threadfence();
                return;
            }
            n = php_mt_rand(&info);
            t = seq[j];
            j++;
        } while (n == t);
    }
    return;
}

int main(int argc, char** argv){
    if(argc <= 2){ printf("mt_rand_crack_cuda.exe  RAND_SEQ..."); return 0;}
    int seqlen = argc-1;
    if(seqlen <=0){
        printf("missing rand sequence\n");
        return 0;
    }
    size_t size = seqlen * sizeof(uint32_t);
    uint32_t* seq = (uint32_t*)malloc(size);
    for (int i = 0; i < seqlen; i++) {
        seq[i] = atoi(argv[i + 1]);
    }
    uint32_t* d_seq = 0;
    if (cudaMalloc(&d_seq, size) != cudaSuccess) {
        return 0;
    }
    uint32_t * global_data;
    if (cudaMalloc(&global_data, sizeof(global_data)*4) != cudaSuccess) {
        return 0;
    }
    cudaMemcpy(d_seq, seq, size, cudaMemcpyHostToDevice);
    free(seq);
    time_t start_time = time(NULL);
    printf("Running\n");
    // read prop
    // cudaDeviceProp prop = { 0 };
    // cudaGetDeviceProperties(&prop, 0);

    // GTX-1080Ti
    // 不知道为什么是0x300而非0x400。不管了。
    crack <<<0x20, 0x300>>> (global_data, d_seq, seqlen);
    cudaDeviceSynchronize();
    time_t end_time = time(NULL);
    printf("Start Time: %s", ctime(&start_time));
    printf("Stop  Time: %s", ctime(&end_time));
    printf("Delta Time: %d seconds\n", end_time-start_time);
    cudaFree(d_seq);
    cudaFree(global_data);
    return 1;
}
