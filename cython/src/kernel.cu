#include <stdio.h>

void __global__ kernel_add_one(int* a, int length) {
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    if (gid < length) {
    	a[gid] += 1;
    }
}
