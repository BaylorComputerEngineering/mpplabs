/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *in_h, *out_h;
    float *in_d, *out_d;
    unsigned in_elements, out_elements;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;
    int i;

    // Allocate and initialize host memory
    if(argc == 1) {
        in_elements = 1000000;
    } else if(argc == 2) {
        in_elements = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./reduction          # Input of size 1,000,000 is used"
           "\n    Usage: ./reduction <m>      # Input of size m is used"
           "\n");
        exit(0);
    }
    initVector(&in_h, in_elements);

    out_elements = in_elements / (BLOCK_SIZE<<1);
    if(in_elements % (BLOCK_SIZE<<1)) out_elements++;

    out_h = (float*)malloc(out_elements * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n", in_elements);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&in_d, in_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&out_d, out_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, in_elements * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemset(out_d, 0, out_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    dim_grid.x = out_elements; dim_grid.y = dim_grid.z = 1;
    reduction<<<dim_grid, dim_block>>>(out_d, in_d, in_elements);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(out_h, out_d, out_elements * sizeof(float),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    /* Accumulate partial sums on host */
	for(i=1; i<out_elements; i++) {
		out_h[0] += out_h[i];
	}

	/* Verify the result */
    verify(in_h, in_elements, out_h[0]);

    // Free memory ------------------------------------------------------------

    cudaFree(in_d); cudaFree(out_d);
    free(in_h); free(out_h);

    return 0;
}

