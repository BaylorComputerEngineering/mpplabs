/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void preScanKernel(float *out, float *in, unsigned size, float *sum)
{

	    // INSERT CODE HERE
}


__global__ void addKernel(float *out, float *sum, unsigned size)
{
    // INSERT CODE HERE

}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
	float *sum;
	unsigned num_blocks;
	cudaError_t cuda_ret;
	dim3 dim_grid, dim_block;

	num_blocks = in_size/(BLOCK_SIZE*2);
	if(in_size%(BLOCK_SIZE*2) !=0) num_blocks++;

	dim_block.x = BLOCK_SIZE; dim_block.y = 1; dim_block.z = 1;
	dim_grid.x = num_blocks; dim_grid.y = 1; dim_grid.z = 1;

	if(num_blocks > 1) {
		cuda_ret = cudaMalloc((void**)&sum, num_blocks*sizeof(float));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, sum);
		preScan(sum, sum, num_blocks);
		addKernel<<<dim_grid, dim_block>>>(out, sum, in_size);

		cudaFree(sum);
	}
	else
		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, NULL);
}
