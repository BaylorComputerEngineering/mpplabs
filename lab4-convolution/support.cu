/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

Matrix allocateMatrix(unsigned height, unsigned width)
{
	Matrix mat;
	mat.height = height;
	mat.width = mat.pitch = width;
	mat.elements = (float*)malloc(height*width*sizeof(float));
	if(mat.elements == NULL) FATAL("Unable to allocate host");

	return mat;
}

void initMatrix(Matrix mat)
{
    for (unsigned int i=0; i < mat.height*mat.width; i++) {
        mat.elements[i] = (rand()%100)/100.00;
    }
}

Matrix allocateDeviceMatrix(unsigned height, unsigned width)
{
	Matrix mat;
	cudaError_t cuda_ret;

	mat.height = height;
	mat.width = mat.pitch = width;
	cuda_ret = cudaMalloc((void**)&(mat.elements), height*width*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

	return mat;
}

void copyToDeviceMatrix(Matrix dst, Matrix src)
{
	cudaError_t cuda_ret;
	cuda_ret = cudaMemcpy(dst.elements, src.elements, src.height*src.width*sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
}

void copyFromDeviceMatrix(Matrix dst, Matrix src)
{
	cudaError_t cuda_ret;
	cuda_ret = cudaMemcpy(dst.elements, src.elements, src.height*src.width*sizeof(float), cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy from device");
}

void verify(Matrix M, Matrix  N, Matrix P) {

  const float relativeTolerance = 1e-6;

  for(int row = 0; row < N.height; ++row) {
    for(int col = 0; col < N.width; ++col) {
      float sum = 0.0f;
      for(int i = 0; i < M.height; ++i) {
        for(int j = 0; j < M.width; ++j) {
            int iN = row - M.height/2 + i;
            int jN = col - M.width/2 + j;
            if(iN >= 0 && iN < N.height && jN >= 0 && jN < N.width) {
                sum += M.elements[i*M.width + j]*N.elements[iN*N.width + jN];
            }
        }
      }
      float relativeError = (sum - P.elements[row*P.width + col])/sum;
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
        printf("TEST FAILED\n\n");
        exit(0);
      }
    }
  }
  printf("TEST PASSED\n\n");

}

void freeMatrix(Matrix mat)
{
	free(mat.elements);
	mat.elements = NULL;
}

void freeDeviceMatrix(Matrix mat)
{
	cudaFree(mat.elements);
	mat.elements = NULL;
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

