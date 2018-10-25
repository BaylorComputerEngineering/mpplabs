/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initVector(unsigned int **vec_h, unsigned int size, unsigned int num_bins)
{
    *vec_h = (unsigned int*)malloc(size*sizeof(unsigned int));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%num_bins);
    }

}

void verify(unsigned int* input, uint8_t* bins, unsigned int num_elements, unsigned int num_bins) {

  // Initialize reference
  uint8_t* bins_ref = (uint8_t*) malloc(num_bins*sizeof(uint8_t));
  for(unsigned int binIdx = 0; binIdx < num_bins; ++binIdx) {
      bins_ref[binIdx] = 0;
  }

  // Compute reference bins
  for(unsigned int i = 0; i < num_elements; ++i) {
      unsigned int binIdx = input[i];
      if(bins_ref[binIdx] < 255u) {
          ++bins_ref[binIdx];
      }
  }

  // Compare to reference bins
  for(unsigned int binIdx = 0; binIdx < num_bins; ++binIdx) {
      if(bins[binIdx] != bins_ref[binIdx]) {
        printf("TEST FAILED at bin %u, cpu = %u, gpu = %u\n\n", binIdx, bins_ref[binIdx], bins[binIdx]);
        exit(0);
      }
  }
  printf("TEST PASSED\n\n");

  free(bins_ref);

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

