#ifndef _VR_CUDA_GLOBALDEFINE_CUH
#define _VR_CUDA_GLOBALDEFINE_CUH

#include "VR_MACRO.h"

#if USE_CUDA
#include "driver_types.h"
#include <stdlib.h>
#include <stdio.h>

#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>




static void HandleError( cudaError_t err,
	const char *file,
	int line ) {
		if (err != cudaSuccess) {
			printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
				file, line );
			exit( EXIT_FAILURE );
		}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define GRIDCOUNT(THREAD_COUNT, THREAD_COUNT_PER_BLOCK) ((THREAD_COUNT + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK)

#define WRAP_SIZE (32)
#define MAX_BLOCK_SIZE (8192)
#define MAX_KERNEL_PER_BLOCK (1024)
#endif

#endif//_VR_CUDA_GLOBALDEFINE_CUH