#include "stdafx.h"
#include "VR_Global_Define.h"
#include <cuda.h> // free gpu memory
#include <stdio.h>

unsigned long inKB(unsigned long bytes)

{ return bytes/1024; }

unsigned long inMB(unsigned long bytes)

{ return bytes/(1024*1024); }

void printStats(unsigned long free, unsigned long total)
{
	printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), inMB(free));
	printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), inMB(total));
	printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, 100.0*(total - free)/(double)total);
}

int getCurrentGPUMemoryInfo()
{
	unsigned int free, total;
	int gpuCount, i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	cuInit(0);
	cuDeviceGetCount(&gpuCount);
	printf("Detected %d GPU\n",gpuCount);

	for (i=0; i<gpuCount; i++)
	{
		cuDeviceGet(&dev,i);
		cuCtxCreate(&ctx, 0, dev);
		res = cuMemGetInfo(&free, &total);
		if(res != CUDA_SUCCESS)
			printf("!!!! cuMemGetInfo failed! (status = %x)", res);

		printf("^^^^ Device: %d\n",i);
		printStats(free, total);
		cuCtxDetach(ctx);
	}
	return 0;

}