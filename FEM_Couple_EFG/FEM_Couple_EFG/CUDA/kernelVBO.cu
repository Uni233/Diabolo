// Simple kernel to modify vertex positions in sine wave pattern
#include "cuPrintf.cu"

__constant__ float4 vertex[8];
__constant__ short2 line[12];


#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define __CUDA_ARCH   100
#define CUPRINTF cuPrintf 
#else						//Compute capability 2.x architectures
#define __CUDA_ARCH   200
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
								blockIdx.y*gridDim.x+blockIdx.x,\
								threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
								__VA_ARGS__)
#endif

__global__ void testKernel(int val)
{
	CUPRINTF("\tValue is:%d\n", val);
}

__global__ void kernel(float4* pos, uchar4 *colorPos, float time)
{
	unsigned int idx = blockIdx.x;
	pos[idx * 2 ] = vertex[line[idx].x];
	pos[idx * 2 + 1] = vertex[line[idx].y];
	colorPos[idx * 2] = make_uchar4(0,128,0,0);
	colorPos[idx * 2 + 1] = make_uchar4(0,128,0,0);
	//CUPRINTF("\t (%d)\n",idx );
	//unsigned int width = 512;
	//unsigned int height = 512;
	//
 //   unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
 //   unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

 //   // calculate uv coordinates
 //   float u = x / (float) width;
 //   float v = y / (float) height;
 //   u = u*2.0f - 1.0f;
 //   v = v*2.0f - 1.0f;

 //   // calculate simple sine wave pattern
 //   float freq = 4.0f;
 //   float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	//CUPRINTF("\t (%f,%f,%f)\n",255.f *0.5*(1.f+sinf(w+x)) ,255.f *0.5*(1.f+sinf(x)*cosf(y)),255.f *0.5*(1.f+sinf(w+time/10.f)) );

    // write output vertex
    /*pos[y*width+x] = make_uchar4(0, 1, 0, 1.0f);
    colorPos[y*width+x].w = 0;
    colorPos[y*width+x].x = 255.f *0.5*(1.f+sinf(w+x));
    colorPos[y*width+x].y = 255.f *0.5*(1.f+sinf(x)*cosf(y));
    colorPos[y*width+x].z = 255.f *0.5*(1.f+sinf(w+time/10.f));*/
}

// Wrapper for the __global__ call that sets up the kernel call
 
	
 


