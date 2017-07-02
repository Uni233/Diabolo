#include "stdafx.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>

// includes
#include <cuda_runtime.h>
#include <helper_string.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_timer.h>
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
//#include <cutil_gl_error.h>
#include <rendercheck_gl.h>


void initCuda(int argc, char** argv)
{
  // First initialize OpenGL context, so we can properly set the GL
  // for CUDA.  NVIDIA notes this is necessary in order to achieve
  // optimal performance with OpenGL/CUDA interop.  use command-line
  // specified CUDA device, otherwise use device with highest Gflops/s
	cudaError_t cudaRet;
	if( checkCmdLineFlag(argc, (const char**)argv, "device") ) {
		//		gpuGLDeviceInit(argc, argv);
	} else {
		cudaRet = cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );
	}
}

