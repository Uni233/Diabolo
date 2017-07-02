#ifndef _VR_GPU_UTILS_H
#define _VR_GPU_UTILS_H

#include "VR_MACRO.h"

#if USE_CUDA
namespace YC
{
	namespace Geometry
	{
		namespace GPU
		{
			namespace Utils
			{
				void getCurrentGPUMemoryInfo();
			}
		}
	}
}

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
	int color_id = cid; \
	color_id = color_id%num_colors;\
	nvtxEventAttributes_t eventAttrib = {0}; \
	eventAttrib.version = NVTX_VERSION; \
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
	eventAttrib.colorType = NVTX_COLOR_ARGB; \
	eventAttrib.color = colors[color_id]; \
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
	eventAttrib.message.ascii = name; \
	nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
class MyTracer {
public:
	MyTracer(const char* name,int colorId) {
		PUSH_RANGE(name,colorId);
	}
	~MyTracer() {
		POP_RANGE;
	}
};
#define MY_RANGE(name,colorId) MyTracer uniq_name_using_macros(name,colorId);
#else
//#define PUSH_RANGE(name,cid)
//#define POP_RANGE
#define MY_RANGE(name,colorId)
#endif
#endif

#endif//_VR_GPU_UTILS_H