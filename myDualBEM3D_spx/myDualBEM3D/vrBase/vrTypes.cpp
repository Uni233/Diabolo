#include "vrTypes.h"

namespace VR
{
	vrFloat3 make_vrFloat3(vrFloat x, vrFloat y, vrFloat z)
	{
		vrFloat3 retV;
		retV.x = x; retV.y = y; retV.z = z;
		return retV;
	}

	vrGLMVec3 make_glm_vec3(const vrVec3& src)
	{
		return vrGLMVec3(src.x(), src.y(), src.z());
	}
}//VR