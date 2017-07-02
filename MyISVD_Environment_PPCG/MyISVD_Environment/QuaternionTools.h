#include "VR_Global_Define.h"

#include "Frame/Mat_YC.h"
#include "Frame/Axis_YC.h"
//#include "Physic_State_DomainDependent.h"

namespace YC
{
	Axis::Quaternion NormalizeQuaternion(float x, float y, float z, float w);
	Axis::Quaternion InverseSignQuaternion(const Axis::Quaternion& q);
	float innerproduct(const Axis::Quaternion& q1, const Axis::Quaternion& q2);
	bool AreQuaternionsClose(const Axis::Quaternion& q1, const Axis::Quaternion& q2);
	Axis::Quaternion AverageQuaternion(Eigen::Vector4f& cumulative, Axis::Quaternion newRotation, const Axis::Quaternion& firstRotation, int addAmount);
}