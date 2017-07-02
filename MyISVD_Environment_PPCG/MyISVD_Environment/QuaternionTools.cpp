#include "QuaternionTools.h"


namespace YC
{
	Axis::Quaternion NormalizeQuaternion(float x, float y, float z, float w)
	{
		float lengthD = 1.0f / (w*w + x*x + y*y + z*z);
		w *= lengthD;
		x *= lengthD;
		y *= lengthD;
		z *= lengthD;

		return Axis::Quaternion(x, y, z, w);
	}

	Axis::Quaternion InverseSignQuaternion(const Axis::Quaternion& q)
	{

		const Axis::MySReal * Q = q.ptr();
		return Axis::Quaternion(-1.f*Q[0], -1.f*Q[1], -1.f*Q[2], -1.f*Q[3]);
	}

	float innerproduct(const Axis::Quaternion& q1, const Axis::Quaternion& q2)
	{
		const Axis::MySReal * Q1 = q1.ptr();
		const Axis::MySReal * Q2 = q2.ptr();
		return Q1[0] * Q2[0] + Q1[1] * Q2[1] + Q1[2] * Q2[2] + Q1[3] * Q2[3];
		//return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
	}

	bool AreQuaternionsClose(const Axis::Quaternion& q1, const Axis::Quaternion& q2){

		float dot = innerproduct(q1, q2);

		if (dot < 0.0f){

			return false;
		}

		else{

			return true;
		}
	}

	Axis::Quaternion AverageQuaternion(Eigen::Vector4f& cumulative, Axis::Quaternion newRotation, const Axis::Quaternion& firstRotation, int addAmount)
	{
		float w = 0.0f;
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;

		//Before we add the new rotation to the average (mean), we have to check whether the quaternion has to be inverted. Because
		//q and -q are the same rotation, but cannot be averaged, we have to make sure they are all the same.
		if (!AreQuaternionsClose(newRotation, firstRotation)){

			newRotation = InverseSignQuaternion(newRotation);
		}
		const float * newRotationPtr = newRotation.ptr();
		//Average the values
		float addDet = 1.f / (float)addAmount;
		cumulative[3] += newRotation[3];
		w = cumulative[3] * addDet;
		cumulative[0] += newRotation[0];
		x = cumulative[0] * addDet;
		cumulative[1] += newRotation[1];
		y = cumulative[1] * addDet;
		cumulative[2] += newRotation[2];
		z = cumulative[2] * addDet;

		//note: if speed is an issue, you can skip the normalization step
		return NormalizeQuaternion(x, y, z, w);
	}
}