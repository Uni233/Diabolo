#ifndef _VR_MATERIAL_H
#define _VR_MATERIAL_H

#include "VR_Global_Define.h"
namespace YC
{

	namespace Material
	{
		const MyFloat GravityFactor = -9.8f;
		const MyFloat YoungModulus = 700000000;
		const MyFloat PossionRatio = 0.3333333f;
		const MyFloat Density = 770.f;
		const MyFloat damping_alpha = 0.183f;
		const MyFloat damping_beta = 0.00128f ;
	}//namespace Material
}//namespace YC
#endif//_VR_MATERIAL_H