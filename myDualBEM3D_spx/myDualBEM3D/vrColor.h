#ifndef _vrColor_h_
#define _vrColor_h_

#include "bemDefines.h"
#include "vrBase/vrTypes.h"
namespace VR
{

	namespace Colors
	{
		

#define ColorBarSize  (5)
#define ColorBarScope (1.0/ColorBarSize)
		//static MyDenseVector vecColorBar[ColorBarSize+1];

		MyDenseVector weightToColor(const MyFloat weight);
	}//namespace Colors
}//namespace VR

#endif//_vrColor_h_