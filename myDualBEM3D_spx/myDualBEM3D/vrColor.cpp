#include "vrColor.h"

namespace VR
{
	namespace Colors
	{
		MyDenseVector vecColorBar[ColorBarSize+1] = {MyDenseVector(0,0,1),MyDenseVector(0,1,1),MyDenseVector(0,1,0),MyDenseVector(1,1,0),MyDenseVector(1,0,0),MyDenseVector(1,0,0)};
		MyDenseVector weightToColor(const MyFloat weight)
		{


			MyInt nColorBase = (MyInt)(weight /ColorBarScope/* 1./5. */);
			MyFloat localScalar = (weight - nColorBase * ColorBarScope) / ColorBarScope;
			MyDenseVector colorStep = vecColorBar[nColorBase+1] - vecColorBar[nColorBase];
			MyDenseVector color = vecColorBar[nColorBase] + localScalar * colorStep;

			return color;
		}
	}
}