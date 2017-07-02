#ifndef _discontinuousColletionPoint_h_
#define _discontinuousColletionPoint_h_
#include "bemDefines.h"
namespace VR
{
	struct DisContinuousCollectionPoint
	{
		DisContinuousCollectionPoint():l(std::sqrt(2.0) / 2.0), l1(l * 0.25),l2(l * 0.75),s1(l2 / l), s2(l1 / (2 * l)) 

		{

		}
	private:
		const MyFloat l;
		const MyFloat l1;
		const MyFloat l2;
	public:
		const MyFloat s1;
		const MyFloat s2;
	};
	
}
#endif//_discontinuousColletionPoint_h_