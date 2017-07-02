#ifndef _TRIPLETNODE_H
#define _TRIPLETNODE_H

#include "VR_Global_Define.h"

namespace VR_FEM
{
	struct TripletNode
	{
		TripletNode():val(0.0){}
		MyFloat val;
	};
}
#endif//_TRIPLETNODE_H