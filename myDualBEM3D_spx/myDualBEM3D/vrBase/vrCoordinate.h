#ifndef _vrCoordinate_h_
#define _vrCoordinate_h_

#include "vrTypes.h"
namespace VR
{
	namespace Coord
	{
		class vec3LessCompare
		{
		public:
			bool operator ()(const vrVec3& lhs, const vrVec3& rhs)const;
		};

		class vec2LessCompare
		{
		public:
			bool operator ()(const vrVec2& lhs, const vrVec2& rhs)const;
		};
	}//namespace Coord
}

#endif//_vrCoordinate_h_