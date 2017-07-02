#include "vrCoordinate.h"

namespace VR
{
	namespace Coord
	{
		bool vec3LessCompare::operator()(const vrVec3& lhs, const vrVec3& rhs)const
		{
			if (lhs[2] < rhs[2])
			{
				return true;
			}
			else if (lhs[2] > rhs[2])
			{
				return false;
			}
			// Otherwise z is equal
			if (lhs[1] < rhs[1])
			{
				return true;
			}
			else if (lhs[1] > rhs[1])
			{
				return false;
			}
			// Otherwise z and y are equal
			if (lhs[0] < rhs[0])
			{
				return true;
			}
			/* Simple optimization Do not need this test
			If this fails or succeeded the result is false.
			else if( lhs.x > rhs.x )
			{    return false;
			}*/
			// Otherwise z and y and x are all equal
			return false;
		}

		
		bool vec2LessCompare::operator()(const vrVec2& lhs, const vrVec2& rhs)const
		{
			// Otherwise z is equal
			if (lhs[1] < rhs[1])
			{
				return true;
			}
			else if (lhs[1] > rhs[1])
			{
				return false;
			}
			// Otherwise z and y are equal
			if (lhs[0] < rhs[0])
			{
				return true;
			}
			/* Simple optimization Do not need this test
			If this fails or succeeded the result is false.
			else if( lhs.x > rhs.x )
			{    return false;
			}*/
			// Otherwise z and y and x are all equal
			return false;
		}
	}//namespace Coord
}//namespace VR