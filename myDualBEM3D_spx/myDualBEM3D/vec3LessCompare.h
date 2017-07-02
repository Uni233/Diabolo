#ifndef _vec3LessCompare_h_
#define _vec3LessCompare_h_

#include "bemDefines.h"

namespace VR
{
	class vec3LessCompare
	{
	public:
		bool operator()(const MyVec3& lhs,const MyVec3& rhs)const
		{
			if( lhs.z() < rhs.z() )
			{   return true;
			}
			else if(lhs.z() > rhs.z())
			{   return false;
			}
			// Otherwise z is equal
			if( lhs.y() < rhs.y() )
			{   return true;
			}
			else if( lhs.y() > rhs.y() )
			{   return false;
			}
			// Otherwise z and y are equal
			if( lhs.x() < rhs.x() )
			{   return true;
			}
			/* Simple optimization Do not need this test
			If this fails or succeeded the result is false.
			else if( lhs.x > rhs.x )
			{    return false;
			}*/
			// Otherwise z and y and x are all equal
			return false;
		}
	};
}
#endif//_vec3LessCompare_h_