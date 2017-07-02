#ifndef _PLANE_H
#define _PLANE_H

#include "VR_Global_Define.h"

namespace VR_FEM
{
	class Plane
	{
	public:
		Plane(MyPoint a,MyPoint b,MyPoint c)
			:m_a(a),m_b(b),m_c(c)
		{
			m_normal = (m_a - m_b).cross( m_c-m_b );
		}
		MyFloat checkPointPlane(const MyPoint& p)const
		{
			MyDenseVector tmpVec = (p-m_a);
			return m_normal[0]*tmpVec[0] + m_normal[1]*tmpVec[1] + m_normal[2]*tmpVec[2];
		}

		bool checkIntersect(const std::vector< std::pair<MyPoint,MyPoint> >& vecLines)const
		{
			bool bRet = false;
			MyPoint tmp;
			for (unsigned i=0;i<vecLines.size();++i)
			{
				const std::pair<MyPoint,MyPoint>& curPair = vecLines[i];
				bRet = bRet || CheckLineTri(curPair.first,curPair.second,m_a,m_b,m_c,tmp);
			}
			return bRet;
		}

	private:
		bool CheckLineTri( const MyPoint &L1, const MyPoint &L2, const MyPoint &PV1, const MyPoint &PV2, const MyPoint &PV3, MyPoint &HitP )const
		{
			MyPoint VIntersect;

			// Find Triangle Normal, would be quicker to have these computed already
			MyPoint VNorm;
			VNorm = ( PV2 - PV1 ).cross( PV3 - PV1 );
			//printf("tmpVec3(%f,%f,%f)\n",VNorm.x,VNorm.y,VNorm.z);
			VNorm.normalize();
			//printf("VNorm(%f,%f,%f)\n",VNorm.x,VNorm.y,VNorm.z);

			// Find distance from L1 and L2 to the plane defined by the triangle
			MyFloat fDst1 = (L1-PV1).dot( VNorm );
			//printf("fDst1 is %f \n",fDst1);
			MyFloat fDst2 = (L2-PV1).dot( VNorm );
			//printf("fDst2 is %f \n",fDst2);

			if ( (fDst1 * fDst2) >= 0.0f) return false;  // line doesn't cross the triangle.
			if ( fDst1 == fDst2) {return false;} // line and plane are parallel

			// Find point on the line that intersects with the plane
			VIntersect = L1 + (L2-L1) * ( -fDst1/(fDst2-fDst1) );
			//printf("VIntersect(%f,%f,%f)\n",VIntersect.x,VIntersect.y,VIntersect.z);

			// Find if the interesection point lies inside the triangle by testing it against all edges
			MyPoint VTest;
			VTest = VNorm.cross( PV2-PV1 );
			//printf("VTest(%f,%f,%f)\n",VTest.x,VTest.y,VTest.z);
			if ( VTest.dot( VIntersect-PV1 ) < 0.0f ) return false;
			VTest = VNorm.cross( PV3-PV2 );
			//printf("VTest(%f,%f,%f)\n",VTest.x,VTest.y,VTest.z);
			if ( VTest.dot( VIntersect-PV2 ) < 0.0f ) return false;
			VTest = VNorm.cross( PV1-PV3 );
			//printf("VTest(%f,%f,%f)\n",VTest.x,VTest.y,VTest.z);
			if ( VTest.dot( VIntersect-PV1 ) < 0.0f ) return false;

			HitP = VIntersect;

			return true;
		}
	public:
		MyPoint m_a,m_b,m_c;
		MyDenseVector m_normal;
	};
}
#endif//_PLANE_H