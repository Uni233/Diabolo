#ifndef _TRIANGLEMESHSTRUCT
#define _TRIANGLEMESHSTRUCT

namespace YC
{
	namespace Geometry
	{
		struct TriangleMeshNode
		{
			//bool m_bInside;
			float m_TriLinearWeight[8];
			int m_VertexDofs[8*3];
			int nBelongCellId;//for mesh cutting
		};
	}//namespace Geometry
}//namespace YC
#endif//_TRIANGLEMESHSTRUCT