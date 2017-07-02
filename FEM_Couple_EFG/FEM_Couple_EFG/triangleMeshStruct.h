#ifndef _TRIANGLEMESHSTRUCT
#define _TRIANGLEMESHSTRUCT

struct TriangleMeshNode
{
	TriangleMeshNode()
	{
		memset(&m_TriLinearWeight[0],'\0',sizeof(m_TriLinearWeight));
		memset(&m_VertexDofs[0],'\0',sizeof(m_VertexDofs));
		memset(&m_verticeToCellVertexLength[0],'\0',sizeof(m_verticeToCellVertexLength));
		m_bInside = false;
		m_nMinestLengthOfCellVertex = -1;
		nBelongToCellIdx = -1;
	}
	bool m_bInside;
	float m_TriLinearWeight[8];
	unsigned m_VertexDofs[8*3];
	float m_verticeToCellVertexLength[8];
	int   m_nMinestLengthOfCellVertex;
	int  nBelongToCellIdx;
};
#endif//_TRIANGLEMESHSTRUCT