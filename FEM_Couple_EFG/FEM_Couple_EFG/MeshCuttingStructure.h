#ifndef _MeshCuttingStructure_H
#define _MeshCuttingStructure_H

#include "VR_Global_Define.h"

#define MaxLineShareTri (3)

namespace VR_FEM
{
	struct MC_Vertex 
	{
		bool m_isValid;
		bool m_isJoint;
		bool m_isSplit;
		int  m_nVertexId;
		MyDenseVector m_VertexPos;
		MyDenseVector m_VertexNormal;
		int m_CloneVertexIdx[2];
		MyFloat m_distanceToBlade;
		char m_state;
	};

	struct MC_Edge
	{//none order
		bool m_hasClone;
		bool m_isValid;
		bool m_isJoint;
		bool m_isCut;

		char m_state;
		int  m_nLineId;
		int  m_Vertex[2];
		int  m_belongToTri[MaxLineShareTri];//valid is -1
		int  m_belongToTriVertexIdx[MaxLineShareTri][2];		
		MyDenseVector m_intersectPos;
		int m_CloneIntersectVertexIdx[2];
		int m_CloneEdgeIdx[2];

		void printInfo()
		{
			printf("m_hasClone[%s],m_isValid[%s],m_isJoint[%s],m_isCut[%s]\n",(m_hasClone? "true":"false"),(m_isValid? "true":"false"),(m_isJoint? "true":"false"),(m_isCut? "true":"false") );
			printf("m_state[%d],m_nLineId[%d],m_Vertex[%d,%d],m_belongToTri[%d,%d,%d]\n",m_state,m_nLineId,m_Vertex[0],m_Vertex[1],m_belongToTri[0],m_belongToTri[1],m_belongToTri[2]);
			printf("m_belongToTriVertexIdx[%d,%d][%d,%d][%d,%d],\n",m_belongToTriVertexIdx[0][0],m_belongToTriVertexIdx[0][1],m_belongToTriVertexIdx[1][0],m_belongToTriVertexIdx[1][1],m_belongToTriVertexIdx[2][0],m_belongToTriVertexIdx[2][1]);
			printf("m_intersectPos(%f,%f,%f),m_CloneIntersectVertexIdx[%d,%d],m_CloneEdgeIdx[%d,%d]\n",m_intersectPos[0],m_intersectPos[1],m_intersectPos[2],m_CloneIntersectVertexIdx[0],m_CloneIntersectVertexIdx[1],m_CloneEdgeIdx[0],m_CloneEdgeIdx[1]);
		}
	};

	struct MC_Surface
	{
		bool m_isValid;
		bool m_isJoint;
		int  m_nSurfaceId;
		int  m_Vertex[3];
		int  m_Lines[3];
		char m_state;
		MyPoint m_center;
	};


}

#endif//_MeshCuttingStructure_H