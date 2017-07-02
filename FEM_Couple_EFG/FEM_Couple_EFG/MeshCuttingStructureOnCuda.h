#ifndef _MeshCuttingStructureOnCuda_H
#define _MeshCuttingStructureOnCuda_H

#define MaxLineShareTri (3)

struct MC_Vertex_Cuda 
{
	bool m_isValid;
	bool m_isJoint;
	bool m_isSplit;
	int  m_nVertexId;
	float m_VertexPos[3];
	float m_VertexNormal[3];
	int m_CloneVertexIdx[2];
	float m_distanceToBlade;// > 0 Up; <0 Down
	char m_state;
	int m_elemVertexRelatedDofs[24];
    float m_TriLinearWeight[8];
    int m_MeshVertex2CellId;

	//cutting mesh generate
	int m_brotherPoint;//clone point  modify in cuda_spliteLine
};

struct MC_Edge_Cuda
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
	float m_intersectPos[3];
	int m_CloneIntersectVertexIdx[2];
	int m_CloneEdgeIdx[2];

	//for cutting mesh generate
	//int m_cuttingMeshValidPort;//-1 : not cutting; 0 : 0point; 1: 1point
	//int m_nParentId;
};

struct MC_Surface_Cuda
{
	bool m_isValid;
	bool m_isJoint;
	int  m_nSurfaceId;
	int  m_Vertex[3];
	int  m_Lines[3];
	char m_state;
	float m_center[3];
    int  m_VertexNormal[3];
	float m_R;
	float m_R2;
	//for cutting mesh generate
	int  m_nParentId4MC;
	int m_nCloneBladeLine[2]; // modify in spliteSurface; 0: Up; 1 : Down
};

struct MC_CuttingEdge_Cuda
{
	bool bCut;
	int  m_MeshSurfaceParentId;
	char validPort;//which point near the mass point
};
#define MaxBladeTriHasPoint (5)
struct MC_CuttingFace_Cuda
{
	int m_pointId[MaxBladeTriHasPoint];
	float m_angle[MaxBladeTriHasPoint];
	char m_pointSize;// point >= 2
	char m_state;//1 : one point; 2 : two point
};


#endif//_MeshCuttingStructureOnCuda_H