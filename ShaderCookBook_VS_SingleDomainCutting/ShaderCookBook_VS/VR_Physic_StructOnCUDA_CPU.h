#ifndef _VR_PHYSIC_STRUCTONCUDA_CPU_H
#define _VR_PHYSIC_STRUCTONCUDA_CPU_H

#include "VR_MACRO.h"
#include "MyGLM.h"
#define PHYSIC_STRUCT (1)
#if PHYSIC_STRUCT

struct FEMShapeValue
{
	int    m_nDomainId;
	float  shapeFunctionValue_8_8[8][8];
	float  shapeDerivativeValue_8_8_3[8][8][MyDIM];
};

struct VertexOnCuda
{
	int m_nId;
	int m_nCloneId;
	glm::vec3 local;//float local[3];
	int m_nGlobalDof[3];
	//int m_nLocalDofs[3];
	//int m_nCoupleDofs[3];
	int m_createTimeStamp;//0 : initial; 1 : first cutting;...
	//int m_fromDomainId;//initialize on cpu
};

struct CommonCellOnCuda
{
	/************************************************************************/
	/* Common data                                                                     */
	/************************************************************************/
	char cellType;//FEM:0; EFG:1; Couple:2
	int vertexId[8];//initialize on cpu
	bool m_bLeaf;//initialize on cpu
	int m_nLinesCount;//initialize on cpu
	int m_nLinesBaseIdx;//initialize on cpu
	bool m_bNewOctreeNodeList;//initialize and modify on cuda
	bool m_bTopLevelOctreeNodeList;//initialize and modify on cuda
	int  m_nGhostCellCount;//initialize on cpu
	int m_nGhostCellIdxInVec;//initialize on cpu
	float m_nJxW;
	int m_nCloneCellIdx;//be clone Cell Index
	glm::vec3 m_centerPos;//m_centerPos[MyDIM];
	float m_radius;
	/************************************************************************/
	/* FEM Cell Structure                                                                     */
	/************************************************************************/
	int m_nRhsIdx;
	int m_nMassMatrixIdx;
	int m_nStiffnessMatrixIdx;
	int m_nFEMShapeIdx;
	int m_nLevel;

#if USE_CO_RATION
	/************************************************************************/
	/* Co-ration                                                                     */
	/************************************************************************/
	float old_u[Geometry_dofs_per_cell];
	float RotationMatrix[3*3],RotationMatrix4Inverse[3*3],RotationMatrixTranspose[3*3];
	float radiusx2;float weight4speedup;
	float Pj[Geometry_dofs_per_cell];
	float RxK[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float R[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float Rt[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float RKR[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float CorotaionRhs[Geometry_dofs_per_cell];
	float RKRtPj[Geometry_dofs_per_cell];
	float RotationMatrix_InitSpeedUp[8][3*3];
#endif

#if 1 // speed up for memory band low
	float localMassMatrixOnCuda[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float localRhsVectorOnCuda[Geometry_dofs_per_cell];
#endif

#if USE_CUTTING
	bool m_needBeCutting;
	glm::vec3 m_bladeForceDirct;
	int m_bladeForceDirectFlag;

	float nPointPlane[8];
#endif
};
#endif

#define SKINNING_STRUCT (1)
#if SKINNING_STRUCT

struct MC_Vertex_Cuda 
{
	bool m_isValid;
	bool m_isJoint;//不参与本次分裂
	bool m_isSplit;
	int  m_nVertexId;
	glm::vec3 m_VertexPos;//float m_VertexPos[3];
	float m_VertexPos4CloneTestBladeDistance[3];

	//float m_VertexNormal[3];
	int m_CloneVertexIdx[2];
	float m_distanceToBlade;
	char m_state;
	int m_elemVertexRelatedDofs[24];
	float m_TriLinearWeight[8];
	int m_MeshVertex2CellId;

	//int m_nCellBelongDomainId;//for multi domain

#if USE_DYNAMIC_VERTEX_NORMAL
	int m_nShareTriangleCount;
	int m_eleShareTriangle[MaxVertexShareTriangleCount];
	float m_VertexNormalValue[3];
#endif
};

struct MC_Edge_Cuda
{//none order
	bool m_hasClone;
	bool m_isValid;
	bool m_isJoint;//不参与本次分裂
	bool m_isCut;

	char m_state;
	int  m_nLineId;
	int  m_Vertex[2];
	int  m_belongToTri[MaxLineShareTri];//valid is -1
	int  m_belongToTriVertexIdx[MaxLineShareTri][2];		
	float m_intersectPos[3];
	int m_CloneIntersectVertexIdx[2];//splite vertex for blade
	int m_CloneEdgeIdx[2];//splite edge for blade

	int m_beCuttingBladeId;//for complex blade
	float m_intersectAngle;
};

struct MC_Surface_Cuda
{
	bool m_isValid;
	bool m_isValid4InnerBladeFace;
	bool m_isJoint;//不参与本次分裂
	int  m_nSurfaceId;
	int  m_Vertex[3];
	int  m_Lines[3];
	char m_state;
	float m_center[3];
	int  m_VertexNormal[3];
	float m_R;
	float m_R2;
	bool isInnerBlade;

#if USE_DYNAMIC_VERTEX_NORMAL
	float m_vertexNormalValue[3];

#endif
};

struct CuttingBladeStruct
{
#define MaxCuttingBladeStructCount (999)
#define MinCuttingBladeArea (1e-6)
	glm::vec3 m_currentRayHandle;
	glm::vec3 m_currentRayTip;
	glm::vec3 m_lastRayHandle;
	glm::vec3 m_lastRayTip;

	float m_cuttingArea_lh_lt_ct,m_cuttingArea_ct_ch_lh;
	glm::vec3 m_bladeNormal_lh_lt_ct;
	glm::vec3 m_bladeNormal_ct_ch_lh;
};

struct CuttingRayStruct// model native ray from arcball
{
	glm::vec3 m_RayHandle;
	glm::vec3 m_RayTip;
	bool  m_isIntersection;
	float m_IntersectionWithSurfacePos[3];
	int   m_IntersectionSurfaceId;

#if USE_CUTTING
	float m_bladeNormal[3];//v0 = m_RayEndPos, v1 = m_RayStartPos of last ray, v2 = m_RayStartPos
#endif
};

struct RefineCuttingRayStruct
{
	float m_RayStartPos[3];
	float m_RayEndPos[RayEndCount][3];
	int m_4trike;
	float m_bladeStartPos[3];
	int m_bladeId;
	float m_intersectAngle;
	bool m_isValid;
	int m_MC_EdgeId;
	int m_RayEndPosBelongCellId[RayEndCount];
	int m_RayStartPosBelongCellId;
	int m_RayEndPosId[RayEndCount][2];//2 means up and down the blade
	float m_distanceToBlade[2];
	float nouse_clonePosForDebug[2][3];
};

struct ExternalForceRayStruct
{
	float m_forceStartPos[3];
	float m_forceEndPos[3];
	float m_forceDirct[3];
	float m_forceDirctForDraw[3];
	bool  m_isIntersection;
	float m_IntersectionWithSurfacePos[3];
	int   m_IntersectionSurfaceId;
	int   m_IntersectionCellId[3];
	int   m_IntersectionCellFromDomainId[3];
	float m_forceScaleFactor;
};

struct VBOStructForDraw
{
	//Draw Line Set
	int * vbo_line_vertex_pair;//{leftVertexId,rightVertexId,domainId}
	int vbo_lineCount;

	//Draw Triangle Set
	int g_nMCVertexSize;
	int g_nMaxMCVertexSize;
	int g_nLastVertexSize;
	int g_nMCEdgeSize;
	int g_nMaxMCEdgeSize;
	int g_nLastEdgeSize;
	int g_nMCSurfaceSize;
	int g_nMaxMCSurfaceSize;
	int g_nLastSurfaceSize;
	int g_nVertexNormalSize;
	MC_Vertex_Cuda*		g_MC_Vertex_Cuda;
	MC_Edge_Cuda*		g_MC_Edge_Cuda;
	MC_Surface_Cuda*	g_MC_Surface_Cuda;
	float * g_elementVertexNormal;//g_nVertexNormalSize * 3

	int *g_CuttedEdgeFlagOnCpu;
	int *g_CuttedEdgeFlagOnCuda;

	int *g_SplittedFaceFlagOnCpu;
	int *g_SplittedFaceFlagOnCuda;

	//cutting
	/*int g_CuttingRayOnCpuSize,g_Nouse;
	CuttingRayStruct * g_g_CuttingRayStructOnCpu;
	CuttingRayStruct * g_g_CuttingRayStructOnCuda;

	int g_RefineCuttingRayOnCpuSize,g_RefineCuttingRayOnCpuSizeDebug;
	RefineCuttingRayStruct * g_RefineCuttingRayStructOnCpu;
	RefineCuttingRayStruct * g_RefineCuttingRayStructOnCuda;*/
	int g_nCuttingBladeCount;
	CuttingBladeStruct * g_CuttingBladeStructOnCpu;
	CuttingBladeStruct * g_CuttingBladeStructOnCuda;

	//external force
	int g_ExternalForceRaySize;
	ExternalForceRayStruct* g_ExternalForceRayOnCpu;
	ExternalForceRayStruct* g_ExternalForceRayOnCuda;
};
#endif//#if SKINNING_STRUCT
#endif//_VR_PHYSIC_STRUCTONCUDA_CPU_H