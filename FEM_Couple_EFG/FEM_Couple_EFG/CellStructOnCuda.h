#ifndef _CellStructOnCuda
#define _CellStructOnCuda

#define Geometry_dofs_per_cell (24)
#define Geometry_dofs_per_cell_squarte (Geometry_dofs_per_cell * Geometry_dofs_per_cell)
#define Geometry_tris_per_cell (12)
#define MaxDofCount (15000)
#define MaxVertexCount (11000)
#define MaxCellCount (10000)
#define EFG_BasisNb_ (4)
#define dim (3)
/*struct CellOnCuda
{
	int m_nRhsIdx;
	int m_nMassMatrixIdx;
	int m_nStiffnessMatrixIdx;
	int localDofs[Geometry_dofs_per_cell];
	int globalDofs[Geometry_dofs_per_cell];
	int vertexId[8];
	float nPointPlane[8];
	int m_nLinesCount;
	int m_nLinesBaseIdx;
	bool m_bLeaf;
	int m_nLevel;
	bool m_bNewOctreeNodeList;
	bool m_bTopLevelOctreeNodeList;
	int  m_nGhostCellCount;
	int m_nGhostCellIdxInVec;
};*/

struct FEMShapeValue
{
	float  shapeFunctionValue_8_8[8][8];
	float  shapeDerivativeValue_8_8_3[8][8][dim];
};

//2 Cell Radius Max Influnce Points --->8
//3 Cell Radius Max Influnce Points --->36
//4 Cell Radius Max Influnce Points --->68
// 80
#define EFGInflunceMaxSize (8)
#define EFGInflunceMaxSize_0 (0)
#define EFGInflunceMaxSize_1  (EFGInflunceMaxSize)
#define EFGInflunceMaxSize_2  (EFGInflunceMaxSize_1 + EFGInflunceMaxSize)
#define EFGInflunceMaxSize_3  (EFGInflunceMaxSize_2 + EFGInflunceMaxSize)
struct EFGCellOnCuda
{
	/************************************************************************/
	/* Common data                                                                     */
	/************************************************************************/
	char cellType;//FEM:0; EFG:1; Couple:2
	int vertexId[8];//initialize on cpu
	float nPointPlane[8];//initialize and modify on cuda
	bool m_bLeaf;//initialize on cpu
	int m_nLinesCount;//initialize on cpu
	int m_nLinesBaseIdx;//initialize on cpu
	bool m_bNewOctreeNodeList;//initialize and modify on cuda
	bool m_bTopLevelOctreeNodeList;//initialize and modify on cuda
	int  m_nGhostCellCount;//initialize on cpu
	int m_nGhostCellIdxInVec;//initialize on cpu

	int m_nCloneCellIdx;//be clone Cell Index
	/************************************************************************/
	/* FEM Cell Structure                                                                     */
	/************************************************************************/
	int m_nRhsIdx;
	int m_nMassMatrixIdx;
	int m_nStiffnessMatrixIdx;
	int m_nLevel;
	/************************************************************************/
	/* EFG Cell Structure                                                                     */
	/************************************************************************/
	float m_EFGGobalGaussPoint[8][3];//initialize on cpu
	int m_nCellInfluncePointSize;//[0,EFGInflunceMaxSize]
	float m_nJxW;//initialize on cpu
	int m_influncePointList[8][EFGInflunceMaxSize];
	int m_cellInflunceVertexList[EFGInflunceMaxSize];
	float  N_[8][EFGInflunceMaxSize];
	float  D1N_[8][EFGInflunceMaxSize][dim];

	/************************************************************************/
	/* Couple Cell Structure                                                                     */
	/************************************************************************/
	//int m_nEFGPtsSize;// m_nEFGPtsSize < 8
	//int m_VertexIdInEFG_Boundary[8];
	float m_ShapeFunction_R[8];
	float m_ShapeDeriv_R[8][dim];

	/************************************************************************/
	/* Co-ration                                                                     */
	/************************************************************************/
	float old_u[Geometry_dofs_per_cell];
	float RotationMatrix[3*3],RotationMatrix4Inverse[3*3];
	float radiusx2;
	//float Pj[Geometry_dofs_per_cell];
	float RxK[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float R[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float Rt[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float RKR[Geometry_dofs_per_cell*Geometry_dofs_per_cell];
	float CorotaionRhs[Geometry_dofs_per_cell];
	float RKRtPj[Geometry_dofs_per_cell];
};

struct EFGCellOnCuda_Ext
{
	float  A_[8][EFG_BasisNb_][EFG_BasisNb_];
	float  Ax_[8][EFG_BasisNb_][EFG_BasisNb_];
	float  Ay_[8][EFG_BasisNb_][EFG_BasisNb_];
	float  Az_[8][EFG_BasisNb_][EFG_BasisNb_];

	float  B_[8][EFG_BasisNb_][EFGInflunceMaxSize];   
	float  Bx_[8][EFG_BasisNb_][EFGInflunceMaxSize];  
	float  By_[8][EFG_BasisNb_][EFGInflunceMaxSize];  
	float  Bz_[8][EFG_BasisNb_][EFGInflunceMaxSize]; 

	float  P_[8][EFG_BasisNb_];
	float  D1P_[8][EFG_BasisNb_][dim];

	float lltOfA[8][EFG_BasisNb_][EFG_BasisNb_];
};

struct VertexOnCuda
{
	int m_nId;
	int m_nCloneId;
	float local[3];
	int m_nDof[3];
	int m_createTimeStamp;//0 : initial; 1 : first cutting;...
	int m_fromDomainId;//initialize on cpu
};

#endif//_CellStructOnCuda