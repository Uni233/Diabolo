#ifndef _VR_MACRO_H
#define _VR_MACRO_H

namespace YC
{
	
namespace MACRO
{
	

	#define Platform_Windows (1)
	#define PlatForm_Linux   (2)
	#define PlatformType Platform_Windows

	#define MyError(X)do{printf("%s\n",#X);exit(66);}while(false)
	#define MyPause system("pause")
	#define MyExit exit(66)
	#define Q_ASSERT(X) do{if (!(X)){printf("%s\n",#X);system("pause");}}while(false)
	#define GL_CHECK_ERRORS assert(glGetError()== GL_NO_ERROR);
	#define MyNotice
	#define MyMod(A,B) (A - B * (A / B))

	#define windowTitle "VR Physics Scene"
	#define LogInfo printf
	#define cutilCheckError
	#define windowTitle "VR Physics Scene"
	#define windowWidth (800)
	#define windowHeight (600)
	#define MyBufferSize (256)
	#define _nExternalMemory (2)
	#define MaterialMatrixSize (6)
	#define ValueScaleFactor (10000000000)
	#define ModelStep (0.f)

	#define Invalid_Id (-1)
	#define _CellTypeFEM (0)
	#define _CellTypeEFG (1)
	#define _CellTypeCouple (2)
	#define _CellTYPEInvalid (3)

	typedef enum{FEM=_CellTypeFEM,EFG=_CellTypeEFG,COUPLE=_CellTypeCouple,INVALIDTYPE=_CellTYPEInvalid} CellType;

	#define MyZero (0)
	#define MyNull (0)
	#define X__AXIS (0)
	#define Y__AXIS (1)
	#define Z__AXIS (2)
	#define MyDIM (3)
	#define MyYES (0)
	#define MyNo  (Invalid_Id)

#define ModelNameIsArmadillo (1)
#define ModelNameIsBunny (0)
#define ModelNameIsBeam ()
#define CellLevel (6)
#if CellLevel == 4
	#define CellRaidus   (0.015625f*4)
#elif CellLevel == 5
	#define CellRaidus   (0.015625f*2)
#elif CellLevel == 6
	#define CellRaidus   (0.015625f)
#else
	#error UnSupport Cell Level (4,5,6).
#endif

#define SHOW_SHADOWMAP (1)
#if SHOW_SHADOWMAP
	#define USE_PCF (1)
#endif
#define SHOWFPS (1)
#define SHOWTRIMESH (0)
#define SHOWGRID (1)

	#define USE_CUDA (1)
	#define USE_CO_RATION (1)
	#define USE_CUTTING (1)
	/*
	#define USE_GLOBAL_CUTTING (1)
	#define USE_CUDA (1)
	#define ShowTriangle (1)
	#define ShowLines (0)*/


	/************************************************************************/
	/* CUDA Macro Definition
	/************************************************************************/
	#define Geometry_dofs_per_cell (24)
	#define Geometry_dofs_per_cell_squarte (Geometry_dofs_per_cell * Geometry_dofs_per_cell)
	#define VertxMaxInflunceCellCount (8)
	#define Dim_Count (3)
	#define CellMaxInflunceVertexCount (8)
	#define nMaxNonZeroSize  (VertxMaxInflunceCellCount*Dim_Count*CellMaxInflunceVertexCount)
	#define MAX_DOFS_COUNT (30000)
	#define MyMatrixRows (3)
	#define MaxCellCount (10000)
	#define KERNEL_COUNT_TMP (30000)
	#define MaxVertexCount (100000)
	#define NewMarkConstant_0 (16384.0f)
	#define NewMarkConstant_1 (128.0f)
	#define nMaxNonZeroSizeInFEM  (192)
	#define MaxCellDof4FEM   (24)
	#define MASS_MATRIX_COEFF_2 (400.f)
	#define LocalMaxDofCount_YC (24)
	#define Material_Damping_Alpha (0.183f)
	#define Material_Damping_Beta (0.00128f)
	//(0.00128f)
	#define  USE_DYNAMIC_VERTEX_NORMAL (0)
	#define MaxVertexShareTriangleCount (10)
	#define MaxLineShareTri (3)
	#define Max_Triangle_Count (100000)
	#define Max_Vertex_Count (MaxVertexCount)
	#define RayEndCount (1)
	#define MaxCuttingRayCount (40)
	#define MaxRefineCuttingRayCount (999)
	#define VBO_LineSetSize (3)
	#define USE_OUTPUT_RENDER_OBJ_MESH (0)
	#define MY_DEBUG_OUTPUT_GPU_MATRIX (1)
	#define CONST_24x24 (576)
	#define CONST_24 (24)
	#define USE_HOST_MEMORY (1)
	#define USE_CUDA_STREAM (1)
	#define USE_NVTX (0)
	#define MY_VIDEO_OUTPUT_OBJ (0)

	/************************************************************************/
	/* Mesh Cutting Macro Definition
	/************************************************************************/
	#define USE_Mesh_Cutting (1)
}//namespace MARCO

}//namespace YC

#endif//_VR_MACRO_H