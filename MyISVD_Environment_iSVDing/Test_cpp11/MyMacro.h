#ifndef _MY_MACRO_H
#define _MY_MACRO_H
#define DoublePrecision (1)

#define  USE_ROCORATION (0)
#define  window_width  (1024)
#define  window_height (768)
#define Invalid_Id (-1)
#define MyNan (-17)
#define MyZero (0)
#define MyNull (0)
#define X_AXIS (0)
#define Y_AXIS (1)
#define Z_AXIS (2)
#define MyError(X)do{printf("%s\n",#X);exit(66);}while(false)
#define MyPause system("pause")
#define MyExit exit(66)
#define Q_ASSERT(X) do{if (!(X)){/*printf("%s\n",#X);*/system("pause");}}while(false)
#define myDim (3)
#define MyDIM myDim
#define MaterialMatrixSize (6)
#define MaxTimeStep (10000)
#define ValueScaleFactor (10000000000)
#define CellRaidus   (0.015625)
//#define CellRaidus   (0.129077f)

#define ShowTriangle 1
#define ShowLines 1
#define MyBufferSize (256)

#define USE_SUBSPACE (0)
#define USE_MODAL_WARP (1)
#define PARDISO_SOLVER_IS_AVAILABLE
#define MyNotice
#define LogInfo printf

#define MyMeshPath "D:/MyWorkspace/MyMesh/OBJ/"

#define USE_TBB (1)
#define USE_MAKE_CELL_SURFACE (1)
#define USE_SUBSPACE_Whole_Spectrum (0)
#define USE_SUBSPACE_SVD (1)

#define USE_MultiDomain (0)
#if USE_MultiDomain

	#define USE_PPCG (1)
	#define LocalDomainCount (6)
#else
	#define USE_PPCG (0)
	#define LocalDomainCount (1)
	#define SingleDomainISVD (1)
#endif//USE_MultiDomain



#endif//_MY_MACRO_H