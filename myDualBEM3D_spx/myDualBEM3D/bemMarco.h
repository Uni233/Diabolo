#ifndef _bemMarco_h_
#define _bemMarco_h_

#define DoublePrecision (1)
#define MyError(X)do{printf("%s\n",#X);exit(66);}while(false)
#define MyPause system("pause")
#define MyExit exit(66)
#define Q_ASSERT(X) do{if (!(X)){printf("%s\n",#X);system("pause");}}while(false)
#define MyArrayBase (1)
#define MyTest (1)
#define SHOWMATRIX(mat) std::cout << #mat << " : " << std::endl << mat << std::endl;
#define SHOWVECTOR(vec) std::cout << #vec << " : " << std::endl << vec.transpose() << std::endl;
#define SHOWVAR(var) std::cout << #var << " : " << std::endl << var << std::endl;

#define SHOWPLOT (0)
#define MyDim (3)
#define MyParaDim (2)

#define Hole_Sample (0)
#define Lplate_Sample (0)
#define Spanner_Sample (0)
#define uniaxial_tension_Sample (1)

#if DoublePrecision
#define MyEPS DBL_EPSILON
#else
#define  MyEPS FLT_EPSILON
#endif

#define MyNotice
#define USE_TBB (1)

#define USE_Nouse (0)
#define USE_Mantic_CMat (1)

#define USE_Fracture (1)

#define MyNoticeMsg(X)

#define USE_NEW_VERTEX (1)
#define USE_DUAL (1)
#define USE_VDB (1)
#define USE_SST_DEBUG (0)
#define MYNOTICE
#define USE_DIS (1)
#define DEBUG_5_28 (0)
#define SPEEDUP_5_31 (1)
#define USE_SUBTRI_INTE (1)
#define USE_Sigmoidal (1)
#define USE_MI_METHOD (0)
#define USE_NEW_DUAL_6_3 (1)
#define USE_Aliabadi (1)
#define USE_Aliabadi_RegularSample (1)
#define USE_Nagetive_InDebugBeam (1)
#define USE_Peng_Kernel (0)
#define USE_MI_NegativeSingular (1)
#define USE_cuSolverDn (0)
#define USE_360_Sample (1)
#define USE_UniformSampling (1)
#define USE_DebugGMatrix (1)
#endif//_bemMarco_h_