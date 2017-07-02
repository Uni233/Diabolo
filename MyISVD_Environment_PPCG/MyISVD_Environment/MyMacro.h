#ifndef _MY_MACRO_H
#define _MY_MACRO_H
#define DoublePrecision (1)
#define USE_LM (0)

#define  USE_ROCORATION (0)
#define  window_width  (1024)
#define  window_height (768)
#define Invalid_Id (-1)
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


#define LocalDomainCount (6)
#define USE_ALM (0)

#define USE_SUBSPACE (0)
#define USE_MODAL_WARP (1)
#define PARDISO_SOLVER_IS_AVAILABLE
#define MyNotice
#define LogInfo printf

#define MyMeshPath "D:/MyWorkspace/MyMesh/OBJ/"

#define USE_TBB (1)
#define USE_MAKE_CELL_SURFACE (1)

#define USE_MultidomainIndependent (1)
#define USE_PPCG (1)
#define USE_PPCG_Permutation (1)

namespace PMI //PhysicMatrixIdentity
{
	enum{ ComputeMat = 0, StiffMat = 1, MassMat = 2, DampingMat = 3, ConstraintMat = 4, RhatMat = 5, PhysicMatNum = 6 };
	enum{
		ComputeRhs_Vec = 0, R_rhs_Vec = 1, R_rhs_externalForce_Vec = 2, mass_rhs_Vec = 3, damping_rhs_Vec = 4,
		displacement_Vec = 5, velocity_Vec = 6, acceleration_Vec = 7, old_acceleration_Vec = 8,
		old_displacement_Vec = 9, incremental_displacement_Vec = 10, PhysicVecNum = 11
	};
}
namespace VMI //VibrateMatrixIdentity
{
	enum{ ComputeMat = 0, 
		  ComputeInverseMat = 1, 
		  StiffMat = 2, 
		  MassMat = 3, 
		  DampingMat = 4, 
		  ConstraintMat = 5, 
		  BasisUMat = 6, 
		  ModalWarpBasisMat=7, 
		  RhatMat = 8, 
		  BasisUhatMat = 9, 
		  VibrateMatNum = 10 };
	enum{
		MR_computeRhs_Vec = 0,
		R_MR_rhs_Vec = 1,
		R_MR_rhs_externalForce_Vec = 2,
		mass_MR_rhs_Vec = 3,
		damping_MR_rhs_Vec = 4,
		MR_displacement_Vec = 5,
		MR_velocity_Vec = 6,
		MR_acceleration_Vec = 7,
		MR_old_acceleration_Vec = 8,
		MR_old_displacement_Vec = 9,
		MR_incremental_displacement_Vec = 10,
		EigenValuesVector_Vec = 11,
		VibrateVecNum = 12
	};
}

#if USE_PPCG
namespace PPCGI
{
	enum{ G22=0, A=1, A_stiff=2, A_damping=3, A_mass=4, B=5, B1=6, B1_Inv=7, B2=8, C=9,R_hat=10 ,PPCGMatNum=11};
	enum{ d=0, computeRhs=1, R_rhs=2, mass_rhs=3, damping_rhs=4, displacement=5, velocity=6, acceleration=7, 
		old_acceleration = 8, old_displacement = 9, incremental_displacement = 10, R_rhs_externalForce =11, c=12, PPCGVecNum = 13
	};
}
#endif//USE_PPCG

#endif//_MY_MACRO_H