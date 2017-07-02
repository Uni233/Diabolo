#ifndef _Physic_State_DomainDependent_h_
#define _Physic_State_DomainDependent_h_

#include "VR_Global_Define.h"
#include "Vertex.h"
#include "MyIterMacro.h"
#if USE_MultidomainIndependent

namespace YC
{
	struct Physic_PPCG_State_Independent;

	struct Physic_State_DomainIndependent
	{
		MyInt m_nGlobalDofsBase;
		MyInt m_nGlobalDofs;
		MyInt m_nLocalDomainDof;
		MyInt m_nGlobal_Dofs_Q;
		MyInt m_nLocalDC_Dofs;
		MyMatrix m_LocalMatrix[PMI::PhysicMatNum];
		MyVector m_LocalVector[PMI::PhysicVecNum];
		std::vector< VertexPtr > m_vecDCBoundaryCondition;
		std::vector< VertexPtr > m_vecForceBoundaryCondition;

		void resize();
		void getConstraintMat(const MyMatrix& globalCMat);
#if USE_PPCG
		void assembleMatrixLocalToGlobal_PPCG(Physic_PPCG_State_Independent & globalState);
#else//USE_PPCG
		void assembleMatrixLocalToGlobal(Physic_State_DomainIndependent& globalState);
#endif//USE_PPCG
		
		void printInfo();
	};

#if USE_PPCG
	struct Physic_PPCG_State_Independent
	{
		MyInt m_nGlobalDofs;		
		MyInt m_nGlobal_Dofs_Q;		
		MyMatrix m_LocalMatrix[PPCGI::PPCGMatNum];
		MyVector m_LocalVector[PPCGI::PPCGVecNum];
		std::vector< VertexPtr > m_vecDCBoundaryCondition;
		std::vector< VertexPtr > m_vecForceBoundaryCondition;
#if USE_PPCG_Permutation
		MyMatrix m_matPermutation;
		MyVector m_vecBoundaryFlag;
#endif
		
		void resize()
		{
			const MyInt n = m_nGlobalDofs;
			const MyInt m = m_nGlobal_Dofs_Q;
			const MyInt n_m = n - m;
			m_LocalMatrix[PPCGI::A].resize(n, n);
			m_LocalMatrix[PPCGI::A_stiff].resize(n, n);
			m_LocalMatrix[PPCGI::A_damping].resize(n, n);
			m_LocalMatrix[PPCGI::A_mass].resize(n, n);
			m_LocalMatrix[PPCGI::G22].resize(n_m, n_m);
			m_LocalMatrix[PPCGI::B].resize(m, n);
			m_LocalMatrix[PPCGI::B1].resize(m, m);
			m_LocalMatrix[PPCGI::B1_Inv].resize(m, m);
			m_LocalMatrix[PPCGI::B2].resize(m, n_m);
			m_LocalMatrix[PPCGI::C].resize(m, m);
			m_LocalMatrix[PPCGI::R_hat].resize(n, n);

			for (int i = 0; i < PPCGI::PPCGMatNum;++i)
			{
				m_LocalMatrix[i].setZero();
			}
			for (int i = 0; i < PPCGI::PPCGVecNum;++i)
			{
				m_LocalVector[i].resize(n);
				m_LocalVector[i].setZero();
			}
			m_LocalVector[PPCGI::d].resize(m);
			m_LocalVector[PPCGI::d].setZero();

		}
	};
#endif//USE_PPCG

	struct Vibrate_State_DomainIndependent
	{
		MyInt m_nPhysicDomainId;
		MyInt m_nLocalSubspaceDofs;

		MyMatrix m_LocalMRMatrix[VMI::VibrateMatNum];
		MyVector m_LocalMRVector[VMI::VibrateVecNum];
		

#if USE_SUBSPACE_SVD
		MyMatrix m_matDisplacePool;
		MyInt m_matDisplacePoolCapacity;
		MyInt m_matDisplacePoolSize;
		int m_nDC_Dofs4SVD;
#endif
		void createSubspaceMatrix(const Physic_State_DomainIndependent& localState, const float NewMarkConstant_0, const float NewMarkConstant_1)
		{
			const MyInt nPhysicDofs = localState.m_nLocalDomainDof;

			const int nDof = m_nLocalSubspaceDofs;

			for (MyInt i = 0; i < VMI::VibrateMatNum; ++i)
			{
				m_LocalMRMatrix[i].resize(nDof,nDof);
				m_LocalMRMatrix[i].setZero();
			}

			for (MyInt i = 0; i < VMI::VibrateVecNum; ++i)
			{
				m_LocalMRVector[i].resize(nDof);
				m_LocalMRVector[i].setZero();
			}

			m_LocalMRMatrix[VMI::MassMat] = m_LocalMRMatrix[VMI::BasisUMat].transpose() * localState.m_LocalMatrix[PMI::MassMat] * m_LocalMRMatrix[VMI::BasisUMat];
			//m_MR_global_MassMatrix = m_BasisU.transpose() * globalState.m_global_MassMatrix * m_BasisU;

			m_LocalMRMatrix[VMI::StiffMat] = m_LocalMRMatrix[VMI::BasisUMat].transpose() * localState.m_LocalMatrix[PMI::StiffMat] * m_LocalMRMatrix[VMI::BasisUMat];
			//m_MR_global_StiffnessMatrix = m_BasisU.transpose() * globalState.m_global_StiffnessMatrix * m_BasisU;

			m_LocalMRMatrix[VMI::DampingMat] = m_LocalMRMatrix[VMI::BasisUMat].transpose() * localState.m_LocalMatrix[PMI::DampingMat] * m_LocalMRMatrix[VMI::BasisUMat];
			//m_MR_global_DampingMatrix = m_BasisU.transpose() * globalState.m_global_DampingMatrix * m_BasisU;

			m_LocalMRVector[VMI::MR_computeRhs_Vec] = m_LocalMRMatrix[VMI::BasisUMat].transpose() *localState.m_LocalVector[PMI::ComputeRhs_Vec];
			//m_MR_computeRhs = m_BasisU.transpose() * globalState.m_computeRhs;
			
			m_LocalMRMatrix[VMI::ComputeMat] = m_LocalMRMatrix[VMI::StiffMat];//m_MR_computeMatrix = m_MR_global_StiffnessMatrix;
			m_LocalMRMatrix[VMI::ComputeMat] += NewMarkConstant_0 * m_LocalMRMatrix[VMI::MassMat];//m_MR_computeMatrix += NewMarkConstant_0 * m_MR_global_MassMatrix;
			m_LocalMRMatrix[VMI::ComputeMat] += NewMarkConstant_1 * m_LocalMRMatrix[VMI::DampingMat];//m_MR_computeMatrix += NewMarkConstant_1 * m_MR_global_DampingMatrix;
			
			computeInverse();// m_MR_computeMatrix_Inverse = m_MR_computeMatrix.inverse();

			m_LocalMRVector[VMI::R_MR_rhs_externalForce_Vec] = m_LocalMRMatrix[VMI::BasisUMat].transpose() * localState.m_LocalVector[PMI::R_rhs_externalForce_Vec];// R_MR_rhs_externalForce = m_BasisU.transpose() * globalState.R_rhs_externalForce*1.f;

#if USE_MODAL_WARP
			m_LocalMRMatrix[VMI::ComputeMat].resize(nPhysicDofs, nPhysicDofs);
			m_LocalMRMatrix[VMI::BasisUhatMat] = m_LocalMRMatrix[VMI::BasisUMat];
#endif
		}

		void computeInverse()
		{
			m_LocalMRMatrix[VMI::ComputeInverseMat] = m_LocalMRMatrix[VMI::ComputeMat].inverse();
		}
	};
}//namespace YC
#else //USE_MultidomainIndependent
namespace YC
{

	struct Physic_State
	{
		int m_nDof;
		int m_nDC_Dofs;
		MySpMat m_computeMatrix, m_global_MassMatrix, m_global_StiffnessMatrix, m_global_DampingMatrix;
		MyVector m_computeRhs, R_rhs, R_rhs_externalForce, R_rhs_distance, R_rhs_distanceForce, mass_rhs, damping_rhs, displacement, velocity, acceleration, old_acceleration, old_displacement, incremental_displacement, incremental_displacement_MidOutput, displacement_newmark, velocity_newmark, acceleration_newmark;
		std::vector< VertexPtr > m_vecDCBoundaryCondition;
		std::vector< VertexPtr > m_vecForceBoundaryCondition;

#if USE_MODAL_WARP
		MySpMat m_R_hat;
#endif

#if 0
		MySpMat m_global_ModalWarp;
#endif

		void resize()
		{
			const int nDof = m_nDof;
			m_computeMatrix.resize(nDof, nDof);
			m_global_MassMatrix.resize(nDof, nDof);
			m_global_StiffnessMatrix.resize(nDof, nDof);
			m_global_DampingMatrix.resize(nDof, nDof);
			m_computeRhs.resize(nDof);

			{
				R_rhs.resize(nDof);
				R_rhs_distance.resize(nDof);
				R_rhs_distanceForce.resize(nDof);
				R_rhs_externalForce.resize(nDof);
				mass_rhs.resize(nDof);
				damping_rhs.resize(nDof);
				displacement.resize(nDof);
				velocity.resize(nDof);
				acceleration.resize(nDof);
#if USE_CO_RATION
				m_RotationRHS.resize(nDof);
#endif

				displacement_newmark.resize(nDof);
				velocity_newmark.resize(nDof);
				acceleration_newmark.resize(nDof);

				old_acceleration.resize(nDof);
				old_displacement.resize(nDof);
				incremental_displacement.resize(nDof);
				incremental_displacement_MidOutput.resize(nDof);

				R_rhs.setZero();
				R_rhs_distance.setZero();
				R_rhs_externalForce.setZero();
				R_rhs_distanceForce.setZero();
				mass_rhs.setZero();
				damping_rhs.setZero();
				displacement.setZero();
				velocity.setZero();
				acceleration.setZero();

#if USE_CO_RATION
				m_RotationRHS.setZero();
#endif

				displacement_newmark.setZero();
				velocity_newmark.setZero();
				acceleration_newmark.setZero();

				old_acceleration.setZero();
				old_displacement.setZero();
				incremental_displacement.setZero();
				incremental_displacement_MidOutput.setZero();
			}

			m_computeMatrix.setZero();
			m_global_MassMatrix.setZero();
			m_global_StiffnessMatrix.setZero();
			m_global_DampingMatrix.setZero();
			m_computeRhs.setZero();
		}
	};
	struct Vibrate_State
	{
		int m_globalDofs;
		int m_nSubspaceDofs;
		MyDenseMatrix m_BasisU;
		MyVector m_EigenValuesVector;
		MyDenseMatrix m_MR_computeMatrix, m_MR_computeMatrix_Inverse, m_MR_global_MassMatrix, m_MR_global_StiffnessMatrix, m_MR_global_DampingMatrix;
		MyVector m_MR_computeRhs, R_MR_rhs, R_MR_rhs_externalForce, R_MR_rhs_distance, R_MR_rhs_distanceForce, mass_MR_rhs, damping_MR_rhs, MR_displacement, MR_velocity, MR_acceleration, MR_old_acceleration, MR_old_displacement, MR_incremental_displacement, MR_incremental_displacement_MidOutput, MR_displacement_newmark, MR_velocity_newmark, MR_acceleration_newmark;
#if USE_MODAL_WARP
		MyDenseMatrix m_ModalWarp_Basis;
		MySpMat m_R_hat;//block-diagonal rotation matrix
		MyDenseMatrix m_BasisU_hat;
#endif

#if USE_SUBSPACE_SVD
		MyMatrix m_matDisplacePool;
		MyInt m_matDisplacePoolCapacity;
		MyInt m_matDisplacePoolSize;
		int m_nDC_Dofs4SVD;
#endif
		void createSubspaceMatrix(const Physic_State& globalState, const float NewMarkConstant_0, const float NewMarkConstant_1)
		{
			m_globalDofs = globalState.m_nDof;

			const int nDof = m_nSubspaceDofs;
			m_MR_computeMatrix.resize(nDof, nDof);
			m_MR_global_MassMatrix.resize(nDof, nDof);
			m_MR_global_StiffnessMatrix.resize(nDof, nDof);
			m_MR_global_DampingMatrix.resize(nDof, nDof);
			m_MR_computeRhs.resize(nDof);

			{
				R_MR_rhs.resize(nDof);
				R_MR_rhs_distance.resize(nDof);
				R_MR_rhs_distanceForce.resize(nDof);
				R_MR_rhs_externalForce.resize(nDof);
				mass_MR_rhs.resize(nDof);
				damping_MR_rhs.resize(nDof);
				MR_displacement.resize(nDof);
				MR_velocity.resize(nDof);
				MR_acceleration.resize(nDof);

				MR_displacement_newmark.resize(nDof);
				MR_velocity_newmark.resize(nDof);
				MR_acceleration_newmark.resize(nDof);

				MR_old_acceleration.resize(nDof);
				MR_old_displacement.resize(nDof);
				MR_incremental_displacement.resize(nDof);
				MR_incremental_displacement_MidOutput.resize(nDof);

				R_MR_rhs.setZero();
				R_MR_rhs_distance.setZero();
				R_MR_rhs_externalForce.setZero();
				R_MR_rhs_distanceForce.setZero();
				mass_MR_rhs.setZero();
				damping_MR_rhs.setZero();
				MR_displacement.setZero();
				MR_velocity.setZero();
				MR_acceleration.setZero();

				MR_displacement_newmark.setZero();
				MR_velocity_newmark.setZero();
				MR_acceleration_newmark.setZero();

				MR_old_acceleration.setZero();
				MR_old_displacement.setZero();
				MR_incremental_displacement.setZero();
				MR_incremental_displacement_MidOutput.setZero();
			}

			m_MR_computeMatrix.setZero();
			m_MR_global_MassMatrix.setZero();
			m_MR_global_StiffnessMatrix.setZero();
			m_MR_global_DampingMatrix.setZero();
			m_MR_computeRhs.setZero();

			m_MR_global_MassMatrix = m_BasisU.transpose() * globalState.m_global_MassMatrix * m_BasisU;

			m_MR_global_StiffnessMatrix = m_BasisU.transpose() * globalState.m_global_StiffnessMatrix * m_BasisU;

			//m_MR_global_DampingMatrix = MyDenseMatrix<MyFloat, 3, 3>::Identity();;//MyDenseMatrix::Identity();
			m_MR_global_DampingMatrix = m_BasisU.transpose() * globalState.m_global_DampingMatrix * m_BasisU;

			m_MR_computeRhs = m_BasisU.transpose() * globalState.m_computeRhs;



			m_MR_computeMatrix = m_MR_global_StiffnessMatrix;
			m_MR_computeMatrix += NewMarkConstant_0 * m_MR_global_MassMatrix;
			m_MR_computeMatrix += NewMarkConstant_1 * m_MR_global_DampingMatrix;
			m_MR_computeMatrix_Inverse = m_MR_computeMatrix.inverse();

			R_MR_rhs_externalForce = m_BasisU.transpose() * globalState.R_rhs_externalForce*1.f;

			/*std::cout << m_MR_global_MassMatrix << std::endl;;
			std::cout << "******************************************" << std::endl;
			std::cout << m_MR_global_StiffnessMatrix << std::endl;MyExit;*/
			//std::cout << m_MR_computeRhs.transpose() << std::endl;MyPause;
			//std::cout << R_MR_rhs_externalForce << std::endl;MyPause;
			/*std::cout << "m_MR_computeMatrix : " << std::endl << m_MR_computeMatrix * tmp;
			MyPause;*/
#if USE_MODAL_WARP
			//m_ModalWarp_Basis = globalState.m_global_ModalWarp * m_BasisU;
			m_R_hat.resize(m_globalDofs, m_globalDofs);
			m_BasisU_hat = m_BasisU;//initialize
#endif
		}
	};
}
#endif//USE_MultidomainIndependent
#endif//_Physic_State_DomainDependent_h_