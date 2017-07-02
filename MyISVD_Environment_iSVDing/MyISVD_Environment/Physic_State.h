#ifndef _Physic_State_h_
#define _Physic_State_h_

#include "VR_Global_Define.h"
#include "Vertex.h"
namespace YC
{
#if USE_PPCG
	struct Physic_PPCG_State
	{
		MyInt nDof_ALM;
		MyInt nDof_Q;
		MySpMat m_sp_ppcg_P;
		MySpMat m_sp_ppcg_G, m_sp_ppcg_G22;
		MySpMat m_sp_ppcg_A;
		MySpMat m_sp_ppcg_A_stiff;
		MySpMat m_sp_ppcg_A_damping;
		MySpMat m_sp_ppcg_A_mass;
		MySpMat m_sp_ppcg_B, m_sp_ppcg_B1, m_sp_ppcg_B2, m_sp_ppcg_B1_Inv;
		MySpMat m_sp_ppcg_C;
		MyVector m_ppcg_d;
		MyVector m_computeRhs_PPCG, R_rhs_PPCG, mass_rhs_PPCG, damping_rhs_PPCG, displacement_PPCG, velocity_PPCG, acceleration_PPCG, old_acceleration_PPCG, old_displacement_PPCG, incremental_displacement_PPCG;

#if USE_MODAL_WARP
		MySpMat m_R_hat;
#endif
		void resize()
		{
			m_sp_ppcg_A.resize(nDof_ALM, nDof_ALM);
			m_sp_ppcg_A_stiff.resize(nDof_ALM, nDof_ALM);
			m_sp_ppcg_A_damping.resize(nDof_ALM, nDof_ALM);
			m_sp_ppcg_A_mass.resize(nDof_ALM, nDof_ALM);
			//m_ppcg_c.resize(nDof_ALM);
			m_ppcg_d.resize(nDof_Q);

			m_computeRhs_PPCG.resize(nDof_ALM);
			R_rhs_PPCG.resize(nDof_ALM);
			mass_rhs_PPCG.resize(nDof_ALM);
			damping_rhs_PPCG.resize(nDof_ALM);
			displacement_PPCG.resize(nDof_ALM);
			velocity_PPCG.resize(nDof_ALM);
			acceleration_PPCG.resize(nDof_ALM);
			old_acceleration_PPCG.resize(nDof_ALM);
			old_displacement_PPCG.resize(nDof_ALM);
			incremental_displacement_PPCG.resize(nDof_ALM);

			m_sp_ppcg_A.setZero();
			m_sp_ppcg_A_stiff.setZero();
			m_sp_ppcg_A_damping.setZero();
			m_sp_ppcg_A_mass.setZero();
			m_ppcg_d.setZero();

			m_computeRhs_PPCG.setZero();
			R_rhs_PPCG.setZero();
			mass_rhs_PPCG.setZero();
			damping_rhs_PPCG.setZero();
			displacement_PPCG.setZero();
			velocity_PPCG.setZero();
			acceleration_PPCG.setZero();
			old_acceleration_PPCG.setZero();
			old_displacement_PPCG.setZero();
			incremental_displacement_PPCG.setZero();
		}
	};
#endif//USE_PPCG
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
#endif//_Physic_State_h_