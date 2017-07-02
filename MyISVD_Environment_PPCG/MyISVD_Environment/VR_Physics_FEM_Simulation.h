#pragma once

#include "VR_Global_Define.h"
#include "Cell.h"
#include "VR_Numerical_NewmarkContant.h"

#include "Frame/Mat_YC.h"
#include "Frame/Axis_YC.h"
namespace YC
{

	class VR_Physics_FEM_Simulation
	{
	public:
		struct Physic_State
		{
			int m_nDof;
			int m_nDC_Dofs;
			MySpMat m_computeMatrix,m_global_MassMatrix,m_global_StiffnessMatrix,m_global_DampingMatrix;
			MyVector m_computeRhs, R_rhs,R_rhs_externalForce,R_rhs_distance,R_rhs_distanceForce,mass_rhs,damping_rhs,displacement,velocity,acceleration,old_acceleration,old_displacement, incremental_displacement,incremental_displacement_MidOutput,displacement_newmark,velocity_newmark,acceleration_newmark;
			std::vector< VertexPtr > m_vecDCBoundaryCondition;
			std::vector< VertexPtr > m_vecForceBoundaryCondition;

#if (!USE_SUBSPACE) && (USE_MODAL_WARP)
			MySpMat m_R_hat;
#endif

#if 0
			MySpMat m_global_ModalWarp;
#endif

			void resize()
			{
				const int nDof = m_nDof;
				m_computeMatrix.resize(nDof,nDof);
				m_global_MassMatrix.resize(nDof,nDof);
				m_global_StiffnessMatrix.resize(nDof,nDof);
				m_global_DampingMatrix.resize(nDof,nDof);
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
			MyDenseMatrix m_MR_computeMatrix,m_MR_computeMatrix_Inverse,m_MR_global_MassMatrix,m_MR_global_StiffnessMatrix,m_MR_global_DampingMatrix;
			MyVector m_MR_computeRhs, R_MR_rhs,R_MR_rhs_externalForce,R_MR_rhs_distance,R_MR_rhs_distanceForce,mass_MR_rhs,damping_MR_rhs,MR_displacement,MR_velocity,MR_acceleration,MR_old_acceleration,MR_old_displacement, MR_incremental_displacement,MR_incremental_displacement_MidOutput,MR_displacement_newmark,MR_velocity_newmark,MR_acceleration_newmark;
#if USE_MODAL_WARP
			MyDenseMatrix m_ModalWarp_Basis;
			MySpMat m_R_hat;//block-diagonal rotation matrix
			MyDenseMatrix m_BasisU_hat;
#endif
			void createSubspaceMatrix(const Physic_State& globalState, const float NewMarkConstant_0, const float NewMarkConstant_1)
			{
				m_globalDofs = globalState.m_nDof;

				const int nDof = m_nSubspaceDofs;
				m_MR_computeMatrix.resize(nDof,nDof);
				m_MR_global_MassMatrix.resize(nDof,nDof);
				m_MR_global_StiffnessMatrix.resize(nDof,nDof);
				m_MR_global_DampingMatrix.resize(nDof,nDof);
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
				m_R_hat.resize(m_globalDofs,m_globalDofs);
				m_BasisU_hat = m_BasisU;//initialize
#endif
			}
		};
	public:
		VR_Physics_FEM_Simulation(void);
		~VR_Physics_FEM_Simulation(void);

	public:
		void loadOctreeNode_Global(const int XCount, const int YCount, const int ZCount);
		void distributeDof_global();
		void createDCBoundaryCondition_Global();
		void createForceBoundaryCondition_Global(const int XCount, const int YCount, const int ZCount);
		bool isDCCondition_Global(const MyPoint& pos);
		bool isForceCondition_Global(const MyPoint& pos,const int XCount, const int YCount, const int ZCount);
		void createGlobalMassAndStiffnessAndDampingMatrix_FEM_Global();
		void createNewMarkMatrix_Global();

		void simulationOnCPU_Global(const int nTimeStep);
#if USE_MODAL_WARP
		void simulationOnCPU_Global_WithModalWrap(const int nTimeStep);
		void compute_Local_R_Global(Physic_State& globalState);
		void compute_ModalWrap_Rotation(const MyVector& globalDisplacement, MySpMat& modalWrap_R_hat);
		void update_displacement_ModalWrap(Physic_State& globalState);
#endif
		void update_rhs_Global(const int nStep);
		void apply_boundary_values_Global();
		void setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx );
		void solve_linear_problem_Global();
		void solve_linear_problem_Global_Inverse();
		void update_u_v_a_Global();
		void render_Global();
		void printfMTX(const char* lpszFileName, const MySpMat& sparseMat);
		Physic_State& getGlobalState(){ return m_global_State; }
	private:
		std::vector< CellPtr > m_vec_cell;
		Physic_State m_global_State;
		Numerical::NewmarkContant<MyFloat> m_db_NewMarkConstant;
	public:
		bool m_bSimulation;
		void TestVtxCellId();
#if USE_SUBSPACE
	public:
		void createModalReduction(int nSubspaceModeNum);
		bool LinearModeAnalysis_Global(const int nEigenValueCount, MyFloat * _EigenValueList, MyFloat * _EigenVectorList);
		bool LinearModeAnalysis(const MySpMat& K, const MySpMat& M, const int nDofs, const int nDCDofs, const std::vector< VertexPtr >& BC, const int nModeCount, MyFloat * _EigenValueList, MyFloat * _EigenVectorList);
		void simulationSubspaceOnCPU(const int nTimeStep);
		void update_Subspace_rhs(const int nStep);
		void solve_linear_Subspace_problem();
		void update_Subspace_u_v_a();
		void computeReferenceDisplace();
		
		std::vector< Axis::Quaternion > vecShareCellQuaternion;
#if USE_MODAL_WARP
		void computeReferenceDisplace_ModalWarp(Physic_State& globalState, Vibrate_State& subspaceState);
		void compute_modalWarp_R(Vibrate_State& subspaceState);
		void compute_R_Basis(Vibrate_State& subspaceState);
		void compute_Local_R(Vibrate_State& subspaceState);
		void compute_rhs_ModalWrap(const Physic_State& globalState, Vibrate_State& subspaceState);
		Axis::Quaternion covert_matrix2Quaternion(const MyDenseMatrix& mat);
		void printRotationFrame();
		void compute_CellRotation();
		std::vector< Axis::Quaternion > m_vec_frame_quater;
		std::vector< std::pair< MyDenseVector,Axis::Quaternion  > > m_testCellRotation;
		std::vector< MyMatrix_3X3 > m_vec_VtxLocalRotaM;
		std::vector< MyMatrix_3X3 > m_vec_CelLocalRotaM;
#endif
	private:
		Vibrate_State m_global_subspace_State;
#endif

#if USE_MAKE_CELL_SURFACE
		void creat_Outter_Skin(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs);
		void getSkinDisplacement(const Physic_State& currentState, const YC::MyIntMatrix& matV_dofs, YC::MyMatrix& matU);
		MyFloat xMin, xMax, yMin, yMax, zMin, zMax;
#endif
	};
}

