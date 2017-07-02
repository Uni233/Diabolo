#ifndef _VR_Physics_FEM_Simulation_MultiDomain_h_
#define _VR_Physics_FEM_Simulation_MultiDomain_h_

#include "VR_Global_Define.h"
#include "Cell.h"
#include "VR_Numerical_NewmarkContant.h"
#include "Physic_State.h"
#include <vector>

namespace YC
{
#if USE_MultiDomain
	class VR_Physics_FEM_Simulation_MultiDomain
	{
	public:
		VR_Physics_FEM_Simulation_MultiDomain(void);
		~VR_Physics_FEM_Simulation_MultiDomain(void);
	public:
		bool loadOctreeNode_MultiDomain(const int XCount, const int YCount, const int ZCount);
		void distributeDof_ALM_PPCG();
		void distributeDof_local();
		void createDCBoundaryCondition_ALM();
		bool isDCCondition_ALM(const MyPoint& pos);
		void createGlobalMassAndStiffnessAndDampingMatrix_ALM();
		void printMatrix(const MySpMat& spMat, const std::string& lpszFile);
		void printVector(const MyVector& vec, const std::string& lpszFile);
		void setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx);
		void createNewMarkMatrix_ALM();
		void makeALM_PPCG();
		void makeALM_PPCG_PerVertex();
		void makeALM_New();
		void makeALM_PerVertex();

		void simulation_PPCG();
		void update_rhs_PPCG();
		void apply_boundary_values_PPCG();
		void solve_linear_problem_PPCG();
		void update_u_v_a_PPCG();

		void simulation_ALM();
		void apply_boundary_values_ALM();
		void update_rhs_ALM();
		void update_u_v_a_ALM();
		void solve_linear_problem_ALM();

#if USE_MAKE_CELL_SURFACE
		void creat_Outter_Skin(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs, YC::MyIntVector& vecF_ColorId);
		void getSkinDisplacement(YC::MyVector& displacement, YC::MyMatrix& matV, YC::MyIntMatrix& matV_dofs, YC::MyMatrix& matU);
		MyFloat xMin, xMax, yMin, yMax, zMin, zMax;
#endif
		Physic_PPCG_State& Global_PPCG_State(){ return global_ppcg_state; }
		Physic_State& Global_ALM_State(){ return global_alm_state; }
		bool isLagrangeMultiplierCell(CellPtr curCellPtr);

#if USE_MODAL_WARP
		
		void compute_Local_R_ALM(Physic_PPCG_State& globalState);
		void compute_ModalWrap_Rotation_ALM(const MyVector& globalDisplacement, MySpMat& modalWrap_R_hat);
		void update_displacement_ModalWrap_ALM(Physic_PPCG_State& globalState);
		void solve_linear_Subspace_problem(Vibrate_State& curState);
		void computeReferenceDisplace_ModalWarp(Physic_State& globalState, Vibrate_State& subspaceState);
		void compute_modalWarp_R(Vibrate_State& subspaceState);
		void compute_R_Basis(Vibrate_State& subspaceState);
		void compute_Local_R(Vibrate_State& subspaceState);
		void compute_rhs_ModalWrap(const Physic_State& globalState, Vibrate_State& subspaceState);
		Axis::Quaternion covert_matrix2Quaternion(const MyDenseMatrix& mat);
		void printRotationFrame();
		void compute_CellRotation(Vibrate_State& subspaceState);
		std::vector< Axis::Quaternion > m_vec_frame_quater;
		std::vector< std::pair< MyDenseVector, Axis::Quaternion  > > m_testCellRotation;
		std::vector< MyMatrix_3X3 > m_vec_VtxLocalRotaM;
		std::vector< MyMatrix_3X3 > m_vec_CelLocalRotaM;
		std::vector< Axis::Quaternion > vecShareCellQuaternion;
#endif
	private:		
		Numerical::NewmarkContant<MyFloat> m_db_NewMarkConstant;
		std::vector< CellPtr > m_vec_cell;
		MyInt m_nDof_ALM;
		MyInt m_nDof_Q;
		std::vector< MyInt > m_vecLocalDof;
		std::vector< VertexPtr > m_vecDCBoundaryCondition_ALM;

		Physic_PPCG_State global_ppcg_state;
		Physic_State global_alm_state;
	public:
		bool m_bSimulation;
	};
#endif//USE_MultiDomain
}

#endif//_VR_Physics_FEM_Simulation_MultiDomain_h_