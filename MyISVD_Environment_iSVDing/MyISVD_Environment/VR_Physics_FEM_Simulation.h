#pragma once

#include "VR_Global_Define.h"
#include "Cell.h"
#include "VR_Numerical_NewmarkContant.h"

#include "Frame/Mat_YC.h"
#include "Frame/Axis_YC.h"
#include "Physic_State.h"

#if SingleDomainISVD
#include "MyISVD.h"
#endif//SingleDomainISVD
namespace YC
{

	class VR_Physics_FEM_Simulation
	{
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


#if USE_MODAL_WARP
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
		std::vector< std::pair< MyDenseVector,Axis::Quaternion  > > m_testCellRotation;
		std::vector< MyMatrix_3X3 > m_vec_VtxLocalRotaM;
		std::vector< MyMatrix_3X3 > m_vec_CelLocalRotaM;
		std::vector< Axis::Quaternion > vecShareCellQuaternion;
#endif

#if USE_MAKE_CELL_SURFACE
		void creat_Outter_Skin(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs);
		void getSkinDisplacement(Physic_State& currentState, YC::MyMatrix& matV, YC::MyIntMatrix& matV_dofs, YC::MyMatrix& matU);
		MyFloat xMin, xMax, yMin, yMax, zMin, zMax;
#endif

#if USE_SUBSPACE_Whole_Spectrum
		void create_Whole_Spectrum(const Physic_State& globalState, Vibrate_State& subspaceState);
		Vibrate_State& getWholeSpectrumState(){ return m_global_Whole_Spectrum_State; }
		void simulationWholeSpectrumOnCPU(const int nTimeStep);
	private:
		Vibrate_State m_global_Whole_Spectrum_State;
#endif

#if USE_SUBSPACE_SVD
		enum{ nSampleInterval = 2 };
		enum{ nSVDModeNum = 40 };

#if SingleDomainISVD
	public:
		void init_iSVD_updater(const Physic_State& globalState, MyInt nPoolCapacity);
		void simulation_iSVDOnCPU(const int nTimeStep){}
		void simulation_iSVDOnCPU_Subspace_WithModalWrap(const int nTimeStep){}
		
	private:
		SVD_updater m_iSVD_updater;
#else
		void initSVD(const Physic_State& globalState, MyInt nPoolCapacity);		
		void simulationSVDOnCPU(const int nTimeStep);
		void simulationSVDOnCPU_Subspace_WithModalWrap(const int nTimeStep);
		void appendGlobalDisplace2Pool(const MyVector& global_displacement);
		
	
#endif//SingleDomainISVD
	public:
		Vibrate_State& getSVDState(){ return m_global_SVD_State; }
		bool isSimulationOnSubspace()const { return m_usingSubspaceSimulation; }
	private:
		Vibrate_State m_global_SVD_State;
		bool m_usingSubspaceSimulation;
#endif//USE_SUBSPACE_SVD


	};
}

