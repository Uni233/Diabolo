#ifndef _VR_Physics_FEM_Simulation_MultiDomainIndependent_h_
#define _VR_Physics_FEM_Simulation_MultiDomainIndependent_h_

#include "VR_Global_Define.h"
#include "Cell.h"
#include "VR_Numerical_NewmarkContant.h"
#include "Physic_State_DomainDependent.h"
#include <vector>

namespace YC
{
	class VR_Physics_FEM_Simulation_MultiDomainIndependent
	{
	public:
		VR_Physics_FEM_Simulation_MultiDomainIndependent(void);
		~VR_Physics_FEM_Simulation_MultiDomainIndependent(void);

	public:
		bool loadOctreeNode_MultidomainIndependent(const int XCount, const int YCount, const int ZCount);
		void distributeDof_local_Independent();
		void distributeDof_global();
		void createDCBoundaryCondition_Independent();
		void createGlobalMassAndStiffnessAndDampingMatrix_Independent();
		void createNewMarkMatrix_DMI();
		void createConstraintMatrix_DMI();
		void assembleGlobalMatrix_Independent();
#if USE_PPCG_Permutation
		void premutationGlobalMatrix();
#endif
		void TestVtxCellId();
		void printfMTX(const char* lpszFileName, const MyMatrix& sparseMat);
#if USE_PPCG
	public:
		Physic_PPCG_State_Independent& getGlobalState_DMI_PPCG(){return m_globalPPCGDomain;}
		void simulationOnCPU_DMI_Global_PPCG(const int nTimeStep);
	private:
		void update_rhs_DMI_Global_PPCG(const int nStep);
		void apply_boundary_values_DMI_Global_PPCG();
		void solve_linear_problem_DMI_Global_PPCG();

		void update_u_v_a_DMI_Global_PPCG();
		void compute_Local_R_DMI_PPCG(Physic_PPCG_State_Independent& dmiState);
		void compute_ModalWrap_Rotation_DMI_PPCG(const MyVector& globalDisplacement, MyMatrix& modalWrap_R_hat);
		void update_displacement_ModalWrap_DMI_PPCG(Physic_PPCG_State_Independent& dmiState);
#else//USE_PPCG
	public:
		Physic_State_DomainIndependent& getGlobalState_DMI(){ return m_globalPhysicDomain; }
		void simulationOnCPU_DMI_Global(const int nTimeStep);
	private:
		void update_rhs_DMI_Global(const int nStep);
		void apply_boundary_values_DMI_Global();
		void solve_linear_problem_DMI_Global();
		void update_u_v_a_DMI_Global();
		void compute_Local_R_DMI(Physic_State_DomainIndependent& dmiState);
		void compute_ModalWrap_Rotation_DMI(const MyVector& globalDisplacement, MyMatrix& modalWrap_R_hat);
		void update_displacement_ModalWrap_DMI(Physic_State_DomainIndependent& dmiState);
#endif//USE_PPCG
		
#if USE_MAKE_CELL_SURFACE
	public:
		void creat_Outter_Skin(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs, YC::MyIntVector& vecF_ColorId);
		void getSkinDisplacement(YC::MyVector& displacement, YC::MyMatrix& matV, YC::MyIntMatrix& matV_dofs, YC::MyMatrix& matU);
		MyFloat xMin, xMax, yMin, yMax, zMin, zMax;
#endif
		
	private:
		
		void setMatrixRowZeroWithoutDiag(MyMatrix& matrix, const int nDofs, const int  rowIdx);
		
	private:
		bool isLagrangeMultiplierCell(CellPtr curCellPtr);
		bool isDCCondition_DMI(const MyPoint& pos);
		void setFromTriplets(MyMatrix &mat, const std::map<long, std::map<long, Cell::TripletNode > >& TripletNodeMap);
		void setFromTriplets(MyVector & vec, const std::map<long, Cell::TripletNode >& RhsTripletNode);
	private:
		bool m_bSimulation;
		std::vector< CellPtr > m_vec_cell;
		Physic_State_DomainIndependent m_localPhysicDomain[LocalDomainCount];
#if USE_PPCG
		Physic_PPCG_State_Independent m_globalPPCGDomain;
#else//USE_PPCG
		Physic_State_DomainIndependent m_globalPhysicDomain;
#endif//USE_PPCG
		
		std::vector< CellPtr > m_vecLocalCellPool[LocalDomainCount];
		MyInt m_nLocalDofs[LocalDomainCount];
		MyInt m_nGlobalDofs;
		MyInt m_nDof_Q;
		Numerical::NewmarkContant<MyFloat> m_db_NewMarkConstant;
	};
}
#endif//_VR_Physics_FEM_Simulation_MultiDomainIndependent_h_