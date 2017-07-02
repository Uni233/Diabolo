#ifndef _ElasticSimulation_Couple_H
#define _ElasticSimulation_Couple_H
#include "ElasticSimulation.h"
#include "MeshParser_Obj/objParser.h"
#include "Cell.h"
#include "ForceCoupleNode.h"

namespace VR_FEM
{
	class ElasticSimulation_Couple : public ElasticSimulation
	{
	public:
		ElasticSimulation_Couple();

	public:
		bool parserObj(bool hasCoord,bool hasVerticeNormal,const char* lpszMeshPath);
		bool loadOctreeNode(const char* lpszModelType);
		bool loadOctreeNode_Beam(const char* lpszModelType);
		bool LoadGLTextures(const char* lpszTexturePath);
		void generateCoupleDomain();
		void distributeDof_global();
		void distributeDof_local();
		void distributeDof_Couple();
		void createGlobalMassAndStiffnessAndDampingMatrixFEM(const int DomainId);
		void createGlobalMassAndStiffnessAndDampingMatrixEFG(const int DomainId);
		void createNewMarkMatrix(const int DomainId);
		void createNewMarkMatrix_Couple_EFG(const int DomainId);
		void createDCBoundaryCondition(const int DomainId);
		void createForceBoundaryCondition(const int DomainId);
		void createOutBoundary();
		void createOutBoundary_Force();
		void createInnerBoundary();
		void createInnerBoundary_Force();
		void simulation();
		void simulation_Couple_EFG();
		void simulation_Couple_FEM_EFG();
		void print();
		void pushMatrix();
		void popMatrix();
	private:
		void generateNewMarkParam();
		void apply_boundary_values(const MyInt DomainId);
		void update_rhs(MyInt nStep,MyInt id);
		void update_rhs_forNewMark(MyInt nStep,MyInt id);
		void update_rhs_inertia(MyInt id);
		void update_u_v_a(MyInt id);
		void update_u_v_a_forNermark(MyInt id);
		void solve_linear_problem(MyInt id);
		bool isDCCondition(const int DomainId,const MyPoint& pos);
		bool isForceCondition(const int DomainId,const MyPoint& pos);
		void setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx );

		void update_rhs_Couple_EFG(MyInt id);
		void update_u_v_a_Couple_EFG(MyInt id);
		void solve_linear_problem_Couple_EFG(MyInt id);

		void getOutBoundaryValue();
		void getOutBoundaryValue_Force();
		void applyOutBoundaryValueToInner();
		
		void getInnerBoundaryValue();
		void getInnerBoundaryValueForce();
		void applyInnerBoundaryValueToOut();
		void applyInnerBoundaryValueToOutForce();
		void applyInnerForceValueToOut();
	private:
		
		ObjParserData m_obj_data;
		MyFloat m_db_NewMarkConstant[8];
		MyFloat m_2_div_timeX2;
		GLuint m_texture;
		std::vector< CellPtr > m_vec_cell;

		MyInt m_nGlobalDof;
		std::vector< MyInt > m_vecLocalDof;
		std::vector< MyInt > m_vecCoupleDof;

		MySpMat m_computeMatrix[Cell::LocalDomainCount],m_computeMatrix_backup[Cell::LocalDomainCount],m_global_MassMatrix[Cell::LocalDomainCount],m_global_StiffnessMatrix[Cell::LocalDomainCount],m_global_DampingMatrix[Cell::LocalDomainCount];
		MyVector m_computeRhs[Cell::LocalDomainCount], R_rhs[Cell::LocalDomainCount],R_rhs_externalForce[Cell::LocalDomainCount],R_rhs_distance[Cell::LocalDomainCount],R_rhs_distanceForce[Cell::LocalDomainCount],mass_rhs[Cell::LocalDomainCount],damping_rhs[Cell::LocalDomainCount],displacement[Cell::LocalDomainCount],velocity[Cell::LocalDomainCount],acceleration[Cell::LocalDomainCount],old_acceleration[Cell::LocalDomainCount],old_displacement[Cell::LocalDomainCount], incremental_displacement[Cell::LocalDomainCount],displacement_newmark[Cell::LocalDomainCount],velocity_newmark[Cell::LocalDomainCount],acceleration_newmark[Cell::LocalDomainCount];

		MySpMat m_computeMatrix_EFG[Cell::CoupleDomainCount],m_global_MassMatrix_EFG[Cell::CoupleDomainCount],m_global_StiffnessMatrix_EFG[Cell::CoupleDomainCount],m_global_DampingMatrix_EFG[Cell::CoupleDomainCount];
		MyVector m_computeRhs_EFG[Cell::CoupleDomainCount], R_rhs_EFG[Cell::CoupleDomainCount],mass_rhs_EFG[Cell::CoupleDomainCount],damping_rhs_EFG[Cell::CoupleDomainCount],displacement_EFG[Cell::CoupleDomainCount],velocity_EFG[Cell::CoupleDomainCount],acceleration_EFG[Cell::CoupleDomainCount],old_acceleration_EFG[Cell::CoupleDomainCount],old_displacement_EFG[Cell::CoupleDomainCount], incremental_displacement_EFG[Cell::CoupleDomainCount];
		
		std::vector< VertexPtr > m_vecDCBoundaryCondition[Cell::LocalDomainCount];
		std::vector< VertexPtr > m_vecForceBoundaryCondition[Cell::LocalDomainCount];

		std::vector< MyVectorI > m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId;
		std::vector< std::pair< MyVectorI,MyDenseVector >   > m_vecOutBoundaryDof2Value;
		std::vector< MyVectorI > m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId,m_vecInnerBoundary_Force_VtxId_FEMDomainId_EFGDomainId;
		std::vector< std::pair< MyVectorI,MyDenseVector >   > m_vecInnerBoundaryDof2Value;

	public:
		bool m_isSimulate;
		void setSimulate(bool f){m_isSimulate = f;}
		void computeCoupleForceForNewmark();
	private:
		std::vector< ForceCoupleNode > m_vecForceCoupleNode;
	};
}
#endif//_ElasticSimulation_Couple_H