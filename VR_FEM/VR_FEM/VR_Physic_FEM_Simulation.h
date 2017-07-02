#ifndef _VR_PHYSIC_FEM_SIMULATION_H
#define _VR_PHYSIC_FEM_SIMULATION_H

#include "VR_MACRO.h"
#include "VR_Geometry_MeshDataStruct.h"
#include "VR_Physic_Cell.h"
#include "VR_Numerical_NewmarkContant.h"
#include "VR_Geometry_TriangleMeshStruct.h"
#include <vector>

#include <vector_types.h>//float3
namespace YC
{
	namespace Physics
	{
		namespace GPU
		{
			using namespace Geometry;
			class VR_Physics_FEM_Simulation
			{
			public:
				VR_Physics_FEM_Simulation();
				virtual ~VR_Physics_FEM_Simulation();

				int initialize(const float backgroundGrid[][4], const int nCellSize, const MeshDataStruct& objMeshInfo);
				void simulationOnCPU(const int nTimeStep);	
				MeshDataStruct& getObjMesh(){return m_objMesh;}
				int getCellSize()const{return m_vec_cell.size();}
				void generateDisplaceLineSet(float3** posPtr,int2 ** indexPtr);
			private:
				void distributeDof();
				void createForceBoundaryCondition();
				bool isForceCondition(const MyPoint& pos);
				bool isDCCondition(const MyPoint& pos);
				void createGlobalMassAndStiffnessAndDampingMatrixFEM();
				void createNewMarkMatrix();
				void createDCBoundaryCondition();
				void createTrilinearWeightForSkinning(const MeshDataStruct& objMeshInfo);
				void update_rhs(const int nStep);
				void apply_boundary_values();
				void setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx );
				void solve_linear_problem();
				void compuateDisplacementVertexWithTrilinear();
				void update_u_v_a();
#if USE_CO_RATION
				void assembleRotationSystemMatrix();
#endif
			private:
				Numerical::NewmarkContant<float> m_db_NewMarkConstant;
				std::vector< CellPtr > m_vec_cell;
				int m_nGlobalDof;
				std::vector< VertexPtr > m_vecDCBoundaryCondition;
				std::vector< VertexPtr > m_vecForceBoundaryCondition;

				MySpMat m_computeMatrix,m_global_MassMatrix,m_global_StiffnessMatrix,m_global_DampingMatrix;
				MyVector m_computeRhs, R_rhs,R_rhs_externalForce,R_rhs_distance,R_rhs_distanceForce,mass_rhs,damping_rhs,displacement,velocity,acceleration,old_acceleration,old_displacement, incremental_displacement,incremental_displacement_MidOutput,displacement_newmark,velocity_newmark,acceleration_newmark;

				std::vector<Geometry::TriangleMeshNode > m_vec_vertexIdx2NodeIdxInside;

				MeshDataStruct m_objMesh;
#if USE_CO_RATION
				MyVector m_RotationRHS;
#endif
			};
		}//namespace GPU
	}//namespace Physics
}//namespace YC
#endif//_VR_PHYSIC_FEM_SIMULATION_H