#ifndef _VR_PHYSIC_FEM_SIMULATION_H
#define _VR_PHYSIC_FEM_SIMULATION_H

#include "VR_MACRO.h"
#include "VR_Geometry_MeshDataStruct.h"
#include "VR_Physic_Cell.h"
#include "VR_Numerical_NewmarkContant.h"
#include "VR_Geometry_TriangleMeshStruct.h"
#include <vector>
#include "MyVBOLineSet.h"//for AABB
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
				void simulationOnCUDA(const int nTimeStep, unsigned int vao_handle, unsigned int vbo_lines, unsigned int vbo_linesIndex, unsigned int vbo_vertexes, unsigned int vbo_normals, unsigned int vbo_triangle, unsigned int & nTriangeSize, unsigned int& nLineSize);

				int getLineSet();
				
				void registerVBOID(const std::vector< unsigned int >& vecVBOID );

#if USE_Mesh_Cutting
				int getTriangleMesh(unsigned int vao_handle,unsigned int vbo_vertexes, unsigned int vbo_normals, unsigned int vbo_triangle );
				MyVBOLineSet * getAABB();
				int getBladeList(unsigned int vao_handle,unsigned int vbo_vertexes, unsigned int vbo_indexes);
				int getBladeTriangleList(unsigned int vao_handle,unsigned int vbo_vertexes, unsigned int vbo_normals);
				void addBlade(const MyPoint& p0,const MyPoint& p1);
				std::vector< std::pair<MyPoint,MyPoint> > vecBladeList;
#endif
			private:
				void distributeDof();
				void createForceBoundaryCondition();
				bool isForceCondition(const MyPoint& pos);
				bool isDCCondition(const MyPoint& pos);
				void createGlobalMassAndStiffnessAndDampingMatrixFEM();
				void createNewMarkMatrix();
				void createDCBoundaryCondition();
				void createTrilinearWeightForSkinning(const MeshDataStruct& objMeshInfo);
#if USE_CUDA
				void initLocalStructForCUDA();
#endif
				void initSkinningStructForCUDA();
				void makeLineStructForCuda(int ** line_vertex_pair_and_belongDomain/*3 times of the lineCount*/,int& lineCount);
				void freeLineStructForCuda(int ** line_vertex_pair_and_belongDomain/*3 times of the lineCount*/,int& /*lineCount*/);
				void makeMeshStructureOnCuda(int& triangleCount);

				void update_rhs(const int nStep);
				void apply_boundary_values();
				void setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx );
				void solve_linear_problem();
				void compuateDisplacementVertexWithTrilinear();
				void update_u_v_a();
#if USE_CO_RATION
				void assembleRotationSystemMatrix();
#endif

#if MY_DEBUG_OUTPUT_GPU_MATRIX
				void outputGPUMatrix();
				void assembleMatrix(const std::map<long,std::map<long,Cell::TripletNode > >& StiffTripletNodeMap,MySpMat& outputMatrix);
				void compareSparseMatrix( MySpMat& cpuMatrix,  MySpMat& gpuMatrix);
				void printfMTX(const char* lpszFileName, const int nDofs, const std::map<long,std::map<long,Cell::TripletNode > >& StiffTripletNodeMap);
#endif
#if USE_OUTPUT_RENDER_OBJ_MESH
				void outputObjMeshInfoOnCPU();
#endif
#if MY_VIDEO_OUTPUT_OBJ
				void ouputObjMesh4Video(const int nTimeStep,float3 * ptr_vertexes,	float3 * ptr_normals, const int nTriangeSize);
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