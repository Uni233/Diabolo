#include "cookbookogl.h"
#include <windows.h>  //Windows Header
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "VR_Physic_FEM_Simulation.h"
#include <math.h>
#include <cmath>
#include "VR_Material.h"
#include "VR_GlobalVariable.h"
#include "VR_Geometry_TriangleMeshStruct.h"
#include "VR_Global_Define.h"
#include "CG/solvercontrol.h"
#include "CG/solvercg.h"
#include "CG/preconditionssor.h"
#include "VR_Render_ConsoleProgressPrint.h"
#include <fstream>
#include "VR_GPU_Physic_StructInfo.h"
#include "VR_GPU_Utils.h"
#include <map>
#include "VR_Physic_CuttingCheckStructOnCuda.h"
//using namespace YC::GlobalVariable;
using namespace YC::Geometry::GPU;



extern int YC::GlobalVariable::g_bcMaxCount;
extern int YC::GlobalVariable::g_bcMinCount;
extern float YC::GlobalVariable::g_scriptForceFactor;
extern float YC::GlobalVariable::g_externalForceFactor;

int cuda_bcMaxCount;
int cuda_bcMinCount;
float cuda_scriptForceFactor;
int g_nNativeSurfaceVertexCount=0;

extern void createVertex2CellForSkinning();
extern int initCudaStructForSkinningTriLinearWeight(const int nVtxSize, const int nCellSize);
extern void freeCudaStructForSkinningTriLinearWeight();
extern MeshVertex2CellInfo * g_cpu_triangleMeshVertex;
extern Cell2MeshVertexInfo * g_cpu_CellCenterPoint;
extern YC::Geometry::TriangleMeshNode* g_cpu_TriangleMeshNode;

extern std::string getTimeStamp();

namespace CUDA_SIMULATION
{	
	extern void VR_Physics_FEM_Simulation_InitLocalCellMatrixOnCuda(const int nCount,float * localStiffnessMatrixOnCpu,	float * localMassMatrixOnCpu,float * localRhsVectorOnCpu,	FEMShapeValue* femShapeValuePtr);
	extern void  VR_Physics_FEM_Simulation_InitPhysicsCellAndVertex(const int vertexSize, VertexOnCuda** retVertexPtr, const int cellSize, CommonCellOnCuda** retCellPtr);
	extern void  VR_Physics_FEM_Simulation_InitCellCuttingLineSetOnCuda(const int linesCount,CuttingLinePair * linesPtrOnCpu);
	extern void VR_Physics_FEM_Simulation_InitLinePair();
	extern void VR_Physics_FEM_Simulation_InitialLocalDomainOnCuda(const int nDofs, float dbNewMarkConstant[8],	int* boundaryconditionPtr,	int bcCount,
		int * boundaryconditionDisplacementPtr,	int bcCountDisplace	);
	extern void makeGlobalIndexPara( float YoungModulus, float PossionRatio, float Density, float *externForce);
	extern void assembleSystemOnCuda_FEM_RealTime();
	extern void assembleSystemOnCuda_FEM_RealTime_MatrixInitDiag();
	extern void initBoundaryCondition();
	extern void do_loop_single(const int nTimeStep,float3** pos_Lines, int2 ** index_Lines, float3** pos_Triangles, float3** vertexNormal, int3 ** index_Triangles, unsigned int& nTrianleSize, unsigned int& nLineSize);
	
	namespace CUDA_SKNNING_CUTTING
	{
		extern VBOStructForDraw g_VBO_Struct_Node;
		extern void initVBOStructContext();
		extern void initVBODataStruct_LineSet(int* line_vertex_pair,int lineCount);
		extern void initMeshCuttingStructure(const int nVertexSize,MC_Vertex_Cuda* MCVertexCuda,
												const int nEdgeSize,  MC_Edge_Cuda*	 MCEdgeCuda,
												const int nSurfaceSize,MC_Surface_Cuda*	MCSurfaceCuda,
												const int nVertexNormal,float * elementVertexNormal);

#if USE_OUTPUT_RENDER_OBJ_MESH
		void getObjMeshInfoFromCUDA(int & nVertexCount,MC_Vertex_Cuda** curVertexSet,int& nTriSize,MC_Surface_Cuda** curFaceSet,int& nCellCount,CommonCellOnCuda** CellOnCudaPtr,float** displacement);
		void freeObjMeshInfoFromCUDA(int & nVertexCount,MC_Vertex_Cuda** curVertexSet,int& nTriSize,MC_Surface_Cuda** curFaceSet,int& nCellCount,CommonCellOnCuda** CellOnCudaPtr,float** displacement);
#endif
		extern int cuda_SkinningCutting_GetTriangle(float3** cpuVertexPtr, float3** cpuNormalsPtr);
		extern void cuda_SkinningCutting_SetBladeList(float3** ptrVertex, float3 * cpuVertex, int2 ** ptrIndex, int2 * cpuIndex, const int nLineSize);
		extern void cuda_SkinningCutting_SetBladeTriangleList(float3** ptrVertex, float3 * cpuVertex, float3 ** ptrNormal, float3 * cpuNormal, const int nTriSize);
	}
	namespace CUDA_DEBUG
	{
		extern void cuda_OuputObjMesh4Video(float3 ** cpu_vertex, float3 ** cpu_normal,float3 ** cuda_vertex, float3 ** cuda_normal,const int nVtxSize,MC_Vertex_Cuda** tmp_Vertex, MC_Surface_Cuda ** tmp_triangle);
		extern void cuda_Debug_Get_MatrixData(int & nDofs,
			int ** systemInnerIndexPtr, float ** systemValuePtr,
			int ** stiffInnerIndexPtr, float ** stiffValuePtr,
			int ** massInnerIndexPtr, float ** massValuePtr,
			float ** rhsValuePtr);
		extern void cuda_Debug_free_MatrixData(int ** systemInnerIndexPtr, float ** systemValuePtr,
			int ** stiffInnerIndexPtr, float ** stiffValuePtr,
			int ** massInnerIndexPtr, float ** massValuePtr,
			float ** rhsValuePtr);
		extern int3 * g_cpu_faces;
		extern float3 * g_cpu_vertexes;
		extern float3 * g_cpu_normals;
		extern int tmp_cuda_setSkinData(float3** cpuVertexPtr, float3** cpuNormalsPtr, int3** cpuFacesPtr);
		extern int tmp_cuda_freeSkinData();
		extern int tmp_cuda_initSkinData(const int vertexSize, const int normalSize, const int faceSize);
	}

	namespace CUDA_CUTTING_GRID
	{
		extern bool g_needCheckCutting;
		extern int nouseRHSSize;
		extern float nouseRHSVec[16083 * 2];
		extern float nouseCusp_Array_Rhs[16083 * 2];
		extern float nouseCusp_Array_Old_Acceleration[16083 * 2];
		extern float nouseCusp_Array_Old_Displacement[16083 * 2];
		extern float nouseCusp_Array_R_rhs_Corotaion[16083 * 2];
		extern void debug_ShowDofs();
	}
}

extern void func();
//using namespace CUDA_SIMULATION;
namespace YC
{
	namespace Physics
	{
		namespace GPU
		{
			VR_Physics_FEM_Simulation::VR_Physics_FEM_Simulation(){}
			VR_Physics_FEM_Simulation::~VR_Physics_FEM_Simulation(){}
			int VR_Physics_FEM_Simulation::initialize(const float backgroundGrid[][4], const int nCellCount, const MeshDataStruct& objMeshInfo)
			{
				{
					cuda_bcMaxCount = YC::GlobalVariable::g_bcMaxCount;
					cuda_bcMinCount = YC::GlobalVariable::g_bcMinCount;
					cuda_scriptForceFactor = YC::GlobalVariable::g_scriptForceFactor;
				}

				m_objMesh = objMeshInfo;

				for (MyInt c=0;c<nCellCount;++c)
				{
					const float * curCellPtr = &backgroundGrid[c][0];

					std::vector< CellPtr > & refCurrentCellVec = m_vec_cell;

					refCurrentCellVec.push_back(Cell::makeCell(MyPoint(curCellPtr[0]+ModelStep,curCellPtr[1]+ModelStep,curCellPtr[2]+ModelStep),curCellPtr[3]));
					refCurrentCellVec[refCurrentCellVec.size()-1]->computeCellType();
					refCurrentCellVec[refCurrentCellVec.size()-1]->computeCell_NeedBeCutting();

					//const int nLevel = ((int)numbers::logN(2,(1.0/curCellPtr[3])))-1;
					refCurrentCellVec[refCurrentCellVec.size()-1]->setLevel(((int)numbers::logN(2,(1.0/curCellPtr[3])))-1);

					Render::progressBar("Make Cell",((float)c*100)/(float)nCellCount,100);
				}
				printf("cell size %d, vertex size %d\n",Cell::getCellSize(),Vertex::getVertexSize());
				//MyPause;
				distributeDof();

				createForceBoundaryCondition();
				createGlobalMassAndStiffnessAndDampingMatrixFEM();
				createNewMarkMatrix();
				createDCBoundaryCondition();

				createTrilinearWeightForSkinning(m_objMesh);


#if USE_CUDA				
				initLocalStructForCUDA();
				initSkinningStructForCUDA();
#endif
				//MyPause;
				return 0;
			}

			void VR_Physics_FEM_Simulation::distributeDof()
			{
				m_nGlobalDof = Geometry::first_dof_idx;
				const MyInt nCellSize = Cell::getCellSize();
				for (MyInt c=0;c<nCellSize;++c)
				{
					CellPtr curCellPtr = Cell::getCell(c);
					for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
					{
						VertexPtr curVtxPtr = curCellPtr->getVertex(v);
						if ( !(curVtxPtr->isValidGlobalDof()))
						{
							curVtxPtr->setDof(m_nGlobalDof,m_nGlobalDof+1,m_nGlobalDof+2);
							m_nGlobalDof+=3;
						}
					}
				}
				printf("global dof %d.\n",m_nGlobalDof);
			}

			bool VR_Physics_FEM_Simulation::isForceCondition(const MyPoint& pos)
			{
				if (pos[0] < (-0.3f+ModelStep) && pos[1] > (0.3f)+ModelStep)
				{
					return true;
				}
				else
				{
					return false;
				}
			}

			void VR_Physics_FEM_Simulation::createForceBoundaryCondition()
			{
				std::vector< CellPtr >& refCellVec = m_vec_cell;
				const MyInt nCellSize = refCellVec.size();
				std::map<MyInt,bool> mapNouse;
				for (MyInt c=0;c<nCellSize;++c)
				{
					CellPtr curCellPtr = refCellVec[c];
					for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
					{
						VertexPtr curVtxPtr = curCellPtr->getVertex(v);
						const MyPoint & pos = curVtxPtr->getPos();
						if (isForceCondition(pos) )
						{
							mapNouse[curVtxPtr->getId()] = true;

						}
					}
				}

				std::map<MyInt,bool>::const_iterator ci = mapNouse.begin();
				std::map<MyInt,bool>::const_iterator endc = mapNouse.end();
				for (;ci != endc; ++ci)
				{
					m_vecForceBoundaryCondition.push_back(Vertex::getVertex(ci->first));
				}

				printf("Force Boundary Condition %d.\n",m_vecForceBoundaryCondition.size());
				//MyPause;
			}

			void VR_Physics_FEM_Simulation::createGlobalMassAndStiffnessAndDampingMatrixFEM()
			{
				const int nDof = m_nGlobalDof;
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



				std::map<long,std::map<long,Cell::TripletNode > >& StiffTripletNodeMap = Cell::m_TripletNode_Stiffness;
				std::map<long,std::map<long,Cell::TripletNode > >& MassTripletNodeMap = Cell::m_TripletNode_Mass;
				std::map<long,Cell::TripletNode >& RhsTripletNode = Cell::m_TripletNode_Rhs;
				StiffTripletNodeMap.clear();
				MassTripletNodeMap.clear();
				RhsTripletNode.clear();

				std::vector< CellPtr > & curCellVec = m_vec_cell;
				for (unsigned i=0;i<curCellVec.size();++i)
				{
					CellPtr curCellPtr = curCellVec[i];

					curCellPtr->initialize();
					curCellPtr->assembleSystemMatrix();

					Render::progressBar("Initialize Cell",((float)i*100)/(float)curCellVec.size(),100);
				}

				
				//assemble global stiffness matrix
				std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
				std::map<long,std::map<long,Cell::TripletNode > >::const_iterator itr_tri =  StiffTripletNodeMap.begin();
				for (;itr_tri != StiffTripletNodeMap.end();++itr_tri)
				{
					const std::map<long,Cell::TripletNode >&  ref_map = itr_tri->second;
					std::map<long,Cell::TripletNode >::const_iterator itr_2 =  ref_map.begin();
					for (;itr_2 != ref_map.end();++itr_2)
					{
						vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
					}
				}
				if (vec_triplet.size() > 0)
				{
					m_global_StiffnessMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
					StiffTripletNodeMap.clear();
					vec_triplet.clear();
				}


				//assemble global mass matrix

				itr_tri =  MassTripletNodeMap.begin();
				for (;itr_tri != MassTripletNodeMap.end();++itr_tri)
				{
					const std::map<long,Cell::TripletNode >&  ref_map = itr_tri->second;
					std::map<long,Cell::TripletNode >::const_iterator itr_2 =  ref_map.begin();
					for (;itr_2 != ref_map.end();++itr_2)
					{
						vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
					}
				}
				if (vec_triplet.size() > 0)
				{
					m_global_MassMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
					MassTripletNodeMap.clear();
					vec_triplet.clear();
				}


				//assemble global rhs vector
				std::map<long,Cell::TripletNode >::const_iterator itrRhs = RhsTripletNode.begin();
				m_computeRhs.setZero();
				for (;itrRhs != RhsTripletNode.end();++itrRhs)
				{
					m_computeRhs[itrRhs->first] = (itrRhs->second).val;
				}
				RhsTripletNode.clear();

				std::vector< VertexPtr >& refVec = m_vecForceBoundaryCondition;
				MyVector& refRhs = R_rhs_externalForce;
				for (MyInt v=0;v<refVec.size();++v)
				{
					MyInt curDof = refVec[v]->getDofs().y();
					refRhs[curDof] = YC::GlobalVariable::g_externalForceFactor * Material::GravityFactor*(-1.f);
					/*MyInt curDof = refVec[v]->getLocalDof(c).x();
					refRhs[curDof] = Material::Density * Material::GravityFactor*8*2*(0.0001f);*/
				}

				m_global_DampingMatrix = Material::damping_alpha * m_global_MassMatrix + Material::damping_beta * m_global_StiffnessMatrix;
			}

			void VR_Physics_FEM_Simulation::createNewMarkMatrix()
			{
				m_computeMatrix = m_global_StiffnessMatrix;
				m_computeMatrix += m_db_NewMarkConstant[0] * m_global_MassMatrix;
				m_computeMatrix += m_db_NewMarkConstant[1] * m_global_DampingMatrix;
				//MyPause;
				/*LogInfo("matrix(%d,%d)\n",m_global_StiffnessMatrix.rows(),m_global_StiffnessMatrix.cols());
				MyPause;
				std::ofstream outfile_stiff("d:\\shader_stiff.txt");
				std::ofstream outfile_mass("d:\\shader_mass.txt");
				std::ofstream outfile_damp("d:\\shader_damp.txt");
				outfile_stiff << m_global_StiffnessMatrix;
				outfile_mass << m_global_MassMatrix;
				outfile_damp << m_global_DampingMatrix;
				outfile_stiff.close();
				outfile_mass.close();
				outfile_damp.close();*/

				incremental_displacement.setZero();
				velocity.setZero();
				acceleration.setZero();
				displacement.setZero();
				old_acceleration.setZero();
				old_displacement.setZero();
				return ;
			}

			bool VR_Physics_FEM_Simulation::isDCCondition(const MyPoint& pos)
			{
				if (pos[1] < (0.1f-0.5f) )
				{
					return true;
				}
				else
				{
					return false;
				}
			}

			void VR_Physics_FEM_Simulation::createDCBoundaryCondition()
			{
				std::vector< CellPtr >& refCellVec = m_vec_cell;
				const MyInt nCellSize = refCellVec.size();
				std::map<MyInt,bool> mapNouse;
				for (MyInt c=0;c<nCellSize;++c)
				{
					CellPtr curCellPtr = refCellVec[c];
					{
						for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
						{
							VertexPtr curVtxPtr = curCellPtr->getVertex(v);
							const MyPoint pos = curVtxPtr->getPos();
							if (isDCCondition(pos) )
							{
								//m_vecDCBoundaryCondition[DomainId].push_back(curVtxPtr);
								mapNouse[curVtxPtr->getId()] = true;
							}
						}
					}
				}

				std::map<MyInt,bool>::const_iterator ci = mapNouse.begin();
				std::map<MyInt,bool>::const_iterator endc = mapNouse.end();
				for (;ci != endc; ++ci)
				{
					m_vecDCBoundaryCondition.push_back(Vertex::getVertex(ci->first));
				}

				printf("Boundary Condition %d. [%d]\n",m_vecDCBoundaryCondition.size(),mapNouse.size());
				Q_ASSERT(m_vecDCBoundaryCondition.size() > 50);
				//MyPause;
			}

			void VR_Physics_FEM_Simulation::createTrilinearWeightForSkinning(const MeshDataStruct& obj_data)
			{
#if (!USE_CUDA)
				std::vector<Geometry::TriangleMeshNode >& vec_vertexIdx2NodeIdxInside = m_vec_vertexIdx2NodeIdxInside;
				vec_vertexIdx2NodeIdxInside.clear();
				float tmpFloat;
				MyDenseVector tmpVec;
				float weight[8];

				const std::vector< vec3 >& ref_vertices = obj_data.points;
				
				for (int vi=0;vi < ref_vertices.size();++vi)
				{
					float minDistance = 1000.f;
					int   curNodeIdx = -1;
					bool  bInside = false;

					MyPoint vertices = MyPoint(ref_vertices[vi].x,ref_vertices[vi].y,ref_vertices[vi].z);
					//for (MyInt domainIdx=0;domainIdx<Cell::LocalDomainCount;++domainIdx)
					{
						std::vector< CellPtr >& refCellPool = m_vec_cell;
						for (int vj=0;vj < refCellPool.size();++vj )
						{
							tmpVec = vertices - refCellPool[vj]->getCenterPoint();
							tmpFloat = tmpVec.norm();
							if (minDistance > tmpFloat)
							{
								curNodeIdx = refCellPool[vj]->getID();
								minDistance = tmpFloat;
								if ( (refCellPool[vj]->getRadius()) < numbers::max3(std::fabs(tmpVec[0]),std::fabs(tmpVec[1]),std::fabs(tmpVec[2])) )
								{
									bInside = false;
								}
								else
								{
									bInside = true;
								}
							}
						}
					}

					TriangleMeshNode  refNode;// = vec_vertexIdx2NodeIdxInside[vec_vertexIdx2NodeIdxInside.size()-1];
					refNode.nBelongCellId = curNodeIdx;

					MyDenseVector p0 = Cell::getCell(curNodeIdx)->getVertex(0)->getPos();
					MyDenseVector p7 = Cell::getCell(curNodeIdx)->getVertex(7)->getPos();
					float detaX = (p7[0] - ref_vertices[vi][0]) / (p7[0] - p0[0]);
					float detaY = (p7[1] - ref_vertices[vi][1]) / (p7[1] - p0[1]);
					float detaZ = (p7[2] - ref_vertices[vi][2]) / (p7[2] - p0[2]);

					refNode.m_TriLinearWeight[0] = detaX * detaY * detaZ;
					refNode.m_TriLinearWeight[1] = (1-detaX) * detaY * detaZ;
					refNode.m_TriLinearWeight[2] = detaX * (1-detaY) * detaZ;
					refNode.m_TriLinearWeight[3] = (1-detaX) * (1-detaY) * detaZ;
					refNode.m_TriLinearWeight[4] = detaX * detaY * (1-detaZ);
					refNode.m_TriLinearWeight[5] = (1-detaX) * detaY * (1-detaZ);
					refNode.m_TriLinearWeight[6] = detaX * (1-detaY) * (1-detaZ);
					refNode.m_TriLinearWeight[7] = (1-detaX) * (1-detaY) * (1-detaZ);

					

					for (int vv = 0; vv < 8; ++vv)
					{
						MyVectorI& curDofs = Cell::getCell(curNodeIdx)->getVertex(vv)->getDofs();/*>getDofs();*/
						refNode.m_VertexDofs[vv*3+0] = curDofs[0];
						refNode.m_VertexDofs[vv*3+1] = curDofs[1];
						refNode.m_VertexDofs[vv*3+2] = curDofs[2];
					}
					vec_vertexIdx2NodeIdxInside.push_back(refNode);

					Render::progressBar("WeightForSkinning",((float)vi*100)/(float)ref_vertices.size(),100);
				}
#else
				std::vector<Geometry::TriangleMeshNode >& vec_vertexIdx2NodeIdxInside = m_vec_vertexIdx2NodeIdxInside;
				vec_vertexIdx2NodeIdxInside.clear();

				
				const std::vector< vec3 >& ref_vertices = obj_data.points;
				std::vector< CellPtr >& refCellPool = m_vec_cell;
				initCudaStructForSkinningTriLinearWeight(ref_vertices.size(),refCellPool.size());
				//extern MeshVertex2CellInfo * g_cpu_triangleMeshVertex;
				//extern float3 * g_cpu_CellCenterPoint;
				const int nVtxSize = ref_vertices.size();
				for (int vi=0;vi<nVtxSize;++vi)
				{
					g_cpu_triangleMeshVertex[vi].m_vertexPos.x = ref_vertices[vi].x;
					g_cpu_triangleMeshVertex[vi].m_vertexPos.y = ref_vertices[vi].y;
					g_cpu_triangleMeshVertex[vi].m_vertexPos.z = ref_vertices[vi].z;

					g_cpu_triangleMeshVertex[vi].m_dist = FLT_MAX;
					g_cpu_triangleMeshVertex[vi].m_cellIdBelong = Invalid_Id;
				}

				for (int c=0;c<refCellPool.size();++c)
				{
					Cell2MeshVertexInfo & ref = g_cpu_CellCenterPoint[c];
					CellPtr curCellPtr = refCellPool[c];
					ref.m_centerPos.x = curCellPtr->getCenterPoint().x();
					ref.m_centerPos.y = curCellPtr->getCenterPoint().y();
					ref.m_centerPos.z = curCellPtr->getCenterPoint().z();
					ref.m_radius = curCellPtr->getRadius();

					for (int vv = 0; vv < Geometry::vertexs_per_cell; ++vv)
					{
						MyVectorI& curDofs = curCellPtr->getVertex(vv)->getDofs();/*>getDofs();*/
						ref.m_nDofs[vv*3+0] = curDofs[0];
						ref.m_nDofs[vv*3+1] = curDofs[1];
						ref.m_nDofs[vv*3+2] = curDofs[2];
					}
				}
				vec_vertexIdx2NodeIdxInside.resize(nVtxSize);
				createVertex2CellForSkinning();
				std::copy(g_cpu_TriangleMeshNode,g_cpu_TriangleMeshNode+nVtxSize,vec_vertexIdx2NodeIdxInside.begin());
				freeCudaStructForSkinningTriLinearWeight();
				/*for (int i=0;i<nVtxSize;++i)
				{
					Geometry::TriangleMeshNode& ref = vec_vertexIdx2NodeIdxInside[i];
					LogInfo("id[%d]:weight[%f,%f,%f,%f,%f,%f,%f,%f]\n",i,ref.m_TriLinearWeight[0],ref.m_TriLinearWeight[1],ref.m_TriLinearWeight[2],ref.m_TriLinearWeight[3],
																	   ref.m_TriLinearWeight[4],ref.m_TriLinearWeight[5],ref.m_TriLinearWeight[6],ref.m_TriLinearWeight[7]);
					LogInfo("id[%d]:dofs[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",i,
						ref.m_VertexDofs[0],ref.m_VertexDofs[1],ref.m_VertexDofs[2],ref.m_VertexDofs[3],
						ref.m_VertexDofs[4],ref.m_VertexDofs[5],ref.m_VertexDofs[6],ref.m_VertexDofs[7],
						ref.m_VertexDofs[8],ref.m_VertexDofs[9],ref.m_VertexDofs[10],ref.m_VertexDofs[11],
						ref.m_VertexDofs[12],ref.m_VertexDofs[13],ref.m_VertexDofs[14],ref.m_VertexDofs[15],
						ref.m_VertexDofs[16],ref.m_VertexDofs[17],ref.m_VertexDofs[18],ref.m_VertexDofs[19],
						ref.m_VertexDofs[20],ref.m_VertexDofs[21],ref.m_VertexDofs[22],ref.m_VertexDofs[23]);
				}				
				MyExit;*/
#endif
			}

			void VR_Physics_FEM_Simulation::update_rhs(const int nStep)
			{
				R_rhs.setZero();
				mass_rhs.setZero();
				damping_rhs.setZero();

				//incremental_displacement,velocity,acceleration
				mass_rhs += m_db_NewMarkConstant[0] * (displacement);
				mass_rhs += m_db_NewMarkConstant[2] * velocity;
				mass_rhs += m_db_NewMarkConstant[3] * acceleration;

				damping_rhs += m_db_NewMarkConstant[1] * (displacement);
				damping_rhs += m_db_NewMarkConstant[4] * velocity;
				damping_rhs += m_db_NewMarkConstant[5] * acceleration;

				R_rhs += m_computeRhs;

				R_rhs += m_global_MassMatrix * mass_rhs;
				R_rhs += m_global_DampingMatrix * damping_rhs;



#if USE_CO_RATION
				R_rhs -= m_RotationRHS;
#endif
				if (nStep >= cuda_bcMinCount && nStep < cuda_bcMaxCount)
				{
					R_rhs += R_rhs_externalForce;
					//MyPause;
				}
			}

			void VR_Physics_FEM_Simulation::setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx )
			{
				{
					for (MySpMat::InnerIterator it(matrix,rowIdx); it; ++it)
					{
						Q_ASSERT(rowIdx == it.row());
						const int r = it.row();
						const int c = it.col();
						if ( r == rowIdx && (r != c) )
						{
							it.valueRef() = MyNull;
						}
					}
				}
			}

			void VR_Physics_FEM_Simulation::apply_boundary_values()
			{
				std::vector< VertexPtr >& vecBoundaryVtx = m_vecDCBoundaryCondition;
				MySpMat&  computeMatrix = m_computeMatrix;
				MyVector curRhs = R_rhs;
				MyVector curDisplacement = incremental_displacement;
				if (vecBoundaryVtx.size() == 0)
					return;


				const unsigned int n_dofs = computeMatrix.rows();
				std::vector<MyFloat> diagnosticValue(n_dofs,0.0);

				for (int v=0;v < n_dofs;++v)
				{
					diagnosticValue[v] = computeMatrix.coeff(v,v);
					Q_ASSERT(!numbers::isZero( diagnosticValue[v]));
				}

				MyFloat first_nonzero_diagonal_entry = 1;
				for (unsigned int i=0; i<n_dofs; ++i)
				{
					if ( !numbers::isZero( diagnosticValue[i]) )
					{
						first_nonzero_diagonal_entry = diagnosticValue[i];
						break;
					}
				}

				for (unsigned i=0; i < vecBoundaryVtx.size(); ++i)
				{
					VertexPtr curVtxPtr = vecBoundaryVtx[i];
					const MyVectorI& Dofs = curVtxPtr->getDofs();
					for (unsigned c=0;c<MyDIM;++c)
					{
						const unsigned int dof_number = Dofs[c];
						setMatrixRowZeroWithoutDiag(computeMatrix,dof_number);

						MyFloat new_rhs;
						if ( !numbers::isZero(diagnosticValue[dof_number] ) )
						{
							new_rhs = 0 * diagnosticValue[dof_number];
							curRhs(dof_number) = new_rhs;
						}
						else
						{
							computeMatrix.coeffRef(dof_number,dof_number) = first_nonzero_diagonal_entry;
							new_rhs = 0 * first_nonzero_diagonal_entry;
							curRhs(dof_number) = new_rhs;
						}
						curDisplacement(dof_number) = 0;
					}			
				}
			}

			void VR_Physics_FEM_Simulation::solve_linear_problem()
			{
				SolverControl           solver_control (m_nGlobalDof,  1e-3*numbers::l2_norm(R_rhs));
				SolverCG              cg (solver_control);

				PreconditionSSOR preconditioner;
				preconditioner.initialize(m_computeMatrix, 1.2);

				cg.solve (m_computeMatrix, incremental_displacement, R_rhs,	preconditioner);

			}

			void VR_Physics_FEM_Simulation::compuateDisplacementVertexWithTrilinear()
			{
				MyVector& curDisplacement = incremental_displacement;
#ifdef USE_MULTI_DOMAIN
				MyError("global displacement");
#endif

				std::vector<TriangleMeshNode >& vec_vertexIdx2NodeIdxInside = m_vec_vertexIdx2NodeIdxInside;
				//printf("vec_vertexIdx2NodeIdxInside.size()=%d ; curObjData.normalizeVertices.size()=%d\n",vec_vertexIdx2NodeIdxInside.size(),curObjData.normalizeVertices.size());
				Q_ASSERT(vec_vertexIdx2NodeIdxInside.size() == m_objMesh.points.size());
				float p[2][2][2];
				float curDisplace[3];

				m_objMesh.displacedVertices.resize(m_objMesh.points.size());

				const MyInt nVertexCount = m_objMesh.points.size();
				for (MyInt v=0;v<nVertexCount;++v)
				{
					TriangleMeshNode& curTriangleMeshNode = vec_vertexIdx2NodeIdxInside[v];
					for (int step = 0; step < 3; ++step)
					{
						;
						p[0][0][0] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[0*3 + step] ];
						p[1][0][0] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[1*3 + step] ];
						p[0][1][0] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[2*3 + step] ];
						p[1][1][0] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[3*3 + step] ];
						p[0][0][1] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[4*3 + step] ];
						p[1][0][1] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[5*3 + step] ];
						p[0][1][1] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[6*3 + step] ];
						p[1][1][1] = curDisplacement[ curTriangleMeshNode.m_VertexDofs[7*3 + step] ];

						m_objMesh.displacedVertices[v][step] = p[0][0][0] * curTriangleMeshNode.m_TriLinearWeight[0] + 
							p[1][0][0] * curTriangleMeshNode.m_TriLinearWeight[1] + 
							p[0][1][0] * curTriangleMeshNode.m_TriLinearWeight[2] + 
							p[1][1][0] * curTriangleMeshNode.m_TriLinearWeight[3] + 
							p[0][0][1] * curTriangleMeshNode.m_TriLinearWeight[4] + 
							p[1][0][1] * curTriangleMeshNode.m_TriLinearWeight[5] + 
							p[0][1][1] * curTriangleMeshNode.m_TriLinearWeight[6] + 
							p[1][1][1] * curTriangleMeshNode.m_TriLinearWeight[7] + m_objMesh.points[v][step];
					}			
				}
			}

			void VR_Physics_FEM_Simulation::update_u_v_a()
			{
				const MyVector& solu = incremental_displacement;
				MyVector& disp_vec = displacement;
				MyVector& vel_vec = velocity;
				MyVector& acc_vec = acceleration;
				MyVector& old_acc = old_acceleration;
				MyVector& old_solu = old_displacement;

				old_solu = disp_vec;
				disp_vec = solu;
				old_acc  = acc_vec;

				acc_vec *= (-1 * m_db_NewMarkConstant[3]);//    acc_vec.scale(-1 * m_db_NewMarkConstant[3]);
				acc_vec += (disp_vec * m_db_NewMarkConstant[0]); //acc_vec.add(m_db_NewMarkConstant[0], disp_vec);
				acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));//acc_vec.add(-1 * m_db_NewMarkConstant[0], old_solu);
				acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));//acc_vec.add(-1 * m_db_NewMarkConstant[2],vel_vec);

				vel_vec += (old_acc * m_db_NewMarkConstant[6]);//vel_vec.add(m_db_NewMarkConstant[6],old_acc);
				vel_vec += (acc_vec * m_db_NewMarkConstant[7]);//vel_vec.add(m_db_NewMarkConstant[7],acc_vec);

#if USE_CO_RATION
				std::vector< CellPtr >& refCellVec = m_vec_cell;
				for (unsigned i=0; i < refCellVec.size();++i)
				{
					CellPtr cellPtr = refCellVec[i];
					cellPtr->computeRotationMatrix(incremental_displacement);
				}
#endif
			}

			void VR_Physics_FEM_Simulation::simulationOnCPU(const int nTimeStep)
			{
				update_rhs(nTimeStep);
				apply_boundary_values();
				solve_linear_problem ();
				compuateDisplacementVertexWithTrilinear();
				update_u_v_a();
#if USE_CO_RATION
				//assembleRotationSystemMatrix();
#endif
			}

			void VR_Physics_FEM_Simulation::registerVBOID(const std::vector< unsigned int >& vecVBOID )
			{
				for (int i=0;i<vecVBOID.size();++i)
				{
					cudaGLRegisterBufferObject(vecVBOID[i]);
				}
			}

			void VR_Physics_FEM_Simulation::simulationOnCUDA(const int nTimeStep, 
				unsigned int /*vao_handle*/, 
				unsigned int vbo_lines, unsigned int vbo_linesIndex, 
				unsigned int vbo_vertexes, unsigned int vbo_normals, unsigned int vbo_triangle, 
				unsigned int & nTriangeSize, unsigned int& nLineSize)
			{	
				//glBindVertexArray(vao_handle);
				float3 * ptr_vertexes;
				float3 * ptr_normals;
				int3 * ptr_triangles;

				cudaGLMapBufferObject((void**)&ptr_vertexes, vbo_vertexes);
				cudaGLMapBufferObject((void**)&ptr_normals, vbo_normals);
				//cudaGLMapBufferObject((void**)&ptr_triangles, vbo_triangle);
				float3* ptr_lines_vertexes;
				int2* ptr_lines_index;
				cudaGLMapBufferObject((void**)&ptr_lines_vertexes, vbo_lines);
				cudaGLMapBufferObject((void**)&ptr_lines_index, vbo_linesIndex);
				
				bool nouse = false;
				if (CUDA_SIMULATION::CUDA_CUTTING_GRID::g_needCheckCutting)
				{
					nouse = true;
					//outputGPUMatrix();
				}
				CUDA_SIMULATION::do_loop_single(nTimeStep,&ptr_lines_vertexes,&ptr_lines_index,&ptr_vertexes,&ptr_normals,&ptr_triangles,nTriangeSize,nLineSize);
				if (true)
				{
					//outputGPUMatrix();
					/*const int nSize = CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseRHSSize;
					printf("nouseRHSVec :");
					for (int v=0;v< nSize;++v)
					{
					printf("%f,",CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseRHSVec[v]);
					}
					printf("\n");*/
					//MyExit;
					/*printf("nouseCusp_Array_Rhs :");
					for (int v=0;v< nSize;++v)
					{
						printf("%f,",CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_Rhs[v]);
					}
					printf("\n");
					printf("nouseCusp_Array_Old_Acceleration :");
					for (int v=0;v< nSize;++v)
					{
						printf("%f,",CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_Old_Acceleration[v]);
					}
					printf("\n");
					printf("nouseCusp_Array_Old_Displacement :");
					for (int v=0;v< nSize;++v)
					{
						printf("%f,",CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_Old_Displacement[v]);
					}
					printf("\n");
					printf("nouseCusp_Array_R_rhs_Corotaion :");
					for (int v=0;v< nSize;++v)
					{
						printf("%f,",CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_R_rhs_Corotaion[v]);
					}
					printf("\n");

					for (int v=0;v< nSize;++v)
					{
						printf("%f + %f + %f + %f = %f\n",
							CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_Rhs[v],
							CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_Old_Acceleration[v],
							CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_Old_Displacement[v],
							CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseCusp_Array_R_rhs_Corotaion[v],
							CUDA_SIMULATION::CUDA_CUTTING_GRID::nouseRHSVec[v]);
					}
					CUDA_SIMULATION::CUDA_CUTTING_GRID::debug_ShowDofs();
					*/
				}
				
#if MY_VIDEO_OUTPUT_OBJ
				static int nFrameIdx = 0;
				ouputObjMesh4Video(nFrameIdx++,ptr_vertexes,ptr_normals,nTriangeSize);
#endif
				/*cudaGLUnmapBufferObject(vbo_lines);
				cudaGLUnmapBufferObject(vbo_linesIndex);*/
				cudaGLUnmapBufferObject(vbo_vertexes);
				cudaGLUnmapBufferObject(vbo_normals);

				cudaGLUnmapBufferObject(vbo_lines);
				cudaGLUnmapBufferObject(vbo_linesIndex);
				//cudaGLUnmapBufferObject(vbo_triangle);

				//glBindVertexArray(0);
#if USE_OUTPUT_RENDER_OBJ_MESH
				outputObjMeshInfoOnCPU();
#endif
			}
#if MY_VIDEO_OUTPUT_OBJ
			void VR_Physics_FEM_Simulation::ouputObjMesh4Video(const int nTimeStep,float3 * cuda_vertexes,	float3 * cuda_normals, const int nTriangeSize)
			{
				using namespace CUDA_SIMULATION::CUDA_SKNNING_CUTTING;
				float3 * cpu_vertexes,* cpu_normals;
				const int nVtxSize = nTriangeSize * 3;
				cpu_vertexes = new float3[nVtxSize];
				cpu_normals = new float3[nVtxSize];

				
				MC_Vertex_Cuda* tmp_Vertex = new MC_Vertex_Cuda[g_VBO_Struct_Node.g_nMCVertexSize];
				MC_Surface_Cuda* tmp_Triangle = new MC_Surface_Cuda[g_VBO_Struct_Node.g_nMCSurfaceSize];

				CUDA_SIMULATION::CUDA_DEBUG::cuda_OuputObjMesh4Video(&cpu_vertexes,&cpu_normals,&cuda_vertexes,&cuda_normals,nVtxSize,&tmp_Vertex,&tmp_Triangle);

				std::map< int,vec3 > mapId2Pos;
				for (int i=0;i<g_VBO_Struct_Node.g_nMCSurfaceSize;++i)
				{
					int v = i*3;
					mapId2Pos[tmp_Triangle[i].m_Vertex[0]] = vec3(cpu_vertexes[v].x , cpu_vertexes[v].y , cpu_vertexes[v].z);
					v++;
					mapId2Pos[tmp_Triangle[i].m_Vertex[1]] = vec3(cpu_vertexes[v].x , cpu_vertexes[v].y , cpu_vertexes[v].z);
					v++;
					mapId2Pos[tmp_Triangle[i].m_Vertex[2]] = vec3(cpu_vertexes[v].x , cpu_vertexes[v].y , cpu_vertexes[v].z);
				}

				
				std::stringstream ss;
				ss.str("");
				ss << YC::GlobalVariable::g_strCurrentPath << "\\obj4video\\" << YC::GlobalVariable::s_qstrCurrentTimeStamp << "_" << nTimeStep << ".obj";
				std::ofstream outfile(ss.str().c_str());
				for (int v=0;v<g_VBO_Struct_Node.g_nMCVertexSize;++v)
				{
					const vec3& curPos = mapId2Pos.at(v);
					outfile << "v " << curPos.x << " " << curPos.y << " " << curPos.z << std::endl;
				}

				int nBase;
				for (int f=0;f<g_VBO_Struct_Node.g_nMCSurfaceSize;++f)
				{				
					outfile << "f " << tmp_Triangle[f].m_Vertex[0]+1 << " " << tmp_Triangle[f].m_Vertex[1]+1 << " " << tmp_Triangle[f].m_Vertex[2]+1 << std::endl;
				}
				outfile.close();
				delete [] cpu_vertexes;
				delete [] cpu_normals;
				delete [] tmp_Vertex;
				delete [] tmp_Triangle;
			}
#endif
#if USE_CO_RATION
			void VR_Physics_FEM_Simulation::assembleRotationSystemMatrix()
			{
				std::map<long,std::map<long,Cell::TripletNode > >& StiffTripletNodeMap = Cell::m_TripletNode_Stiffness;
				std::map<long,Cell::TripletNode >& RhsTripletNode = Cell::m_TripletNode_Rhs;

				StiffTripletNodeMap.clear();
				RhsTripletNode.clear();

				m_RotationRHS.setZero();

				std::vector< CellPtr >& refCellVec = m_vec_cell;
				const unsigned nCellSize = refCellVec.size();
				for (unsigned v = 0; v < nCellSize; ++v)
				{
					printf("co-ro cell %d\n",v);
					refCellVec[v]->assembleRotationMatrix();
				}

				m_computeMatrix.setZero();
				m_global_StiffnessMatrix.setZero();
				m_global_DampingMatrix.setZero();
				//m_computeRhs.setZero();

				std::vector<int> localDofs;
				//std::map<long,std::map<long,TripletNode > > TripletNode4Mass,TripletNode4Stiffness;
				std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;


				/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


				std::map<long,std::map<long,Cell::TripletNode > >& TripletNode4Stiffness = Cell::m_TripletNode_Stiffness;
				std::map<long,std::map<long,Cell::TripletNode > >::const_iterator itr_tri =  TripletNode4Stiffness.begin();
				for (;itr_tri != TripletNode4Stiffness.end();++itr_tri)
				{
					const std::map<long,Cell::TripletNode >&  ref_map = itr_tri->second;
					std::map<long,Cell::TripletNode >::const_iterator itr_2 =  ref_map.begin();
					for (;itr_2 != ref_map.end();++itr_2)
					{
						vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
					}
				}

				m_global_StiffnessMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
				TripletNode4Stiffness.clear();
				vec_triplet.clear();

				/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if 1
				std::map<long,Cell::TripletNode >& TripletNode4RHS = Cell::m_TripletNode_Rhs;
				std::map<long,Cell::TripletNode >::const_iterator itr_rhs = TripletNode4RHS.begin();
				for (; itr_rhs != TripletNode4RHS.end();++itr_rhs)
				{
					m_RotationRHS[itr_rhs->first] = (itr_rhs->second).val;
				}
				TripletNode4RHS.clear();
#endif

				/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				m_global_DampingMatrix = Material::damping_alpha * m_global_MassMatrix + Material::damping_beta * m_global_StiffnessMatrix;

				m_computeMatrix = m_global_StiffnessMatrix;
				m_computeMatrix += m_db_NewMarkConstant[0] * m_global_MassMatrix;
				m_computeMatrix += m_db_NewMarkConstant[1] * m_global_DampingMatrix;
			}
#endif
			

#if USE_CUDA
			void VR_Physics_FEM_Simulation::initLocalStructForCUDA()
			{
				const std::vector< Cell::tuple_matrix >& refStiffnessMatrix = Cell::getStiffnessMatrixList();
				const std::vector< Cell::tuple_matrix >& refMassMatrix = Cell::getMassMatrixList();
				const std::vector< Cell::tuple_vector >& refRhs = Cell::getRhsList();
				const std::vector< FEMShapeValue >& refShapeValue = Cell::getFEMShapFunctionList();
				const int nStiffnessMatrixSize = refStiffnessMatrix.size();
				const int nMassMatrixSize = refMassMatrix.size();
				const int nRhsVectorSize = refRhs.size();
				const int nFEMValueSize = refShapeValue.size();

				Q_ASSERT( (nStiffnessMatrixSize == nMassMatrixSize)&&(nMassMatrixSize == nRhsVectorSize)&&(nRhsVectorSize == nFEMValueSize) );

				const int nLocalStiffnessMatrix = refStiffnessMatrix.size();
				const int nLocalMassMatrixSize = refMassMatrix.size();
				const int nLocalRhsVectorSize = refRhs.size();
				const int nLocalFEMValueSize = refShapeValue.size();

				Q_ASSERT( (nLocalStiffnessMatrix == nLocalMassMatrixSize)&&(nLocalMassMatrixSize == nLocalRhsVectorSize)&&(nLocalRhsVectorSize == nLocalFEMValueSize) );

				MyFloat * localStiffnessMatrixOnCpu = new MyFloat[nLocalStiffnessMatrix * Geometry::dofs_per_cell * Geometry::dofs_per_cell];
				MyFloat * localMassMatrixOnCpu		= new MyFloat[nLocalMassMatrixSize * Geometry::dofs_per_cell * Geometry::dofs_per_cell];
				MyFloat * localRhsVectorOnCpu       = new MyFloat[nLocalRhsVectorSize * Geometry::dofs_per_cell];
				FEMShapeValue * localFEMShapeValueOnCpu = new FEMShapeValue [nFEMValueSize]; 

				memset(localStiffnessMatrixOnCpu	,'\0'	,nLocalStiffnessMatrix * Geometry::dofs_per_cell * Geometry::dofs_per_cell);
				memset(localMassMatrixOnCpu			,'\0'	,nLocalMassMatrixSize * Geometry::dofs_per_cell * Geometry::dofs_per_cell);
				memset(localRhsVectorOnCpu			,'\0'	,nLocalRhsVectorSize * Geometry::dofs_per_cell );
				memset(localFEMShapeValueOnCpu		,'\0'	,nFEMValueSize * sizeof(FEMShapeValue));

				const int nSize = nLocalStiffnessMatrix;
				Q_ASSERT(1 == nSize);
				for (unsigned idx=0;idx<nSize;++idx)
				{
					const MyMatrix& curStiffnessMatrix = refStiffnessMatrix[idx].matrix;
					const MyMatrix& curMassMatrix = refMassMatrix[idx].matrix;
					const MyVector& curRhsVector = refRhs[idx].vec;

					/*std::cout << std::endl << "*********************** curStiffnessMatrix *********************************" << std::endl;
					std::cout << curStiffnessMatrix << std::endl;
					std::cout << std::endl << "*********************** curStiffnessMatrix *********************************" << std::endl;
					std::cout << curMassMatrix << std::endl;
					std::cout << std::endl << "*********************** curStiffnessMatrix *********************************" << std::endl;
					std::cout << curRhsVector << std::endl;
					MyExit;*/
					for (int row=0;row < Geometry::dofs_per_cell;++row)
					{
						for(int col=0;col < Geometry::dofs_per_cell;++col)
						{
							localStiffnessMatrixOnCpu[ idx*Geometry::dofs_per_cell * Geometry::dofs_per_cell + row * Geometry::dofs_per_cell + col] = curStiffnessMatrix.coeff(row,col);
							localMassMatrixOnCpu     [ idx*Geometry::dofs_per_cell * Geometry::dofs_per_cell + row * Geometry::dofs_per_cell + col] = curMassMatrix.coeff(row,col);
						}
						localRhsVectorOnCpu[idx*Geometry::dofs_per_cell + row] = curRhsVector.coeff(row,0);
						//LogInfo("row,value [%d,%f]\n",idx*Geometry::dofs_per_cell + row,localRhsVectorOnCpu[idx*Geometry::dofs_per_cell + row]);

					}	
					
					for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
					{
						for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
						{
							localFEMShapeValueOnCpu[idx].shapeFunctionValue_8_8[v][i] = refShapeValue[idx].shapeFunctionValue_8_8[v][i];
							localFEMShapeValueOnCpu[idx].shapeDerivativeValue_8_8_3[v][i][0] = refShapeValue[idx].shapeDerivativeValue_8_8_3[v][i][0];
							localFEMShapeValueOnCpu[idx].shapeDerivativeValue_8_8_3[v][i][1] = refShapeValue[idx].shapeDerivativeValue_8_8_3[v][i][1];
							localFEMShapeValueOnCpu[idx].shapeDerivativeValue_8_8_3[v][i][2] = refShapeValue[idx].shapeDerivativeValue_8_8_3[v][i][2];
						}
					}
				}
				CUDA_SIMULATION::VR_Physics_FEM_Simulation_InitLocalCellMatrixOnCuda(nSize,localStiffnessMatrixOnCpu,localMassMatrixOnCpu,localRhsVectorOnCpu,localFEMShapeValueOnCpu);
				Utils::getCurrentGPUMemoryInfo();
				
				
				const unsigned int   n_q_points		= Geometry::vertexs_per_cell;
				const unsigned int   nCellSize		= m_vec_cell.size();
				const unsigned int   VertexCount	= Vertex::getVertexSize();

				if (nCellSize >= MaxCellCount)
				{
					MyError("nCellSize >= MaxCellCount\n");
				}

				VertexOnCuda* VertexOnCudaPtr = MyNull;
				CommonCellOnCuda* CellOnCpuPtr = MyNull;
#if USE_HOST_MEMORY
				CUDA_SIMULATION::VR_Physics_FEM_Simulation_InitPhysicsCellAndVertex(VertexCount, &VertexOnCudaPtr, nCellSize, &CellOnCpuPtr);
#else
				VertexOnCudaPtr = new VertexOnCuda[VertexCount];
				CellOnCpuPtr = new CommonCellOnCuda[nCellSize];
#endif
				
				
				
				std::vector< CellPtr >& refCellVector = m_vec_cell;
				std::vector< std::pair<MyDenseVector,MyDenseVector> > vec_Lines;
				for (int cellOnCpuIdx = 0;cellOnCpuIdx < nCellSize;++cellOnCpuIdx)
				{
					CellPtr curCellPtr = refCellVector[cellOnCpuIdx];
					const MyPoint &currentPos = curCellPtr->getCenterPoint();

					CellOnCpuPtr[cellOnCpuIdx].cellType = curCellPtr->getCellType();
					Q_ASSERT(CellOnCpuPtr[cellOnCpuIdx].cellType == _CellTypeFEM);

					CellOnCpuPtr[cellOnCpuIdx].m_nStiffnessMatrixIdx = curCellPtr->getFEMCellStiffnessMatrixIdx();
					CellOnCpuPtr[cellOnCpuIdx].m_nMassMatrixIdx = curCellPtr->getFEMCellMassMatrixIdx();
					CellOnCpuPtr[cellOnCpuIdx].m_nRhsIdx = curCellPtr->getFEMCellRhsVectorIdx();
					CellOnCpuPtr[cellOnCpuIdx].m_nFEMShapeIdx = curCellPtr->getFEMCellShapeIdx();

#if 1
					CommonCellOnCuda& refCell = CellOnCpuPtr[cellOnCpuIdx];
					memcpy(&refCell.localRhsVectorOnCuda[0],&localRhsVectorOnCpu[refCell.m_nRhsIdx*Geometry_dofs_per_cell],Geometry_dofs_per_cell*sizeof(float));
					memcpy(&refCell.localMassMatrixOnCuda[0],&localMassMatrixOnCpu[refCell.m_nMassMatrixIdx*Geometry_dofs_per_cell_squarte],Geometry_dofs_per_cell_squarte*sizeof(float));
					/*for (int m=0;m<Geometry_dofs_per_cell;++m)
					{
						printf("%f,",refCell.localRhsVectorOnCuda[m]);
					}
					printf("\n");MyPause;*/
					
#endif

					CellOnCpuPtr[cellOnCpuIdx].m_nLevel = curCellPtr->getLevel();
					CellOnCpuPtr[cellOnCpuIdx].m_bNewOctreeNodeList;//initialize and modify on cuda
					CellOnCpuPtr[cellOnCpuIdx].m_bTopLevelOctreeNodeList;//initialize and modify on cuda
					CellOnCpuPtr[cellOnCpuIdx].m_bLeaf = true;
					CellOnCpuPtr[cellOnCpuIdx].m_needBeCutting = true;


					CellOnCpuPtr[cellOnCpuIdx].m_nGhostCellCount = 0;
					CellOnCpuPtr[cellOnCpuIdx].m_nGhostCellIdxInVec = -1;

					for (int k=0;k<n_q_points;++k)
					{
						CellOnCpuPtr[cellOnCpuIdx].vertexId[k] = curCellPtr->getVertex(k)->getId();
					}

					MyPoint centerPos = curCellPtr->getCenterPoint();
					CellOnCpuPtr[cellOnCpuIdx].m_centerPos[0] = centerPos[0];
					CellOnCpuPtr[cellOnCpuIdx].m_centerPos[1] = centerPos[1];
					CellOnCpuPtr[cellOnCpuIdx].m_centerPos[2] = centerPos[2];
					CellOnCpuPtr[cellOnCpuIdx].m_radius = curCellPtr->getRadius();

#if USE_CO_RATION
					MyVector currentPj; curCellPtr->get_postion(currentPj);
					for (MyInt k=0;k<Geometry_dofs_per_cell;++k)
					{
						CellOnCpuPtr[cellOnCpuIdx].Pj[k] = currentPj[k];
					}

					CellOnCpuPtr[cellOnCpuIdx].radiusx2 = curCellPtr->getRadius() * 2.f;
					CellOnCpuPtr[cellOnCpuIdx].weight4speedup = 0.25f * (1.f / CellOnCpuPtr[cellOnCpuIdx].radiusx2);
#endif

					CellOnCpuPtr[cellOnCpuIdx].m_nLinesBaseIdx = vec_Lines.size();
					CellOnCpuPtr[cellOnCpuIdx].m_nLinesCount = 3 + Geometry::lines_per_cell;
					CellOnCpuPtr[cellOnCpuIdx].m_nJxW = curCellPtr->getJxW(0);
					//LogInfo("%f --- %f\n",CellRaidus,curCellPtr->getRadius());
					Q_ASSERT(CellRaidus == curCellPtr->getRadius());
					MyDenseVector  x_step(CellRaidus,0,0),y_step(0,CellRaidus,0),z_step(0,0,CellRaidus);
					vec_Lines.push_back(std::make_pair(currentPos + x_step,currentPos + -1.f*x_step));
					vec_Lines.push_back(std::make_pair(currentPos + y_step,currentPos + -1.f*y_step));
					vec_Lines.push_back(std::make_pair(currentPos + z_step,currentPos + -1.f*z_step));

					for (int lv = 0;lv < Geometry::lines_per_cell;++lv)
					{
						vec_Lines.push_back(std::make_pair(curCellPtr->getVertex(linePair[lv][0])->getPos(),curCellPtr->getVertex(linePair[lv][1])->getPos()));
					}
				}

				

				delete [] localStiffnessMatrixOnCpu;
				delete [] localMassMatrixOnCpu;
				delete [] localRhsVectorOnCpu;
				delete [] localFEMShapeValueOnCpu;

				CuttingLinePair * linePairOnCpuPtr = new CuttingLinePair[vec_Lines.size()];
				for (int v=0,innerIdx = 0;v<vec_Lines.size();++v)
				{
					const MyDenseVector& ref0 = vec_Lines[v].first;
					const MyDenseVector& ref1 = vec_Lines[v].second;
					linePairOnCpuPtr[v] = std::make_pair(glm::vec3(ref0[0],ref0[1],ref0[2]),glm::vec3(ref1[0],ref1[1],ref1[2]));
				}
				/*MyFloat * linesOnCpuPtr = new MyFloat[vec_Lines.size() * 6];
				for (int v=0,innerIdx = 0;v<vec_Lines.size();++v)
				{

					linesOnCpuPtr[innerIdx++] = vec_Lines[v].first(0);
					linesOnCpuPtr[innerIdx++] = vec_Lines[v].first(1);
					linesOnCpuPtr[innerIdx++] = vec_Lines[v].first(2);

					linesOnCpuPtr[innerIdx++] = vec_Lines[v].second(0);
					linesOnCpuPtr[innerIdx++] = vec_Lines[v].second(1);
					linesOnCpuPtr[innerIdx++] = vec_Lines[v].second(2);
				}*/

				
				CUDA_SIMULATION::VR_Physics_FEM_Simulation_InitCellCuttingLineSetOnCuda(vec_Lines.size(),linePairOnCpuPtr);
				
				//delete [] CellOnCpuPtr;//HostAlloc on GPU
				delete [] linePairOnCpuPtr;

				for (int v=0;v<VertexCount;++v)
				{
					VertexPtr curVertexPtr = Vertex::getVertex(v);
					MyDenseVector &refLocal = curVertexPtr->getPos();
					MyVectorI & refDofs = curVertexPtr->getDofs();

					VertexOnCudaPtr[v].m_createTimeStamp = 0;
					VertexOnCudaPtr[v].m_nId = v;
					VertexOnCudaPtr[v].local[0] =  refLocal[0];
					VertexOnCudaPtr[v].local[1] =  refLocal[1];
					VertexOnCudaPtr[v].local[2] =  refLocal[2];
					VertexOnCudaPtr[v].m_nGlobalDof[0] = refDofs[0];
					VertexOnCudaPtr[v].m_nGlobalDof[1] = refDofs[1];
					VertexOnCudaPtr[v].m_nGlobalDof[2] = refDofs[2];
					//VertexOnCudaPtr[v].m_fromDomainId = curVertexPtr->getFromDomainId();

					/*MyVectorI& refLocalDofs = curVertexPtr->getDofs();
					VertexOnCudaPtr[v].m_nGlobalDof[0] = refLocalDofs[0];
					VertexOnCudaPtr[v].m_nGlobalDof[1] = refLocalDofs[1];
					VertexOnCudaPtr[v].m_nGlobalDof[2] = refLocalDofs[2];*/
					
				}
				if (VertexCount >= MaxVertexCount)
				{
					MyError("VertexCount >= MaxVertexCount\n");
				}

#if (!USE_HOST_MEMORY)
				CUDA_SIMULATION::VR_Physics_FEM_Simulation_InitPhysicsCellAndVertex(VertexCount, &VertexOnCudaPtr, nCellSize, &CellOnCpuPtr);
				delete []VertexOnCudaPtr;
				delete []CellOnCpuPtr;
#endif

				CUDA_SIMULATION::VR_Physics_FEM_Simulation_InitLinePair();
				
				const int boundaryconditionSize = m_vecDCBoundaryCondition.size();
				int *elem_boundaryCondition = new int [boundaryconditionSize*3+1];
				for (unsigned c=0;c<boundaryconditionSize;++c)
				{
					MyVectorI& refDofs = m_vecDCBoundaryCondition[c]->getDofs();
					elem_boundaryCondition[3*c+0] = refDofs[0];
					elem_boundaryCondition[3*c+1] = refDofs[1];
					elem_boundaryCondition[3*c+2] = refDofs[2];
				}

				const int forceConditionSize = m_vecForceBoundaryCondition.size();
				int *elem_forceCondition = new int[forceConditionSize+1];
				for (unsigned c=0;c<forceConditionSize;++c)
				{
					MyVectorI& refDofs = m_vecForceBoundaryCondition[c]->getDofs();
					elem_forceCondition[c] = refDofs[1];
				}

				float dbNewMarkConstant[8];
				for (int i=0;i<8;++i)
				{
					dbNewMarkConstant[i] = m_db_NewMarkConstant[i];
				}

				CUDA_SIMULATION::VR_Physics_FEM_Simulation_InitialLocalDomainOnCuda(m_nGlobalDof, dbNewMarkConstant,	elem_boundaryCondition,boundaryconditionSize*3,elem_forceCondition,forceConditionSize);
				
				LogInfo("initial_Cuda\n");
				delete [] elem_boundaryCondition;
				delete [] elem_forceCondition;

				MyDenseVector externalForceVec = Cell::getExternalForce();
				float externForce[3] = {externalForceVec[0],externalForceVec[1],externalForceVec[2]};
				CUDA_SIMULATION::makeGlobalIndexPara(Material::YoungModulus, Material::PossionRatio, Material::Density, &externForce[0]);
				
				Utils::getCurrentGPUMemoryInfo();
				CUDA_SIMULATION::assembleSystemOnCuda_FEM_RealTime();		
					
				
#if MY_DEBUG_OUTPUT_GPU_MATRIX
				/*outputGPUMatrix();
				MyExit;*/
#endif
				Utils::getCurrentGPUMemoryInfo();
				//MyPause;
				CUDA_SIMULATION::initBoundaryCondition();
				CUDA_SIMULATION::assembleSystemOnCuda_FEM_RealTime_MatrixInitDiag();
			}

			void VR_Physics_FEM_Simulation::makeLineStructForCuda(int ** line_vertex_pair_and_belongDomain/*3 times of the lineCount*/,int& lineCount)
			{
				static int linePair[12][2] = {{0,1},{2,3},{0,2},{1,3},{2,6},{3,7},{0,4},{1,5},{4,6},{5,7},{6,7},{4,5}};
				std::map< int, std::map<int,bool> > map_lineId;

				lineCount = 0;
				std::vector< CellPtr >& curDomainCellVec = m_vec_cell;

				const int nCellSize = curDomainCellVec.size();
				for (MyInt c=0;c<nCellSize;++c)
				{
					CellPtr curCellPtr = curDomainCellVec[c];
					for (unsigned l=0;l<12;++l)
					{
						const int leftId = curCellPtr->getVertex(linePair[l][0])->getId();
						const int rightId = curCellPtr->getVertex(linePair[l][1])->getId();
						map_lineId[leftId][rightId] = true;
					}
				}

				std::map< int, std::map<int,bool> >::const_iterator ci = map_lineId/*[domainId]*/.begin();
				std::map< int, std::map<int,bool> >::const_iterator endc = map_lineId/*[domainId]*/.end();
				for (;ci != endc; ++ci)
				{
					lineCount += (*ci).second.size();
				}
				*line_vertex_pair_and_belongDomain = new int[lineCount*3];
				memset(*line_vertex_pair_and_belongDomain,'\0',sizeof(int) * lineCount * 3 );

				int idx = 0;
				//for (MyInt domainId=0;domainId < Cell::LocalDomainCount;++domainId)
				{
					std::map< int, std::map<int,bool> >& refMap = map_lineId/*[domainId]*/;
					std::map< int, std::map<int,bool> >::const_iterator ci = refMap.begin();
					std::map< int, std::map<int,bool> >::const_iterator endc = refMap.end();
					for (;ci!= endc;++ci)
					{
						const int leftVertexId = (*ci).first;
						const std::map<int,bool>& refRightMap = (*ci).second;
						std::map<int,bool>::const_iterator ciLine = refRightMap.begin();
						for (;ciLine != refRightMap.end();++ciLine,idx += 3)
						{
							(*line_vertex_pair_and_belongDomain)[idx] = leftVertexId;
							(*line_vertex_pair_and_belongDomain)[idx+1] = (*ciLine).first;
							(*line_vertex_pair_and_belongDomain)[idx+2] = 0/*domainId*/;
						}
					}
				}

				Q_ASSERT(idx == (lineCount*3) );
			}

			void VR_Physics_FEM_Simulation::freeLineStructForCuda(int ** line_vertex_pair_and_belongDomain/*3 times of the lineCount*/,int& /*lineCount*/)
			{
				delete [] * line_vertex_pair_and_belongDomain;
			}

			void VR_Physics_FEM_Simulation::makeMeshStructureOnCuda(int& triangleCount)
			{
				printf("call makeMeshStructureOnCuda\n");
#if USE_DYNAMIC_VERTEX_NORMAL
				std::map< int /*vertex id*/, std::map<int/*triangle id*/,bool> > map4DynamicVertexNormal;
#endif
				std::vector< vec3 >& vertices = m_objMesh.points;
				const int nVertexSize = vertices.size();

				Q_ASSERT(m_vec_vertexIdx2NodeIdxInside.size() == nVertexSize);

				MC_Vertex_Cuda* tmp_Vertex = new MC_Vertex_Cuda[nVertexSize];
				memset(tmp_Vertex,'\0',nVertexSize * sizeof(MC_Vertex_Cuda));
				for (unsigned v=0;v<nVertexSize;++v)
				{
					MC_Vertex_Cuda& ptV = tmp_Vertex[v];
					ptV.m_isValid = true;ptV.m_isJoint = true;ptV.m_isSplit = false;ptV.m_nVertexId = v;
					ptV.m_VertexPos[0] = vertices[v][0];
					ptV.m_VertexPos[1] = vertices[v][1];
					ptV.m_VertexPos[2] = vertices[v][2];
					ptV.m_VertexPos4CloneTestBladeDistance[0] = vertices[v][0];
					ptV.m_VertexPos4CloneTestBladeDistance[1] = vertices[v][1];
					ptV.m_VertexPos4CloneTestBladeDistance[2] = vertices[v][2];
					ptV.m_CloneVertexIdx[0]=ptV.m_CloneVertexIdx[1]=Invalid_Id;
					ptV.m_distanceToBlade = 0.f;
					ptV.m_state = 0;
#if USE_DYNAMIC_VERTEX_NORMAL
					map4DynamicVertexNormal[v];
					ptV.m_nShareTriangleCount=0;
#endif

					const Geometry::TriangleMeshNode& refNode = m_vec_vertexIdx2NodeIdxInside[v];
					const int curNodeIdx = refNode.nBelongCellId;

					memcpy(&ptV.m_TriLinearWeight[0],&refNode.m_TriLinearWeight[0],8*sizeof(float));
					memcpy(&ptV.m_elemVertexRelatedDofs[0],&refNode.m_VertexDofs[0],24*sizeof(int));

					ptV.m_MeshVertex2CellId = curNodeIdx;
#if 0
					MyFloat minDistance = 1000.f;
					int   curNodeIdx = -1;
					int   curNodeBelongDomainId = -1;
					bool  bInside = false;
					MyFloat tmpFloat;

					for (MyInt domainIdx = 0;domainIdx < Cell::LocalDomainCount;++domainIdx)
					{
						std::vector< CellPtr >& refVecCell = m_vec_cell[domainIdx];
						for (int vj=0;vj < refVecCell.size();++vj )
						{
							tmpFloat = (vertices[v] - refVecCell[vj]->getCenterPoint()).squaredNorm();
							if (minDistance > tmpFloat)
							{
								curNodeIdx = vj;
								curNodeBelongDomainId = domainIdx;
								minDistance = tmpFloat;
							}
						}
					}

					TriangleMeshNode  refNode;// = vec_vertexIdx2NodeIdxInside[vec_vertexIdx2NodeIdxInside.size()-1];

					std::vector< CellPtr >& refVecCell = m_vec_cell[curNodeBelongDomainId];
					MyDenseVector p0 = refVecCell[curNodeIdx]->getVertex(0)->getPos();
					MyDenseVector p7 = refVecCell[curNodeIdx]->getVertex(7)->getPos();
					MyFloat detaX = (p7[0] - vertices[v][0]) / (p7[0] - p0[0]);
					MyFloat detaY = (p7[1] - vertices[v][1]) / (p7[1] - p0[1]);
					MyFloat detaZ = (p7[2] - vertices[v][2]) / (p7[2] - p0[2]);

					ptV.m_TriLinearWeight[0] = refNode.m_TriLinearWeight[0];//detaX * detaY * detaZ;
					ptV.m_TriLinearWeight[1] = refNode.m_TriLinearWeight[1];//(1-detaX) * detaY * detaZ;
					ptV.m_TriLinearWeight[2] = refNode.m_TriLinearWeight[2];//detaX * (1-detaY) * detaZ;
					ptV.m_TriLinearWeight[3] = refNode.m_TriLinearWeight[3];//(1-detaX) * (1-detaY) * detaZ;
					ptV.m_TriLinearWeight[4] = refNode.m_TriLinearWeight[4];//detaX * detaY * (1-detaZ);
					ptV.m_TriLinearWeight[5] = refNode.m_TriLinearWeight[5];//(1-detaX) * detaY * (1-detaZ);
					ptV.m_TriLinearWeight[6] = refNode.m_TriLinearWeight[6];//detaX * (1-detaY) * (1-detaZ);
					ptV.m_TriLinearWeight[7] = refNode.m_TriLinearWeight[7];//(1-detaX) * (1-detaY) * (1-detaZ);

					refNode.m_VertexDofs[];
					for (int vv = 0; vv < 8; ++vv)
					{
						MyVectorI & refDofs = refVecCell[curNodeIdx]->getVertex(vv)->getLocalDof(curNodeBelongDomainId);
						ptV.m_elemVertexRelatedDofs[vv*3+0] = refDofs[0];
						ptV.m_elemVertexRelatedDofs[vv*3+1] = refDofs[1];
						ptV.m_elemVertexRelatedDofs[vv*3+2] = refDofs[2];	

						//printf("(%d,%d,%d)\n",refDofs[0],refDofs[1],refDofs[2]);
					}
					//MyPause;
#endif

					//ptV.m_nCellBelongDomainId = curNodeBelongDomainId;
				}


				std::vector< int >& faces = m_objMesh.faces;
				const int nFaceSize = faces.size() / 3;
				
				std::map< std::pair<int,int>,int > m_map_lineSet;
				std::vector< MyVectorI > face_Position_indicies;
				face_Position_indicies.resize(nFaceSize);
				//std::vector< MyVectorI >& face_VertexNormal_indicies = m_obj_data.vertexNormal_indicies;
				for (unsigned s=0,lineId=0;s<nFaceSize;++s)
				{
					face_Position_indicies[s] = MyVectorI(faces[3*s+0],faces[3*s+1],faces[3*s+2]);
					const MyVectorI& refVec3 = face_Position_indicies[s];
					std::pair<int,int> curPair = std::make_pair(refVec3[0],refVec3[1]);
					if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
					{
						m_map_lineSet[curPair] = lineId;
						lineId++;
					}

					curPair = std::make_pair(refVec3[1],refVec3[2]);
					if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
					{
						m_map_lineSet[curPair] = lineId;
						lineId++;
					}

					curPair = std::make_pair(refVec3[2],refVec3[0]);
					if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
					{
						m_map_lineSet[curPair] = lineId;
						lineId++;
					}
				}

				std::vector< MyVectorI > triLine;
				int linePair[3][2] = {{1,2},{2,0},{0,1}};
				for (unsigned t=0;t<nFaceSize;++t)
				{
					const MyVectorI& refVec3 = face_Position_indicies[t];
					MyVectorI curTriLineOrder;
					//1,2  2,0  0,1
					for (unsigned l=0;l<3;++l)
					{
						curTriLineOrder[l] = m_map_lineSet.at(std::make_pair(refVec3[linePair[l][0]],refVec3[linePair[l][1]]));
					}
					triLine.push_back(curTriLineOrder);
				}
				Q_ASSERT(triLine.size() == face_Position_indicies.size());


				const int nEdgeSize = m_map_lineSet.size();
				MC_Edge_Cuda* tmp_Edge = new MC_Edge_Cuda[nEdgeSize];
				memset(tmp_Edge,'\0',nEdgeSize * sizeof(MC_Edge_Cuda));

				std::map< std::pair<int,int>,int >::const_iterator ci = m_map_lineSet.begin();
				std::map< std::pair<int,int>,int >::const_iterator endc = m_map_lineSet.end();
				for (;ci != endc;++ci)
				{
					const int l = ci->second;
					const std::pair<int,int>& lineSet = ci->first;

					MC_Edge_Cuda& curEdge = tmp_Edge[l];

					curEdge.m_hasClone = false;curEdge.m_isValid=true;curEdge.m_isJoint=true;curEdge.m_isCut=false;
					curEdge.m_state=0;curEdge.m_nLineId = l;curEdge.m_Vertex[0] = lineSet.first;curEdge.m_Vertex[1] = lineSet.second;
					curEdge.m_belongToTri[0] = curEdge.m_belongToTri[1] = curEdge.m_belongToTri[2] = Invalid_Id;
					curEdge.m_CloneIntersectVertexIdx[0] = curEdge.m_CloneIntersectVertexIdx[1] = Invalid_Id;
					curEdge.m_CloneEdgeIdx[0] = curEdge.m_CloneEdgeIdx[1] = Invalid_Id;
				}

				//const int nFaceSize = triLine.size();
				MC_Surface_Cuda* tmp_Surface = new MC_Surface_Cuda[nFaceSize];
				memset(tmp_Surface,'\0',nFaceSize*sizeof(MC_Surface_Cuda));
				std::vector< MyVectorI >& tri = face_Position_indicies;
				//std::vector< MyVectorI >& triVN = face_VertexNormal_indicies;
				for (unsigned s=0;s<nFaceSize;++s)
				{
					MC_Surface_Cuda& curFace = tmp_Surface[s];
					curFace.m_isValid = true;curFace.m_isJoint = true;curFace.m_nSurfaceId=s;
					curFace.m_Vertex[0] = tri[s][0];curFace.m_Vertex[1] = tri[s][1];curFace.m_Vertex[2] = tri[s][2];
					curFace.m_Lines[0] = triLine[s][0];curFace.m_Lines[1] = triLine[s][1];curFace.m_Lines[2] = triLine[s][2];
					curFace.m_VertexNormal[0] = /*triVN*/triLine[s][0];curFace.m_VertexNormal[1] = /*triVN*/triLine[s][1];curFace.m_VertexNormal[2] = /*triVN*/triLine[s][2];
					curFace.m_state = 0;

					for (unsigned l=0;l<3;++l)
					{
						const unsigned lineId = curFace.m_Lines[l];

						int order[2];
						for (unsigned v=0;v<3;++v)
						{
							if (curFace.m_Vertex[v] == tmp_Edge[lineId].m_Vertex[0])
							{
								order[0] = v;
							}
							if (curFace.m_Vertex[v] == tmp_Edge[lineId].m_Vertex[1])
							{
								order[1] = v;
							}
						}

						tmp_Edge[lineId].m_belongToTri[l] = s;
						tmp_Edge[lineId].m_belongToTriVertexIdx[l][0] = order[0];
						tmp_Edge[lineId].m_belongToTriVertexIdx[l][1] = order[1];
					}

#if USE_DYNAMIC_VERTEX_NORMAL
					map4DynamicVertexNormal[curFace.m_Vertex[0]][s]=true;
					map4DynamicVertexNormal[curFace.m_Vertex[1]][s]=true;
					map4DynamicVertexNormal[curFace.m_Vertex[2]][s]=true;
#endif
				}

#if USE_DYNAMIC_VERTEX_NORMAL
				std::map< int , std::map<int,bool> >::const_iterator ci4dynamicNormal = map4DynamicVertexNormal.begin();
				std::map< int , std::map<int,bool> >::const_iterator endc4dynamicNormal = map4DynamicVertexNormal.end();
				/*int nTmpMaxShareTriCount=0;
				for (;ci4dynamicNormal != endc4dynamicNormal;++ci4dynamicNormal)
				{
				const std::map<int,bool>& RefMap = ci4dynamicNormal->second;
				if (nTmpMaxShareTriCount < RefMap.size())
				{
				nTmpMaxShareTriCount = RefMap.size();
				}
				}
				printf("nTmpMaxShareTriCount %d\n",nTmpMaxShareTriCount);*/

				for (ci4dynamicNormal = map4DynamicVertexNormal.begin();ci4dynamicNormal != endc4dynamicNormal;++ci4dynamicNormal)
				{
					const int nVtxId = ci4dynamicNormal->first;
					const std::map<int,bool>& RefMap = ci4dynamicNormal->second;
					if (RefMap.size() > MaxVertexShareTriangleCount)
					{
						LogInfo("%d -- %d\n",RefMap.size() , MaxVertexShareTriangleCount);
						Q_ASSERT(RefMap.size() <= MaxVertexShareTriangleCount);
					}
					
					int * currentShareTriangleBase = &tmp_Vertex[nVtxId].m_eleShareTriangle[0];
					int & nRefCurrentShareTriangleCount = tmp_Vertex[nVtxId].m_nShareTriangleCount;

					std::map<int,bool>::const_iterator ci4SpecialVertex = RefMap.begin();
					std::map<int,bool>::const_iterator endc4SpecialVertex = RefMap.end();
					for (;ci4SpecialVertex != endc4SpecialVertex;++ci4SpecialVertex)
					{
						currentShareTriangleBase[nRefCurrentShareTriangleCount++] = ci4SpecialVertex->first;
					}
				}
				Q_ASSERT(map4DynamicVertexNormal.size() < Max_Vertex_Count);
				g_nNativeSurfaceVertexCount = map4DynamicVertexNormal.size();
				//MyPause;
#endif

				//std::vector< MyDenseVector > verticeNormals;
				std::vector< vec3 >& verticeNormals = m_objMesh.normals;
				const int nVertexNormalSize = verticeNormals.size();
				MyFloat * tmp_VertexNormal = new MyFloat[3*nVertexNormalSize];
				for (unsigned v=0;v<nVertexNormalSize;++v)
				{
					tmp_VertexNormal[3*v+0] = verticeNormals[v][0];
					tmp_VertexNormal[3*v+1] = verticeNormals[v][1];
					tmp_VertexNormal[3*v+2] = verticeNormals[v][2];
				}

				printf("makeMeshStructureOnCuda_Steak Vertex(%d) Edge(%d) Face(%d)\n",nVertexSize,nEdgeSize,nFaceSize);
				//MyPause;
				CUDA_SIMULATION::CUDA_SKNNING_CUTTING::initMeshCuttingStructure(nVertexSize,tmp_Vertex,nEdgeSize,tmp_Edge,nFaceSize,tmp_Surface,nVertexNormalSize,tmp_VertexNormal);

				triangleCount = nFaceSize;

				delete [] tmp_Vertex;
				delete [] tmp_Edge;
				delete [] tmp_Surface;
				delete [] tmp_VertexNormal;
				Utils::getCurrentGPUMemoryInfo();
				return ;
			}

			void VR_Physics_FEM_Simulation::initSkinningStructForCUDA()
			{
				LogInfo("call initSkinningStructForCUDA\n");
				CUDA_SIMULATION::CUDA_SKNNING_CUTTING::initVBOStructContext();
				int * line_vertex_pair;
				int lineCount;
				makeLineStructForCuda(&line_vertex_pair,lineCount);
				LogInfo("lineCount is %d \n",lineCount);

				CUDA_SIMULATION::CUDA_SKNNING_CUTTING::initVBODataStruct_LineSet(line_vertex_pair,lineCount);

				freeLineStructForCuda(&line_vertex_pair,lineCount);
				LogInfo("initVBODataStruct return \n");

				int triangleCount;
				makeMeshStructureOnCuda(triangleCount);
				//initVBOScene(triangleCount*2,lineCount*5);
				LogInfo("triangleCount %d\n",triangleCount);
			}

#if MY_DEBUG_OUTPUT_GPU_MATRIX
			void VR_Physics_FEM_Simulation::outputGPUMatrix()
			{
				static int sCount = 0;
				sCount++;
				int  nDofs;
				int * systemInnerIndexPtr; float * systemValuePtr;
				int * stiffInnerIndexPtr; float * stiffValuePtr;
				int * massInnerIndexPtr; float * massValuePtr;
				float * rhsValuePtr;
				CUDA_SIMULATION::CUDA_DEBUG::cuda_Debug_Get_MatrixData(nDofs,&systemInnerIndexPtr,&systemValuePtr,&stiffInnerIndexPtr,&stiffValuePtr,
					&massInnerIndexPtr,&massValuePtr,&rhsValuePtr);

				std::map<long,std::map<long,Cell::TripletNode > > systemMatrix,stiffMatrix,massMatrix;
				MyVector rhsVector;
				rhsVector.resize(nDofs);
				const int nCount = nDofs * nMaxNonZeroSizeInFEM;

				for (int i=0,nRow=0;i<nCount;++i)
				{
					nRow = i / nMaxNonZeroSizeInFEM;
					systemMatrix[nRow][systemInnerIndexPtr[i]].val += systemValuePtr[i];
					stiffMatrix[nRow][stiffInnerIndexPtr[i]].val += stiffValuePtr[i];
					massMatrix[nRow][massInnerIndexPtr[i]].val += massValuePtr[i];
				}

				for (int i=0;i<nDofs;++i)
				{
					rhsVector[i] = rhsValuePtr[i];
				}
				/*std::string strSysMatrix = GlobalVariable::g_strCurrentPath + std::string("\\systemMatrixGPU.mtx");
				LogInfo("system matrix [%s]\n",strSysMatrix.c_str());
				MySpMat spSysMatrix(nDofs,nDofs);
				assembleMatrix(systemMatrix,spSysMatrix);
				
				std::string strStiffMatrix = GlobalVariable::g_strCurrentPath + std::string("\\stiffMatrixGPU.mtx");
				LogInfo("Stiff matrix [%s]\n",strStiffMatrix.c_str());
				MySpMat spStiffMatrix(nDofs,nDofs);
				assembleMatrix(stiffMatrix,spStiffMatrix);
				
				std::string strMassMatrix = GlobalVariable::g_strCurrentPath + std::string("\\massMatrixGPU.mtx");
				LogInfo("mass matrix [%s]\n",strMassMatrix.c_str());
				MySpMat spMassMatrix(nDofs,nDofs);
				assembleMatrix(massMatrix,spMassMatrix);*/
				std::stringstream ss;
				ss << "d:\\system_false"<< sCount << ".mtx";
				printfMTX(ss.str().c_str(),nDofs,systemMatrix);

				ss.str("");
				ss << "d:\\stiff_false"<< sCount << ".mtx";
				printfMTX(ss.str().c_str(),nDofs,stiffMatrix);

				ss.str("");
				ss << "d:\\mass_false"<< sCount << ".mtx";
				printfMTX(ss.str().c_str(),nDofs,massMatrix);

				
				if (true)
				{
					/*LogInfo("m_computeMatrix begin\n");
					compareSparseMatrix(m_computeMatrix,spSysMatrix);
					LogInfo("m_computeMatrix end\n");
					LogInfo("m_global_StiffnessMatrix begin\n");
					compareSparseMatrix(m_global_StiffnessMatrix,spStiffMatrix);
					LogInfo("m_global_StiffnessMatrix end\n");
					LogInfo("m_global_MassMatrix begin\n");
					compareSparseMatrix(m_global_MassMatrix,spMassMatrix);
					LogInfo("m_global_MassMatrix end\n");*/
				}
				else
				{
					/*std::ofstream outfileSystemMatrix(strSysMatrix);
					outfileSystemMatrix << spSysMatrix;
					outfileSystemMatrix.close();

					std::ofstream outfileStiffMatrix(strStiffMatrix);
					outfileStiffMatrix << spStiffMatrix;
					outfileStiffMatrix.close();

					std::ofstream outfileMassMatrix(strMassMatrix);
					outfileMassMatrix << spMassMatrix;
					outfileMassMatrix.close();*/
				}

				CUDA_SIMULATION::CUDA_DEBUG::cuda_Debug_free_MatrixData(&systemInnerIndexPtr,&systemValuePtr,&stiffInnerIndexPtr,&stiffValuePtr,
					&massInnerIndexPtr,&massValuePtr,&rhsValuePtr);
			}

			void VR_Physics_FEM_Simulation::printfMTX(const char* lpszFileName, const int nDofs, const std::map<long,std::map<long,Cell::TripletNode > >& StiffTripletNodeMap)
			{
				std::ofstream outfile(lpszFileName);

				int nValCount = 0;
				std::map<long,std::map<long,Cell::TripletNode > >::const_iterator ci_row_begin = StiffTripletNodeMap.begin();
				std::map<long,std::map<long,Cell::TripletNode > >::const_iterator ci_row_end = StiffTripletNodeMap.end();
				for (;ci_row_begin != ci_row_end;++ci_row_begin)
				{
					const std::map<long,Cell::TripletNode >& refCols = ci_row_begin->second;
					nValCount += refCols.size();
				}

				outfile << nDofs << " " << nDofs << " " << nValCount << std::endl;
				printf("%d---%d---%d\n",nDofs,nDofs,nValCount);
				ci_row_begin = StiffTripletNodeMap.begin();
				ci_row_end = StiffTripletNodeMap.end();
				for (;ci_row_begin != ci_row_end;++ci_row_begin)
				{
					const std::map<long,Cell::TripletNode >& refCols = ci_row_begin->second;
					const int nRowId = ci_row_begin->first;
					std::map<long,Cell::TripletNode >::const_iterator ci_col_begin = refCols.begin();
					std::map<long,Cell::TripletNode >::const_iterator ci_col_end = refCols.end();
					for (;ci_col_begin != ci_col_end;++ci_col_begin)
					{
						const int nColId = ci_col_begin->first;
						outfile << nRowId << " " << nColId << " " << (ci_col_begin->second).val << std::endl;
					}
				}
				outfile.close();
			}

			void VR_Physics_FEM_Simulation::assembleMatrix(const std::map<long,std::map<long,Cell::TripletNode > >& StiffTripletNodeMap,MySpMat& outputMatrix)
			{
				std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
				std::map<long,std::map<long,Cell::TripletNode > >::const_iterator itr_tri =  StiffTripletNodeMap.begin();
				for (;itr_tri != StiffTripletNodeMap.end();++itr_tri)
				{
					const std::map<long,Cell::TripletNode >&  ref_map = itr_tri->second;
					std::map<long,Cell::TripletNode >::const_iterator itr_2 =  ref_map.begin();
					for (;itr_2 != ref_map.end();++itr_2)
					{
						vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
					}
				}
				if (vec_triplet.size() > 0)
				{
					outputMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
					vec_triplet.clear();
				}
			}

			void VR_Physics_FEM_Simulation::compareSparseMatrix(MySpMat& cpuMatrix, MySpMat& gpuMatrix)
			{
				Q_ASSERT(cpuMatrix.rows() == gpuMatrix.rows() && cpuMatrix.cols() == gpuMatrix.cols() );
				const int nRows = cpuMatrix.rows();
				const int nCols = cpuMatrix.cols();
				for (int r=0;r<nRows;++r)
				{
					for (int c=0;c<nCols;++c)
					{
						if (!numbers::isEqual(cpuMatrix.coeffRef(r,c), gpuMatrix.coeffRef(r,c),10.f))
						{
							LogInfo("value[%d,%d][cpu %f,gpu %f]\n",r,c,cpuMatrix.coeffRef(r,c),gpuMatrix.coeffRef(r,c));
							MyPause;
						}
					}
				}
			}

#endif//#if MY_DEBUG_OUTPUT_GPU_MATRIX

#if USE_OUTPUT_RENDER_OBJ_MESH
			void VR_Physics_FEM_Simulation::outputObjMeshInfoOnCPU()
			{
				int nVertexCount = MyNull;
				MC_Vertex_Cuda* curVertexSet = MyNull;
				int nTriSize = MyZero;
				MC_Surface_Cuda* curFaceSet = MyNull;
				int nCellCount = MyZero;
				CommonCellOnCuda* CellOnCudaPtr = MyNull;
				float* displacement = MyNull;

				CUDA_SIMULATION::CUDA_SKNNING_CUTTING::getObjMeshInfoFromCUDA(nVertexCount,&curVertexSet,nTriSize,&curFaceSet,nCellCount,&CellOnCudaPtr,&displacement);

				std::map< int/*vertex-id*/,MyDenseVector/*vertex-displace-position*/ > mapVertexIdToPoition;
				std::map< int/*triangle-id*/,MyVectorI/*triangle-vertex-id-list*/> mapTriIdToVtxIdx;

				float p[2][2][2];
				float curDisplace[3];

				for (int s=0;s<nTriSize;++s)
				{
					MC_Surface_Cuda& curface = curFaceSet[s];
					if (curface.m_isValid)
					{
						mapTriIdToVtxIdx[s] = MyVectorI(curface.m_Vertex[0],curface.m_Vertex[1],curface.m_Vertex[2]);

						for (int i = 0; i < 3; ++i)//pt0,pt1,pt2
						{
							MC_Vertex_Cuda& curVertex = curVertexSet[curface.m_Vertex[i]];

							for (int step = 0; step < 3; ++step)//x,y,z
							{
								p[0][0][0] = displacement[ curVertex.m_elemVertexRelatedDofs[0*3 + step] ];
								p[1][0][0] = displacement[ curVertex.m_elemVertexRelatedDofs[1*3 + step] ];
								p[0][1][0] = displacement[ curVertex.m_elemVertexRelatedDofs[2*3 + step] ];
								p[1][1][0] = displacement[ curVertex.m_elemVertexRelatedDofs[3*3 + step] ];
								p[0][0][1] = displacement[ curVertex.m_elemVertexRelatedDofs[4*3 + step] ];
								p[1][0][1] = displacement[ curVertex.m_elemVertexRelatedDofs[5*3 + step] ];
								p[0][1][1] = displacement[ curVertex.m_elemVertexRelatedDofs[6*3 + step] ];
								p[1][1][1] = displacement[ curVertex.m_elemVertexRelatedDofs[7*3 + step] ];


								curDisplace[step] = p[0][0][0] * curVertex.m_TriLinearWeight[0] + 
									p[1][0][0] * curVertex.m_TriLinearWeight[1] + 
									p[0][1][0] * curVertex.m_TriLinearWeight[2] + 
									p[1][1][0] * curVertex.m_TriLinearWeight[3] + 
									p[0][0][1] * curVertex.m_TriLinearWeight[4] + 
									p[1][0][1] * curVertex.m_TriLinearWeight[5] + 
									p[0][1][1] * curVertex.m_TriLinearWeight[6] + 
									p[1][1][1] * curVertex.m_TriLinearWeight[7] ;

							}

							mapVertexIdToPoition[curface.m_Vertex[i]] = MyDenseVector(curVertex.m_VertexPos[0] + curDisplace[0],
								curVertex.m_VertexPos[1] + curDisplace[1],
								curVertex.m_VertexPos[2] + curDisplace[2]);
						}
					}
				}

				const std::map< int/*vertex-id*/,MyDenseVector/*vertex-displace-position*/ >& refVtxMap = mapVertexIdToPoition;
				const std::map< int/*triangle-id*/,MyVectorI/*triangle-vertex-id-list*/>& refTriMap = mapTriIdToVtxIdx;

				static std::stringstream ss;
				ss.str("");

				ss << GlobalVariable::g_strCurrentPath << "\\image\\obj4render_" << getTimeStamp() << ".obj";
				std::ofstream outfile(ss.str().c_str());

				for (int v=0;v<nVertexCount;++v)
				{
					const MyDenseVector& vtxPos = refVtxMap.at(v);
					outfile << "v " << vtxPos.x() << " " << vtxPos.y() << " " << vtxPos.z() << std::endl;
				}

				outfile << "# vertex count is " << nVertexCount << std::endl << std::endl;

				std::map< int,MyVectorI>::const_iterator ci = refTriMap.begin();
				std::map< int,MyVectorI>::const_iterator endc = refTriMap.end();
				for (;ci != endc;++ci)
				{
					const MyVectorI& idxList = ci->second;
					outfile << "f " << idxList.x()+1 << " " << idxList.y()+1 << " " << idxList.z()+1 << std::endl;
				}
				outfile << "# triangle count is " << refTriMap.size() << std::endl;
				outfile.close();

				CUDA_SIMULATION::CUDA_SKNNING_CUTTING::freeObjMeshInfoFromCUDA(nVertexCount,&curVertexSet,nTriSize,&curFaceSet,nCellCount,&CellOnCudaPtr,&displacement);

			}
#endif

#if USE_Mesh_Cutting
			int VR_Physics_FEM_Simulation::getTriangleMesh(unsigned int vao_handle,unsigned int vbo_vertexes, unsigned int vbo_normals, unsigned int vbo_triangle )
			{
				static int nTriangeSize = 0;
				Q_ASSERT(m_objMesh.points.size() == m_objMesh.normals.size());
				static float3 * ptr_vertexes;
				static float3 * ptr_normals;
				cudaGLMapBufferObject((void**)&ptr_vertexes, vbo_vertexes);
				cudaGLMapBufferObject((void**)&ptr_normals, vbo_normals);
				
				nTriangeSize = CUDA_SIMULATION::CUDA_SKNNING_CUTTING::cuda_SkinningCutting_GetTriangle(&ptr_vertexes,&ptr_normals);

				cudaGLUnmapBufferObject(vbo_vertexes);
				cudaGLUnmapBufferObject(vbo_normals);

				//printf("nTriangeSize %d\n",nTriangeSize);
				return nTriangeSize*3;
			}

			MyVBOLineSet * VR_Physics_FEM_Simulation::getAABB()
			{
				static unsigned linePairs[12][2] = {{0,1},{4,5},{6,7},{2,3},
				{0,2},{1,3},{5,7},{4,6},
				{0,4},{1,5},{3,7},{2,6}};

				MyVBOLineSet * ret = new MyVBOLineSet;

				std::vector< float > lineSetPos;
				std::vector< int > lineSetIdx;
				std::vector< MyPoint > vertexes;
				MyPoint center(0.,0.,0.);
				float radius = 0.5f;

				int nLineCount = 0;

				nLineCount += 12;

				vertexes.resize(YC::Geometry::vertexs_per_cell);
				//FEM Ë³Ðò
				static MyDenseVector step[YC::Geometry::vertexs_per_cell] = 
				{MyDenseVector(-1,-1,-1), MyDenseVector(1,-1,-1),
				MyDenseVector(-1,1,-1)	, MyDenseVector(1,1,-1),
				MyDenseVector(-1,-1,1)	, MyDenseVector(1,-1,1),
				MyDenseVector(-1,1,1)	, MyDenseVector(1,1,1)};

				for (int v=0;v < Geometry::vertexs_per_cell;++v)
				{
					vertexes[v] = (center + radius * step[v]);
				}

				for (unsigned k=0;k<12;++k)
				{
					MyPoint & p0 = vertexes[linePairs[k][0]];
					MyPoint & p1 = vertexes[linePairs[k][1]];

					lineSetPos.push_back(p0.x());
					lineSetPos.push_back(p0.y());
					lineSetPos.push_back(p0.z());
					lineSetIdx.push_back(lineSetIdx.size());

					lineSetPos.push_back(p1.x());
					lineSetPos.push_back(p1.y());
					lineSetPos.push_back(p1.z());
					lineSetIdx.push_back(lineSetIdx.size());
				}
				//outfile.close();
				ret->initialize(nLineCount,&lineSetPos[0],&lineSetIdx[0]);
				return ret;
			}

			int VR_Physics_FEM_Simulation::getBladeList(unsigned int vao_handle,unsigned int vbo_vertexes, unsigned int vbo_indexes)
			{
				static int nLineSize = 0;
				static float3 * ptr_vertexes;
				static int2 * ptr_indexes;
				cudaGLMapBufferObject((void**)&ptr_vertexes, vbo_vertexes);
				cudaGLMapBufferObject((void**)&ptr_indexes, vbo_indexes);

				nLineSize = vecBladeList.size();
				float3 * cpuVertex = new float3[nLineSize*2];
				int2   * cpuIndex  = new int2[nLineSize];
				for (int i=0;i<vecBladeList.size();++i)
				{
					const MyPoint& p0 = vecBladeList[i].first;
					const MyPoint& p1 = vecBladeList[i].second;

					cpuVertex[2*i+0] = make_float3(p0[0],p0[1],p0[2]);
					cpuIndex[i].x = 2*i+0;

					cpuVertex[2*i+1] = make_float3(p1[0],p1[1],p1[2]);
					cpuIndex[i].y = 2*i+1;
				}

				CUDA_SIMULATION::CUDA_SKNNING_CUTTING::cuda_SkinningCutting_SetBladeList(&ptr_vertexes, cpuVertex, &ptr_indexes, cpuIndex, nLineSize);
				

				cudaGLUnmapBufferObject(vbo_vertexes);
				cudaGLUnmapBufferObject(vbo_indexes);

				delete [] cpuVertex;
				delete [] cpuIndex;


				return nLineSize;
			}

			void makeTriangle(float3* cpuVertex, float3* cpuNormal, const MyPoint& p0, const MyPoint& p1, const MyPoint& p2, int index)
			{
				MyDenseVector nl = (p0 - p1).cross((p1-p2));
				nl.normalize();
				(cpuVertex)[index+0] = make_float3(p0[0],p0[1],p0[2]);
				(cpuNormal)[index+0] = make_float3(nl[0],nl[1],nl[2]);

				(cpuVertex)[index+1] = make_float3(p1[0],p1[1],p1[2]);
				(cpuNormal)[index+1] = make_float3(nl[0],nl[1],nl[2]);

				(cpuVertex)[index+2] = make_float3(p2[0],p2[1],p2[2]);
				(cpuNormal)[index+2] = make_float3(nl[0],nl[1],nl[2]);
			}

			int VR_Physics_FEM_Simulation::getBladeTriangleList(unsigned int vao_handle,unsigned int vbo_vertexes, unsigned int vbo_normals)
			{
				static int nTriSize = 0;
				static float3 * ptr_vertexes;
				static float3 * ptr_normals;
				nTriSize = 0;

				if (vecBladeList.size() > 1)
				{
					cudaGLMapBufferObject((void**)&ptr_vertexes, vbo_vertexes);
					cudaGLMapBufferObject((void**)&ptr_normals, vbo_normals);

					const int nLineSize = vecBladeList.size();
					nTriSize = (nLineSize-1)*4;
					float3 * cpuVertex = new float3[nTriSize*3];
					float3 * cpuNormal = new float3[nTriSize*3];
					for (int i=1,index=0;i<vecBladeList.size();++i)
					{
						const MyPoint& p00 = vecBladeList[i-1].first;
						const MyPoint& p01 = vecBladeList[i-1].second;
						const MyPoint& p10 = vecBladeList[i].first;
						const MyPoint& p11 = vecBladeList[i].second;


						makeTriangle(cpuVertex,cpuNormal,p01,p00,p10,index);

						index += 3;
						makeTriangle(cpuVertex,cpuNormal,p00,p01,p10,index);

						index += 3;
						makeTriangle(cpuVertex,cpuNormal,p11,p01,p10,index);

						index += 3;
						makeTriangle(cpuVertex,cpuNormal,p10,p01,p11,index);

						index += 3;
					}

					

					CUDA_SIMULATION::CUDA_SKNNING_CUTTING::cuda_SkinningCutting_SetBladeTriangleList(&ptr_vertexes, cpuVertex, &ptr_normals, cpuNormal, nTriSize);


					cudaGLUnmapBufferObject(vbo_vertexes);
					cudaGLUnmapBufferObject(vbo_normals);

					delete [] cpuVertex;
					delete [] cpuNormal;
				}


				return nTriSize*3;
			}

			void VR_Physics_FEM_Simulation::addBlade(const MyPoint& p0,const MyPoint& p1)
			{
				vecBladeList.push_back(std::make_pair(p0,p1));
			}
#endif//USE_Mesh_Cutting

#endif//#if USE_CUDA
		}//namespace GPU
	}//namespace Physics
}//namespace YC