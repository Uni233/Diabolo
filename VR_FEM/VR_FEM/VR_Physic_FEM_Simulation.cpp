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

#include <fstream>
#include "VR_GPU_Physic_StructInfo.h"

#include <map>






extern int YC::GlobalVariable::g_bcMaxCount;
extern int YC::GlobalVariable::g_bcMinCount;
extern float YC::GlobalVariable::g_scriptForceFactor;
extern float YC::GlobalVariable::g_externalForceFactor;

int cuda_bcMaxCount;
int cuda_bcMinCount;
float cuda_scriptForceFactor;
int g_nNativeSurfaceVertexCount=0;



extern std::string getTimeStamp();

extern void func();
//using namespace CUDA_SIMULATION;
namespace YC
{
	namespace Physics
	{
		namespace GPU
		{
			static float logN(const int base, float value)
			{
				return (float)(log((double)value) / log ((double)base));
				//,log2 x=(log10 x)/(log10 2)
			}

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

					refCurrentCellVec.push_back(Cell::makeCell(MyPoint(curCellPtr[0],curCellPtr[1],curCellPtr[2]),curCellPtr[3]));
					refCurrentCellVec[refCurrentCellVec.size()-1]->computeCellType();
					refCurrentCellVec[refCurrentCellVec.size()-1]->computeCell_NeedBeCutting();

					//const int nLevel = ((int)numbers::logN(2,(1.0/curCellPtr[3])))-1;
					refCurrentCellVec[refCurrentCellVec.size()-1]->setLevel(((int)logN(2,(1.0/curCellPtr[3])))-1);
				}
				printf("cell size %d, vertex size %d\n",Cell::getCellSize(),Vertex::getVertexSize());
				//MyPause;
				distributeDof();
				createForceBoundaryCondition();
				createGlobalMassAndStiffnessAndDampingMatrixFEM();
				createNewMarkMatrix();
				createDCBoundaryCondition();

				createTrilinearWeightForSkinning(m_objMesh);

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
				if (pos[0] < (-0.3f) && pos[1] > (0.3f))
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
					refRhs[curDof] = YC::GlobalVariable::g_scriptForceFactor * Material::GravityFactor*(-1.f);
					
				}

				m_global_DampingMatrix = Material::damping_alpha * m_global_MassMatrix + Material::damping_beta * m_global_StiffnessMatrix;
			}

			void VR_Physics_FEM_Simulation::createNewMarkMatrix()
			{
				m_computeMatrix = m_global_StiffnessMatrix;
				m_computeMatrix += m_db_NewMarkConstant[0] * m_global_MassMatrix;
				m_computeMatrix += m_db_NewMarkConstant[1] * m_global_DampingMatrix;
				
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
				if (pos[1] < (0.1f-0.5f) /*&& pos[0] < 0.f*/)
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
				//Q_ASSERT(m_vecDCBoundaryCondition.size() > 50);
				//MyPause;
			}

			void VR_Physics_FEM_Simulation::createTrilinearWeightForSkinning(const MeshDataStruct& obj_data)
			{
#if (USE_CUDA)
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

					//printf("distance %f\n",(vertices - Cell::getCell(refNode.nBelongCellId)->getCenterPoint()).norm());
				}
#else
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
				//std::cout << incremental_displacement.transpose() << std::endl;//MyPause;
			}

			void VR_Physics_FEM_Simulation::compuateDisplacementVertexWithTrilinear()
			{
				MyVector& curDisplacement = incremental_displacement;

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
				assembleRotationSystemMatrix();
#endif
			}

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
					//printf("co-ro cell %d\n",v);
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

			void VR_Physics_FEM_Simulation::generateDisplaceLineSet(float3** posPtr,int2 ** indexPtr)
			{
				float3* pos = *posPtr;
				int2* indexLines = *indexPtr;
				static int linePair[24] = {0,2,4,6,0,4,2,6,1,3,5,7,1,5,3,7,0,1,4,5,2,3,6,7};
				int lineBase;
				for (int c=0;c<m_vec_cell.size();++c)
				{
					CellPtr curCellPtr = m_vec_cell[c];
					lineBase = c*24;
					for (int i=0;i<12;++i)
					{
						const int lineIdx = lineBase + 2*i;
						//linePairOnCuda
						const int localStartVtxId = linePair[i*2];
						const int localEndVtxId = linePair[i*2+1];
						const VertexPtr beginVertexPtr = curCellPtr->getVertex(localStartVtxId);
						const VertexPtr endVertexPtr = curCellPtr->getVertex(localEndVtxId);

						MyVectorI beginVtxDofs = beginVertexPtr->getDofs();
						MyDenseVector beginVtxRestPos = beginVertexPtr->getPos();
						MyVectorI endVtxDofs = endVertexPtr->getDofs();
						MyDenseVector endVtxRestPos = endVertexPtr->getPos();

						

						pos[ lineIdx ] = make_float3(beginVtxRestPos[0] + incremental_displacement[beginVtxDofs[0]]/*-0.5f*/,
							beginVtxRestPos[1] + incremental_displacement[beginVtxDofs[1]]/*-0.f*/,
							beginVtxRestPos[2] + incremental_displacement[beginVtxDofs[2]]/*-0.5f*/);	
						pos[lineIdx+1 ] = make_float3(endVtxRestPos[0] + incremental_displacement[endVtxDofs[0]]/*-0.5f*/,
							endVtxRestPos[1] + incremental_displacement[endVtxDofs[1]]/*-0.f*/,
							endVtxRestPos[2] + incremental_displacement[endVtxDofs[2]]/*-0.5f*/);

						indexLines[c*12+i] = make_int2(lineIdx,lineIdx+1);
					}
				}
			}
		}//namespace GPU
	}//namespace Physics
}//namespace YC