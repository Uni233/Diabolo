#include "stdafx.h"
#include "ElasticSimulation_Couple.h"
#include "MeshParser_Obj/objParser.h"
#include "SOIL.h"
#include "CG/solvercontrol.h"
#include "CG/solvercg.h"
#include "CG/preconditionssor.h"
extern float element_steak[3136][4];
extern float rotate_x;
extern float rotate_y;
extern float translate_z;
extern float scaled;
extern std::string g_strMeshId;
namespace VR_FEM
{
	ElasticSimulation_Couple::ElasticSimulation_Couple():m_isSimulate(true)
	{
		generateNewMarkParam();
	}

	bool ElasticSimulation_Couple::parserObj(bool hasCoord,bool hasVerticeNormal,const char* lpszMeshPath)
	{
		objParser parser(hasCoord,hasVerticeNormal,&m_obj_data);
		parser.parser(lpszMeshPath);
		return true;
	}

	bool ElasticSimulation_Couple::loadOctreeNode(const char* lpszModelType)
	{
		MyFloat ** element_data = MyNull;
		MyInt nSamplePointSize=0;
		if (MyNull == strcmp(lpszModelType,"armadillo"))
		{
			printf("not support mesh type armadillo. \n");
			MyExit;
		}
		else if (MyNull == strcmp(lpszModelType,"steak") )
		{
			nSamplePointSize = 3136;
			element_data = (MyFloat **)element_steak;
		}
		else
		{
			printf("UnKnow model type.\n");
			MyExit;
		}

		
		for (MyInt i=0;i<nSamplePointSize; ++i)
		{
			m_vec_cell.push_back(Cell::makeCell(MyPoint(element_steak[i][0],element_steak[i][1],element_steak[i][2]),element_steak[i][3]));
			m_vec_cell[m_vec_cell.size()-1]->computeCellType_Steak();
		}
		Q_ASSERT(m_vec_cell.size() == Cell::getCellSize());

		printf("FEM (%d), EFG (%d), Couple(%d)\n",Cell::s_nFEM_Cell_Count,Cell::s_nEFG_Cell_Count,Cell::s_nCOUPLE_Cell_Count);
		//steak : FEM (1712), EFG (680), Couple(744)
		return true;
	}

	bool ElasticSimulation_Couple::loadOctreeNode_Beam(const char* lpszModelType)
	{
		const int X_Count = 20;
		const int Y_Count = 4;
		const int Z_Count = 4;
		const MyFloat diameter = 2*CellRaidus;
		std::vector< MyFloat > vec_X_SamplePoint,vec_Y_SamplePoint,vec_Z_SamplePoint;
		for (unsigned i=0;i<X_Count;++i)
		{
			vec_X_SamplePoint.push_back(i*diameter+CellRaidus);
		}
		for (unsigned i=0;i<Y_Count;++i)
		{
			vec_Y_SamplePoint.push_back(i*diameter+CellRaidus);
		}
		for (unsigned i=0;i<Z_Count;++i)
		{
			vec_Z_SamplePoint.push_back(i*diameter+CellRaidus);
		}

		printf("vec_X_SamplePoint.size(%d) vec_Y_SamplePoint.size(%d) vec_Z_SamplePoint.size(%d)\n",vec_X_SamplePoint.size(),vec_Y_SamplePoint.size(),vec_Z_SamplePoint.size());
		//MyPause;
		for (unsigned i=0;i<vec_X_SamplePoint.size();++i)
		{
			for (unsigned j=0;j<vec_Y_SamplePoint.size();++j)
			{
				for (unsigned k=0;k<vec_Z_SamplePoint.size();++k)
				{
					m_vec_cell.push_back(Cell::makeCell(MyPoint(vec_X_SamplePoint[i],vec_Y_SamplePoint[j],vec_Z_SamplePoint[k]),CellRaidus));
					m_vec_cell[m_vec_cell.size()-1]->computeCellType_Beam();
				}
			}
		}

		Q_ASSERT(m_vec_cell.size() == Cell::getCellSize());

		CellPtr testCellPtr = m_vec_cell[m_vec_cell.size()-1];
		MyPoint p0 = testCellPtr->getVertex(0)->getPos();
		MyPoint p1 = testCellPtr->getVertex(1)->getPos();
		MyFloat dist = (p1 - p0).norm();

		if (!numbers::isEqual(dist,CellRaidus*2))
		{
			printf("%f != %f \n",dist,CellRaidus*2);
			MyExit;
		}

		printf("FEM (%d), EFG (%d), Couple(%d)\n",Cell::s_nFEM_Cell_Count,Cell::s_nEFG_Cell_Count,Cell::s_nCOUPLE_Cell_Count);

		return true;
	}

	void ElasticSimulation_Couple::createOutBoundary()
	{
		m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId.clear();
		const MyInt nVtxSize = Vertex::getVertexSize();
		for (MyInt v=0;v<nVtxSize;++v)
		{
			VertexPtr curVtxPtr = Vertex::getVertex(v);
			if (curVtxPtr->isTmpBoundary())
			{
				m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId.push_back(MyVectorI(v,curVtxPtr->getTmpLocalDomainId(),curVtxPtr->getTmpCoupleDomainId()));
			}
		}
		printf("Out Boundary size %d.\n",m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId.size());
	}

	void ElasticSimulation_Couple::createOutBoundary_Force()
	{
		m_vecForceCoupleNode.clear();
		const MyInt nVtxSize = Vertex::getVertexSize();
		ForceCoupleNode tmpNode;
		for (MyInt v=0;v<nVtxSize;++v)
		{
			VertexPtr curVtxPtr = Vertex::getVertex(v);
			if (curVtxPtr->isBelongCoupleDomain())
			{
				const std::map< MyInt,MyVectorI >& refCoupleMap = curVtxPtr->m_mapCoupleDofs;
				std::map< MyInt,MyVectorI >::const_iterator ci_couple = refCoupleMap.begin();
				std::map< MyInt,MyVectorI >::const_iterator endc_couple = refCoupleMap.end();
				for (;ci_couple != endc_couple;++ci_couple)
				{
					const MyInt nCoupleDomainId = ci_couple->first;
					const MyVectorI& vecCoupleDomainDofs = ci_couple->second;
					tmpNode.m_nCoupleDomainId = nCoupleDomainId;
					tmpNode.m_vecCoupleDomainDofs = vecCoupleDomainDofs;

					const std::map< MyInt,MyVectorI >& refMap = curVtxPtr->getLocalDofsMap();
					Q_ASSERT(refMap.size() >0 && refMap.size() < 3);
					std::map< MyInt,MyVectorI >::const_iterator ci = refMap.begin();
					std::map< MyInt,MyVectorI >::const_iterator endc = refMap.end();
					for (;ci != endc;++ci)
					{
						tmpNode.m_nLocalDomainId = ci->first;
						tmpNode.m_vecLocalDomainDofs = ci->second;
						m_vecForceCoupleNode.push_back(tmpNode);
					}
				}
				
			}
		}
		printf("Out Boundary size %d.\n",m_vecForceCoupleNode.size());
	}

	void ElasticSimulation_Couple::createInnerBoundary_Force()
	{
		const MyInt nCellSize = Cell::getCellSize();
		MyInt nCount=0;
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = Cell::getCell(c);
			if ( Invalid_Id != (curCellPtr->getCoupleDomainId()) )
			{
				nCount++;
				const MyInt nCoupleId = curCellPtr->getCoupleDomainId();
				const MyInt nLocalId = curCellPtr->getDomainId();
				Q_ASSERT(nLocalId != Invalid_Id);
				for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
				{
					VertexPtr curVtxPtr = curCellPtr->getVertex(v);
					//if ( (curVtxPtr->getVertexShareDomainCount()) == 2 )
					{
						std::map< MyInt,MyVectorI >& refMap = curVtxPtr->m_mapLocalDofs;
						std::map< MyInt,MyVectorI >::const_iterator ci = refMap.begin();
						std::map< MyInt,MyVectorI >::const_iterator endc = refMap.end();

						for (;ci != endc;++ci )
						{
							m_vecInnerBoundary_Force_VtxId_FEMDomainId_EFGDomainId.push_back(MyVectorI(curVtxPtr->getId(),ci->first,nCoupleId));
						}
						
					}
					//Q_ASSERT( (0 < curVtxPtr->getVertexShareDomainCount()) && (3 > curVtxPtr->getVertexShareDomainCount()) );
					
				}
				
			}
		}
		/*printf("Inner Boundary size %d.\n",m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size());
		printf("nCellSize %d, nCount %d.\n ",nCellSize,nCount);
		MyPause;*/
	}

	void ElasticSimulation_Couple::createInnerBoundary()
	{
		std::map<MyInt,bool> mapHash;
		const MyInt nCellSize = Cell::getCellSize();
		MyInt nCount=0;
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = Cell::getCell(c);
			if ( Invalid_Id != (curCellPtr->getCoupleDomainId()) )
			{
				nCount++;
				const MyInt nCoupleId = curCellPtr->getCoupleDomainId();
				const MyInt nLocalId = curCellPtr->getDomainId();
				Q_ASSERT(nLocalId != Invalid_Id);
				for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
				{
					VertexPtr curVtxPtr = curCellPtr->getVertex(v);
					if ( (curVtxPtr->getVertexShareDomainCount()) == 2 && mapHash.find(curVtxPtr->getId()) == mapHash.end() )
					{
						std::map< MyInt,MyVectorI >& refMap = curVtxPtr->m_mapLocalDofs;
						std::map< MyInt,MyVectorI >::const_iterator ci = refMap.begin();
						std::map< MyInt,MyVectorI >::const_iterator endc = refMap.end();

						for (;ci != endc;++ci )
						{
							m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.push_back(MyVectorI(curVtxPtr->getId(),ci->first,nCoupleId));
							printf("{%d,%d,%d},",curVtxPtr->getId(),ci->first,nCoupleId);
						}
						mapHash[curVtxPtr->getId()] = true;
					}
				}
			}
		}

		printf("\n m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size() is %d\n",m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size());
		//MyPause;
	}

	void ElasticSimulation_Couple::generateCoupleDomain()
	{
		std::vector< std::pair< MyInt,MyInt > > vecCoupleDomainPool;
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			if (m_vec_cell[c]->getDomainId() == Cell::CoupleDomainId)
			{
				MyFloat minDistance = FLT_MAX;
				MyInt   cellId = -1;
				const MyPoint& srcCenter = m_vec_cell[c]->getCenterPoint();
				for (MyInt v=0;v<nCellSize;++v)
				{
					if (m_vec_cell[v]->getDomainId() != Cell::CoupleDomainId)
					{
						MyFloat curDistance = ((m_vec_cell[v]->getCenterPoint()) - srcCenter).norm();
						if (curDistance < minDistance)
						{
							minDistance = curDistance;
							cellId = v;
						}
					}
				}

				vecCoupleDomainPool.push_back(std::make_pair(c,cellId));
			}
		}

		for (MyInt v=0;v<vecCoupleDomainPool.size();++v)
		{
			MyInt srcCellId = vecCoupleDomainPool[v].first;
			MyInt dstCellId = vecCoupleDomainPool[v].second;
			m_vec_cell[srcCellId]->m_nDomainId = m_vec_cell[dstCellId]->getDomainId();
		}

		std::vector< MyInt > vecTmp(Cell::LocalDomainCount,0);
		std::vector< MyInt > vecTmp1(Cell::CoupleDomainCount+1,0);
		for (MyInt c=0;c<nCellSize;++c)
		{
			vecTmp[ m_vec_cell[c]->getDomainId() ] ++;
			vecTmp1[ m_vec_cell[c]->getCoupleDomainId() + 1 ]++;
		}

		for (MyInt v=0;v<Cell::LocalDomainCount;++v)
		{
			printf("Local [%d] size %d.\n",v,vecTmp[v]);
		}

		for (MyInt v=0;v<=Cell::CoupleDomainCount;++v)
		{
			printf("Couple [%d] size %d.\n",v,vecTmp1[v]);
		}
	}

	void ElasticSimulation_Couple::distributeDof_global()
	{
		m_nGlobalDof = Geometry::first_dof_idx;
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
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

	bool ElasticSimulation_Couple::isDCCondition(const int DomainId,const MyPoint& pos)
	{
		if (0 == DomainId)
		{
			//return true;
			if (pos[0] < CellRaidus)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}

	bool ElasticSimulation_Couple::isForceCondition(const int DomainId,const MyPoint& pos)
	{
		if (1 == DomainId)
		{
			if (pos[0] > 39*CellRaidus)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}

	void ElasticSimulation_Couple::createForceBoundaryCondition(const int DomainId)
	{
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			if ( (curCellPtr->getDomainId()) == DomainId )
			{
				for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
				{
					VertexPtr curVtxPtr = curCellPtr->getVertex(v);
					const MyPoint & pos = curVtxPtr->getPos();
					if (isForceCondition(DomainId,pos) )
					{
						m_vecForceBoundaryCondition[DomainId].push_back(curVtxPtr);
					}
				}
			}
		}

		printf("Domain [%d] Boundary Condition %d.\n",DomainId,m_vecForceBoundaryCondition[DomainId].size());
	}

	void ElasticSimulation_Couple::createDCBoundaryCondition(const int DomainId)
	{
		if (0 != strcmp(g_strMeshId.c_str(),"steak"))
		{
			printf("createDCBoundaryCondition do not support model %s.\n",g_strMeshId.c_str());
			MyExit;
		}

		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			if ( (curCellPtr->getDomainId()) == DomainId )
			{
				for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
				{
					VertexPtr curVtxPtr = curCellPtr->getVertex(v);
					const MyPoint & pos = curVtxPtr->getPos();
					if (isDCCondition(DomainId,pos) )
					{
						m_vecDCBoundaryCondition[DomainId].push_back(curVtxPtr);
					}
				}
			}
		}

		printf("Domain [%d] Boundary Condition %d.\n",DomainId,m_vecDCBoundaryCondition[DomainId].size());
		//MyPause;
	}

	void ElasticSimulation_Couple::simulation()
	{
		if (m_isSimulate)
		{
			for (unsigned id=0;id < Cell::LocalDomainCount;++id)
			{
				//printf("update_rhs\n");
				update_rhs(0,id);

				apply_boundary_values(id);
				//printf("solve_linear_problem\n");
				solve_linear_problem (id);
				//printf("update_u_v_a\n");
				update_u_v_a(id);
			}
		}
	}

	void ElasticSimulation_Couple::simulation_Couple_EFG()
	{
		for (unsigned id=0;id < Cell::CoupleDomainCount;++id)
		{
			update_rhs_Couple_EFG(id);
		}
		applyOutBoundaryValueToInner();
		for (unsigned id=0;id < Cell::CoupleDomainCount;++id)
		{
			solve_linear_problem_Couple_EFG (id);
		}
	}

	void ElasticSimulation_Couple::computeCoupleForceForNewmark()
	{
		for (unsigned id=0;id < Cell::LocalDomainCount;++id)
		{
			R_rhs_distance[id].setZero();
		}

		const MyInt nVtxSize = Vertex::getVertexSize();
		for (MyInt v=0;v<nVtxSize;++v)
		{
			VertexPtr curVtxPtr = Vertex::getVertex(v);
			if (curVtxPtr->isValidCoupleDof(0))
			{
				const MyInt EFG_Id = 0;
				const MyVectorI& curEfgDofs = curVtxPtr->getCoupleDof(EFG_Id);

				const std::map< MyInt,MyVectorI >& refLocalDofMap = curVtxPtr->getLocalDofsMap();
				std::map< MyInt,MyVectorI >::const_iterator ci=refLocalDofMap.begin();
				std::map< MyInt,MyVectorI >::const_iterator endc=refLocalDofMap.end();
				for (;ci != endc;++ci)
				{
					const MyInt FEM_Id = ci->first;
					const MyVectorI& curFemDofs = ci->second;

					for (MyInt v=0;v<dim;++v)
					{
						R_rhs_distance[FEM_Id][curFemDofs[v]] = incremental_displacement_EFG[EFG_Id][curEfgDofs[v]] - incremental_displacement[FEM_Id][curFemDofs[v]];
						//printf("R_rhs_distance[%d][%d] = %f,(%f)-(%f) \n",FEM_Id,curFemDofs[v],R_rhs_distance[FEM_Id][curFemDofs[v]],incremental_displacement_EFG[EFG_Id][curEfgDofs[v]],incremental_displacement[FEM_Id][curFemDofs[v]]);
					}
				}
			}

		}
		
		/*const MyInt nInnerSize = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size();
		
		for (MyInt i=0;i<nInnerSize;++i)
		{
			const MyVectorI& curNode = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			const MyInt FEM_Id = curNode[1];
			const MyInt EFG_Id = curNode[2];
			VertexPtr curVtxPtr = Vertex::getVertex(vtxId);
			const MyVectorI& curFemDofs = curVtxPtr->getLocalDof(FEM_Id);
			const MyVectorI& curEfgDofs = curVtxPtr->getCoupleDof(EFG_Id);

			for (MyInt v=0;v<dim;++v)
			{
				R_rhs_distance[FEM_Id][curFemDofs[v]] = incremental_displacement_EFG[EFG_Id][curEfgDofs[v]] - incremental_displacement[FEM_Id][curFemDofs[v]];
			}
		}*/

		for (unsigned id=0;id < Cell::LocalDomainCount;++id)
		{

			//R_rhs_distanceForce[id] = m_global_MassMatrix/*m_computeMatrix*/[id] * R_rhs_distance[id] * m_2_div_timeX2;
			R_rhs_distanceForce[id] = m_computeMatrix[id] * R_rhs_distance[id];
		}
		//MyPause;
	}

	void ElasticSimulation_Couple::simulation_Couple_FEM_EFG()
	{
		static MyInt nStep=0;
		if (m_isSimulate)
		{
			nStep++;
			for (unsigned id=0;id < Cell::LocalDomainCount;++id)
			{
				update_rhs(nStep,id);
				apply_boundary_values(id);
				solve_linear_problem (id);//output is fake
				//update_u_v_a(id);
			}

			getOutBoundaryValue();
			for (unsigned id=0;id < Cell::CoupleDomainCount;++id)
			{
				update_rhs_Couple_EFG(id);
			}
			applyOutBoundaryValueToInner();
			for (unsigned id=0;id < Cell::CoupleDomainCount;++id)
			{
				solve_linear_problem_Couple_EFG (id);
			}
			getInnerBoundaryValue();
			
			computeCoupleForceForNewmark();
			for (unsigned id=0;id < Cell::LocalDomainCount;++id)
			{
				//printf("update_rhs\n");
				//update_u_v_a_forNermark(id);
				update_rhs(nStep,id);
				R_rhs[id] += R_rhs_distanceForce[id];
				R_rhs_distance[id].setZero();
			}
			pushMatrix();
			//applyInnerBoundaryValueToOut();
			for (unsigned id=0;id < Cell::LocalDomainCount;++id)
			{
				apply_boundary_values(id);
				//printf("solve_linear_problem\n");
				solve_linear_problem (id);
				//printf("update_u_v_a\n");
				update_u_v_a(id);
			}
			popMatrix();
		}
	}

	void ElasticSimulation_Couple::pushMatrix()
	{
		//for (unsigned id=0;id < Cell::CoupleDomainCount;++id)
		for (MyInt i=0;i<Cell::LocalDomainCount;++i)
		{
			m_computeMatrix_backup[i] = m_computeMatrix[i];
		}
	}

	void ElasticSimulation_Couple::popMatrix()
	{
		for (MyInt i=0;i<Cell::LocalDomainCount;++i)
		{
			m_computeMatrix[i] = m_computeMatrix_backup[i];
		}
	}

	void ElasticSimulation_Couple::getOutBoundaryValue()
	{
		const MyInt nOutSize = m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId.size();
		m_vecOutBoundaryDof2Value.resize(nOutSize);
		for (MyInt i=0;i<nOutSize;++i)
		{
			const MyVectorI& curNode = m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			const MyInt FEM_Id = curNode[1];
			const MyInt EFG_Id = curNode[2];
			VertexPtr curVtxPtr = Vertex::getVertex(vtxId);
			const MyVectorI& curFemDofs = curVtxPtr->getLocalDof(FEM_Id);
			const MyVectorI& curEfgDofs = curVtxPtr->getCoupleDof(EFG_Id);
			MyDenseVector curDisplacement(incremental_displacement[FEM_Id][curFemDofs[0]],
										  incremental_displacement[FEM_Id][curFemDofs[1]],
										  incremental_displacement[FEM_Id][curFemDofs[2]]);
			//printf("{%d,%d,%d,%f,%f,%f},",curFemDofs[0],curFemDofs[1],curFemDofs[2],curDisplacement[0],curDisplacement[1],curDisplacement[2]);
			m_vecOutBoundaryDof2Value[i] = std::make_pair(curEfgDofs,curDisplacement);
		}
		//printf("\n");
		//MyPause;
	}

	void ElasticSimulation_Couple::getOutBoundaryValue_Force()
	{
		for (unsigned id=0;id < Cell::CoupleDomainCount;++id)
		{
			R_rhs_EFG[id].setZero();
			//R_rhs_EFG[id] = m_computeRhs_EFG[id];
		}
		//return ;
		for (unsigned id=0;id < Cell::LocalDomainCount;++id)
		{
			update_rhs(0,id);
		}

		const MyInt nForceCoupleSize = m_vecForceCoupleNode.size();
		for (MyInt i=0;i<nForceCoupleSize;++i)
		{
			const ForceCoupleNode& curNode =  m_vecForceCoupleNode[i];

			for (MyInt j=0;j<dim;++j)
			{
				R_rhs_EFG[curNode.m_nCoupleDomainId][curNode.m_vecCoupleDomainDofs[j]] += (R_rhs[curNode.m_nLocalDomainId][curNode.m_vecLocalDomainDofs[j]]/9.f);
			}
			
		}
	}

	void ElasticSimulation_Couple::getInnerBoundaryValueForce()
	{
		const MyInt nInnerSize = m_vecInnerBoundary_Force_VtxId_FEMDomainId_EFGDomainId.size();
		m_vecInnerBoundaryDof2Value.resize(nInnerSize);
		for (MyInt i=0;i<nInnerSize;++i)
		{
			const MyVectorI& curNode = m_vecInnerBoundary_Force_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			const MyInt FEM_Id = curNode[1];
			const MyInt EFG_Id = curNode[2];
			VertexPtr curVtxPtr = Vertex::getVertex(vtxId);
			const MyVectorI& curFemDofs = curVtxPtr->getLocalDof(FEM_Id);
			const MyVectorI& curEfgDofs = curVtxPtr->getCoupleDof(EFG_Id);
			MyDenseVector curDisplacement(incremental_displacement_EFG[EFG_Id][curEfgDofs[0]],
				incremental_displacement_EFG[EFG_Id][curEfgDofs[1]],
				incremental_displacement_EFG[EFG_Id][curEfgDofs[2]]);
			//printf("{%f,%f,%f},",curDisplacement[0],curDisplacement[1],curDisplacement[2]);
			m_vecInnerBoundaryDof2Value[i] = std::make_pair(curFemDofs,curDisplacement);
		}
	}

	void ElasticSimulation_Couple::getInnerBoundaryValue()
	{
		const MyInt nInnerSize = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size();
		m_vecInnerBoundaryDof2Value.resize(nInnerSize);
		for (MyInt i=0;i<nInnerSize;++i)
		{
			const MyVectorI& curNode = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			const MyInt FEM_Id = curNode[1];
			const MyInt EFG_Id = curNode[2];
			VertexPtr curVtxPtr = Vertex::getVertex(vtxId);
			const MyVectorI& curFemDofs = curVtxPtr->getLocalDof(FEM_Id);
			const MyVectorI& curEfgDofs = curVtxPtr->getCoupleDof(EFG_Id);
			MyDenseVector curDisplacement(incremental_displacement_EFG[EFG_Id][curEfgDofs[0]],
										  incremental_displacement_EFG[EFG_Id][curEfgDofs[1]],
										  incremental_displacement_EFG[EFG_Id][curEfgDofs[2]]);
			//printf("{%f,%f,%f},",curDisplacement[0],curDisplacement[1],curDisplacement[2]);
			m_vecInnerBoundaryDof2Value[i] = std::make_pair(curFemDofs,curDisplacement);
		}
		//printf("\n");
	}

	void ElasticSimulation_Couple::applyOutBoundaryValueToInner()
	{
		const MyInt nOutSize = m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId.size();
		/*printf("nOutSize %d\n",nOutSize);
		MyPause;*/
		for (MyInt i=0;i<nOutSize;++i)
		{
			const MyVectorI& curNode = m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			//const MyInt FEM_Id = curNode[1];
			const MyInt EFG_Id = curNode[2];

			const MyVectorI& curEfgDofs = m_vecOutBoundaryDof2Value[i].first;
			const MyDenseVector& curDisp = m_vecOutBoundaryDof2Value[i].second;

			{
				
				MySpMat&  computeMatrix = m_computeMatrix_EFG[EFG_Id];
				MyVector& curRhs = R_rhs_EFG[EFG_Id];
				MyVector& curDisplacement = incremental_displacement_EFG[EFG_Id];
				


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

				for (unsigned c=0;c<dim;++c)
				{
					const unsigned int dof_number = curEfgDofs[c];
					const MyFloat dof_value		  = curDisp[c];
					setMatrixRowZeroWithoutDiag(computeMatrix,dof_number);

					MyFloat new_rhs;
					if ( !numbers::isZero(diagnosticValue[dof_number] ) )
					{
						new_rhs = dof_value * diagnosticValue[dof_number];
						curRhs(dof_number) = new_rhs;
					}
					else
					{
						computeMatrix.coeffRef(dof_number,dof_number) = first_nonzero_diagonal_entry;
						new_rhs = dof_value * first_nonzero_diagonal_entry;
						curRhs(dof_number) = new_rhs;
					}
					curDisplacement(dof_number) = dof_value;
				}	
			}
		}
	}

	void ElasticSimulation_Couple::applyInnerForceValueToOut()
	{
		const MyFloat c_SpringConstant = 2.0f;
		MyVector RightHandValue_24_1(24);
		/*typedef std::pair< MyInt,MyInt > pair_DomainId_VtxId;
		std::map<pair_DomainId_VtxId,bool> idMap;*/

		std::vector<int> vecDofs;
		MyVector g_springForce[Cell::LocalDomainCount];
		std::map< int, MyDenseVector > mapSpringForceList[Cell::LocalDomainCount];
		for (MyInt i=0;i<Cell::LocalDomainCount;++i)
		{
			g_springForce[i].resize(R_rhs[i].size());
			g_springForce[i].setZero();
		}

		const MyInt nInnerSize = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size();
		for (MyInt i=0;i<nInnerSize;++i)
		{
			const MyVectorI& curNode = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			const MyInt FEM_Id = curNode[1];

			std::map< int, MyDenseVector >& mapSpringForce = mapSpringForceList[FEM_Id];

			const MyVectorI& curFemDofs = m_vecInnerBoundaryDof2Value[i].first;
			const MyDenseVector& curDisp = m_vecInnerBoundaryDof2Value[i].second;

			{
				/*MySpMat&  computeMatrix = m_computeMatrix[FEM_Id];
				MyVector& curRhs = R_rhs[FEM_Id];*/
				MyVector& curDisplacement = incremental_displacement[FEM_Id];

				MyDenseVector distance = MyDenseVector( curDisp[0]-curDisplacement[curFemDofs[0]],
														curDisp[1]-curDisplacement[curFemDofs[1]],
														curDisp[2]-curDisplacement[curFemDofs[2]]);
				mapSpringForce[vtxId] = (c_SpringConstant * distance * 200000000);

				/*printf("distance (%f,%f,%f)\n",distance[0],distance[1],distance[2]);
				MyPause;*/
			}
		}

		const MyInt nCellSize = Cell::getCellSize();
		for (MyInt c=0;c<nCellSize;++c)
		{
			RightHandValue_24_1.setZero();
			CellPtr curCellPtr = Cell::getCell(c);

			const MyInt nCoupleId = curCellPtr->getCoupleDomainId();
			const MyInt FEM_Id = curCellPtr->getDomainId();

			if ( Invalid_Id != (curCellPtr->getCoupleDomainId()) )
			{
				bool bFlag = false;
				std::map< int, MyDenseVector >& mapSpringForce = mapSpringForceList[FEM_Id];

				for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
				{
					const int vertexId = curCellPtr->getVertex(v)->getId();
					if (mapSpringForce.find(vertexId) != mapSpringForce.end())
					{
						MyMatrix tmpShapeMatrix(3,24);
						tmpShapeMatrix.setZero();
						for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
						{
							const unsigned col = k*dim;
							for (unsigned i=0;i<dim;++i)
							{
								tmpShapeMatrix.coeffRef(0,col+0) = curCellPtr->shapeFunctionValue_8_8.coeff(v,k);
								tmpShapeMatrix.coeffRef(1,col+1) = curCellPtr->shapeFunctionValue_8_8.coeff(v,k);
								tmpShapeMatrix.coeffRef(2,col+2) = curCellPtr->shapeFunctionValue_8_8.coeff(v,k);
							}
						}
						RightHandValue_24_1 +=  tmpShapeMatrix.transpose() * mapSpringForce.at(vertexId) * curCellPtr->getJxW(v);
						bFlag = true;
					}
				}

				if (bFlag)
				{
					/*std::cout << RightHandValue_24_1 << std::endl;
					MyPause;*/
					curCellPtr->setCellType(FEM);
					curCellPtr->get_dof_indices_Local_FEM(FEM_Id,vecDofs);;
					/*printf("vecDofs.size %d\n",vecDofs.size());
					MyPause;*/
					for (unsigned v=0;v<vecDofs.size();++v)
					{
						/*printf("vecDofs[%d] %d\n",v,vecDofs[v]);
						MyPause;*/
						g_springForce[FEM_Id][vecDofs[v]] += RightHandValue_24_1[v];
					}
					/*printf("FEMID %d, ForceNorm %f\n",FEM_Id,g_springForce[FEM_Id].norm());
					MyPause;*/
				}				
			}
			
			
		}

		for (MyInt i=0;i<Cell::LocalDomainCount;++i)
		{
			/*printf("R_rhs.norm [%f] ,force.norm [%f]\n",R_rhs[i].norm(),g_springForce[i].norm());
			MyPause;*/
			R_rhs[i] += g_springForce[i];
		}
	}

	void ElasticSimulation_Couple::applyInnerBoundaryValueToOutForce()
	{
		const MyInt nInnerSize = m_vecInnerBoundary_Force_VtxId_FEMDomainId_EFGDomainId.size();
		for (MyInt i=0;i<nInnerSize;++i)
		{
			const MyVectorI& curNode = m_vecInnerBoundary_Force_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			const MyInt FEM_Id = curNode[1];

			const MyVectorI& curFemDofs = m_vecInnerBoundaryDof2Value[i].first;
			const MyDenseVector& curDisp = m_vecInnerBoundaryDof2Value[i].second;

			{
				MySpMat&  computeMatrix = m_computeMatrix[FEM_Id];
				MyVector& curRhs = R_rhs[FEM_Id];
				MyVector& curDisplacement = incremental_displacement[FEM_Id];

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

				for (unsigned c=0;c<dim;++c)
				{
					const unsigned int dof_number = curFemDofs[c];
					const MyFloat dof_value		  = curDisp[c];
					//mapDof[dof_number] = true;
					setMatrixRowZeroWithoutDiag(computeMatrix,dof_number);

					MyFloat new_rhs;
					if ( !numbers::isZero(diagnosticValue[dof_number] ) )
					{
						new_rhs = dof_value * diagnosticValue[dof_number];
						curRhs(dof_number) = new_rhs;
					}
					else
					{
						computeMatrix.coeffRef(dof_number,dof_number) = first_nonzero_diagonal_entry;
						new_rhs = dof_value * first_nonzero_diagonal_entry;
						curRhs(dof_number) = new_rhs;
					}
					curDisplacement(dof_number) = dof_value;
				}	
			}
		}
	}

	void ElasticSimulation_Couple::applyInnerBoundaryValueToOut()
	{
		const MyInt nInnerSize = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size();
		for (MyInt i=0;i<nInnerSize;++i)
		{
			const MyVectorI& curNode = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId[i];
			const MyInt vtxId = curNode[0];
			const MyInt FEM_Id = curNode[1];

			const MyVectorI& curFemDofs = m_vecInnerBoundaryDof2Value[i].first;
			const MyDenseVector& curDisp = m_vecInnerBoundaryDof2Value[i].second;

			{
				MySpMat&  computeMatrix = m_computeMatrix[FEM_Id];
				MyVector& curRhs = R_rhs[FEM_Id];
				MyVector& curDisplacement = incremental_displacement[FEM_Id];

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

				for (unsigned c=0;c<dim;++c)
				{
					const unsigned int dof_number = curFemDofs[c];
					const MyFloat dof_value		  = curDisp[c];

					
					//printf("[%f],",dof_value);
					//mapDof[dof_number] = true;
					setMatrixRowZeroWithoutDiag(computeMatrix,dof_number);

					MyFloat new_rhs;
					if ( !numbers::isZero(diagnosticValue[dof_number] ) )
					{
						new_rhs = dof_value * diagnosticValue[dof_number];
						curRhs(dof_number) = new_rhs;
					}
					else
					{
						computeMatrix.coeffRef(dof_number,dof_number) = first_nonzero_diagonal_entry;
						new_rhs = dof_value * first_nonzero_diagonal_entry;
						curRhs(dof_number) = new_rhs;
					}
					curDisplacement(dof_number) = dof_value;
				}
			}
		}
		
	}

	void ElasticSimulation_Couple::setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx )
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

	void ElasticSimulation_Couple::apply_boundary_values(const MyInt DomainId)
	{
		std::vector< VertexPtr >& vecBoundaryVtx = m_vecDCBoundaryCondition[DomainId];
		MySpMat&  computeMatrix = m_computeMatrix[DomainId];
		MyVector curRhs = R_rhs[DomainId];
		MyVector curDisplacement = incremental_displacement[DomainId];
		printf("apply_boundary_values begin \n");
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
			const MyVectorI& Dofs = curVtxPtr->getLocalDof(DomainId);
			for (unsigned c=0;c<dim;++c)
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

	void ElasticSimulation_Couple::update_rhs_forNewMark(MyInt nStep,MyInt id)
	{
		R_rhs[id].setZero();
		mass_rhs[id].setZero();
		damping_rhs[id].setZero();

		//incremental_displacement,velocity,acceleration
		mass_rhs[id] += m_db_NewMarkConstant[0] * (displacement_newmark[id]);
		mass_rhs[id] += m_db_NewMarkConstant[2] * velocity_newmark[id];
		mass_rhs[id] += m_db_NewMarkConstant[3] * acceleration_newmark[id];

		damping_rhs[id] += m_db_NewMarkConstant[1] * (displacement_newmark[id]);
		damping_rhs[id] += m_db_NewMarkConstant[4] * velocity_newmark[id];
		damping_rhs[id] += m_db_NewMarkConstant[5] * acceleration_newmark[id];

		R_rhs[id] += m_computeRhs[id];

		R_rhs[id] += m_global_MassMatrix[id] * mass_rhs[id];
		R_rhs[id] += m_global_DampingMatrix[id] * damping_rhs[id];

		if (nStep > 10 && nStep < 15)
		{
			R_rhs[id] += R_rhs_externalForce[id];
			//MyPause;
		}
	}

	void ElasticSimulation_Couple::update_rhs(MyInt nStep,MyInt id)
	{
		R_rhs[id].setZero();
		mass_rhs[id].setZero();
		damping_rhs[id].setZero();

		//incremental_displacement,velocity,acceleration
		mass_rhs[id] += m_db_NewMarkConstant[0] * (displacement[id]);
		mass_rhs[id] += m_db_NewMarkConstant[2] * velocity[id];
		mass_rhs[id] += m_db_NewMarkConstant[3] * acceleration[id];

		damping_rhs[id] += m_db_NewMarkConstant[1] * (displacement[id]);
		damping_rhs[id] += m_db_NewMarkConstant[4] * velocity[id];
		damping_rhs[id] += m_db_NewMarkConstant[5] * acceleration[id];

		R_rhs[id] += m_computeRhs[id];

		R_rhs[id] += m_global_MassMatrix[id] * mass_rhs[id];
		R_rhs[id] += m_global_DampingMatrix[id] * damping_rhs[id];

		if (nStep > 10 && nStep < 15)
		{
			R_rhs[id] += R_rhs_externalForce[id];
			//MyPause;
		}
	}

	void ElasticSimulation_Couple::update_rhs_inertia(MyInt id)
	{
		R_rhs[id].setZero();
		mass_rhs[id].setZero();
		damping_rhs[id].setZero();

		//incremental_displacement,velocity,acceleration
		mass_rhs[id] += m_db_NewMarkConstant[0] * displacement[id];
		mass_rhs[id] += m_db_NewMarkConstant[2] * velocity[id];
		mass_rhs[id] += m_db_NewMarkConstant[3] * acceleration[id];

		damping_rhs[id] += m_db_NewMarkConstant[1] * displacement[id];
		damping_rhs[id] += m_db_NewMarkConstant[4] * velocity[id];
		damping_rhs[id] += m_db_NewMarkConstant[5] * acceleration[id];

		R_rhs[id] += m_computeRhs[id];

		R_rhs[id] += m_global_MassMatrix[id] * mass_rhs[id];
		R_rhs[id] += m_global_DampingMatrix[id] * damping_rhs[id];
	}

	void ElasticSimulation_Couple::update_u_v_a(MyInt id)
	{
		const MyVector& solu = incremental_displacement[id];
		MyVector& disp_vec = displacement[id];
		MyVector& vel_vec = velocity[id];
		MyVector& acc_vec = acceleration[id];
		MyVector& old_acc = old_acceleration[id];
		MyVector& old_solu = old_displacement[id];

		old_solu = disp_vec;
		disp_vec = solu;
		old_acc  = acc_vec;

		acc_vec *= (-1 * m_db_NewMarkConstant[3]);//    acc_vec.scale(-1 * m_db_NewMarkConstant[3]);
		acc_vec += (disp_vec * m_db_NewMarkConstant[0]); //acc_vec.add(m_db_NewMarkConstant[0], disp_vec);
		acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));//acc_vec.add(-1 * m_db_NewMarkConstant[0], old_solu);
		acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));//acc_vec.add(-1 * m_db_NewMarkConstant[2],vel_vec);

		vel_vec += (old_acc * m_db_NewMarkConstant[6]);//vel_vec.add(m_db_NewMarkConstant[6],old_acc);
		vel_vec += (acc_vec * m_db_NewMarkConstant[7]);//vel_vec.add(m_db_NewMarkConstant[7],acc_vec);
	}

	void ElasticSimulation_Couple::update_u_v_a_forNermark(MyInt id)
	{
		const MyVector& solu = incremental_displacement[id];
		MyVector& disp_vec = displacement_newmark[id];

		velocity_newmark[id] = velocity[id];
		MyVector& vel_vec = velocity_newmark[id];

		acceleration_newmark[id] = acceleration[id];
		MyVector& acc_vec = acceleration_newmark[id];
		/*MyVector old_acc;
		MyVector old_solu;*/

		MyVector old_solu = disp_vec;
		disp_vec = solu+R_rhs_distance[id];
		MyVector old_acc  = acc_vec;

		acc_vec *= (-1 * m_db_NewMarkConstant[3]);//    acc_vec.scale(-1 * m_db_NewMarkConstant[3]);
		acc_vec += (disp_vec * m_db_NewMarkConstant[0]); //acc_vec.add(m_db_NewMarkConstant[0], disp_vec);
		acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));//acc_vec.add(-1 * m_db_NewMarkConstant[0], old_solu);
		acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));//acc_vec.add(-1 * m_db_NewMarkConstant[2],vel_vec);

		vel_vec += (old_acc * m_db_NewMarkConstant[6]);//vel_vec.add(m_db_NewMarkConstant[6],old_acc);
		vel_vec += (acc_vec * m_db_NewMarkConstant[7]);//vel_vec.add(m_db_NewMarkConstant[7],acc_vec);
	}

	void ElasticSimulation_Couple::solve_linear_problem(MyInt id)
	{
		SolverControl           solver_control (m_vecLocalDof[id],  1e-3*numbers::l2_norm(R_rhs[id]));
		SolverCG              cg (solver_control);

		PreconditionSSOR preconditioner;
		preconditioner.initialize(m_computeMatrix[id], 1.2);

		cg.solve (m_computeMatrix[id], incremental_displacement[id], R_rhs[id],	preconditioner);

	}

	void ElasticSimulation_Couple::update_rhs_Couple_EFG(MyInt id)
	{
#if 0

		R_rhs_EFG[id].setZero();
		mass_rhs_EFG[id].setZero();
		damping_rhs_EFG[id].setZero();

		//incremental_displacement,velocity,acceleration
		mass_rhs_EFG[id] += m_db_NewMarkConstant[0] * displacement_EFG[id];
		mass_rhs_EFG[id] += m_db_NewMarkConstant[2] * velocity_EFG[id];
		mass_rhs_EFG[id] += m_db_NewMarkConstant[3] * acceleration_EFG[id];

		damping_rhs_EFG[id] += m_db_NewMarkConstant[1] * displacement_EFG[id];
		damping_rhs_EFG[id] += m_db_NewMarkConstant[4] * velocity_EFG[id];
		damping_rhs_EFG[id] += m_db_NewMarkConstant[5] * acceleration_EFG[id];

		R_rhs_EFG[id] += m_computeRhs_EFG[id];

		R_rhs_EFG[id] += m_global_MassMatrix_EFG[id] * mass_rhs_EFG[id];
		R_rhs_EFG[id] += m_global_DampingMatrix_EFG[id] * damping_rhs_EFG[id];
#else
		R_rhs_EFG[id].setZero();
		R_rhs_EFG[id] = m_computeRhs_EFG[id];
		/*std::cout << m_computeRhs_EFG[id] << std::endl;
		MyPause;*/
#endif
	}

	void ElasticSimulation_Couple::update_u_v_a_Couple_EFG(MyInt id)
	{
		const MyVector& solu = incremental_displacement_EFG[id];
		MyVector& disp_vec = displacement_EFG[id];
		MyVector& vel_vec = velocity_EFG[id];
		MyVector& acc_vec = acceleration_EFG[id];
		MyVector& old_acc = old_acceleration_EFG[id];
		MyVector& old_solu = old_displacement_EFG[id];

		old_solu = disp_vec;
		disp_vec = solu;
		old_acc  = acc_vec;

		acc_vec *= (-1 * m_db_NewMarkConstant[3]);//    acc_vec.scale(-1 * m_db_NewMarkConstant[3]);
		acc_vec += (disp_vec * m_db_NewMarkConstant[0]); //acc_vec.add(m_db_NewMarkConstant[0], disp_vec);
		acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));//acc_vec.add(-1 * m_db_NewMarkConstant[0], old_solu);
		acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));//acc_vec.add(-1 * m_db_NewMarkConstant[2],vel_vec);

		vel_vec += (old_acc * m_db_NewMarkConstant[6]);//vel_vec.add(m_db_NewMarkConstant[6],old_acc);
		vel_vec += (acc_vec * m_db_NewMarkConstant[7]);//vel_vec.add(m_db_NewMarkConstant[7],acc_vec);
	}

	void ElasticSimulation_Couple::solve_linear_problem_Couple_EFG(MyInt id)
	{
		SolverControl           solver_control (m_vecCoupleDof[id],  1e-3*numbers::l2_norm(R_rhs_EFG[id]));
		SolverCG              cg (solver_control);

		PreconditionSSOR preconditioner;
		preconditioner.initialize(m_computeMatrix_EFG[id], 1.2);

		cg.solve (m_computeMatrix_EFG[id], incremental_displacement_EFG[id], R_rhs_EFG[id],	preconditioner);
	}


	void ElasticSimulation_Couple::createNewMarkMatrix(const int DomainId)
	{
		m_computeMatrix[DomainId] = m_global_StiffnessMatrix[DomainId];
		m_computeMatrix[DomainId] += m_db_NewMarkConstant[0] * m_global_MassMatrix[DomainId];
		m_computeMatrix[DomainId] += m_db_NewMarkConstant[1] * m_global_DampingMatrix[DomainId];

		incremental_displacement[DomainId].setZero();
		velocity[DomainId].setZero();
		acceleration[DomainId].setZero();
		displacement[DomainId].setZero();
		old_acceleration[DomainId].setZero();
		old_displacement[DomainId].setZero();

		/*std::ofstream outfile("d:\\aa.txt");
		outfile << m_computeMatrix[DomainId];
		outfile.close();*/
		return ;
	}

	void ElasticSimulation_Couple::createNewMarkMatrix_Couple_EFG(const int DomainId)
	{
		m_computeMatrix_EFG[DomainId] = m_global_StiffnessMatrix_EFG[DomainId];
		m_computeMatrix_EFG[DomainId] += m_db_NewMarkConstant[0] * m_global_MassMatrix_EFG[DomainId];
		m_computeMatrix_EFG[DomainId] += m_db_NewMarkConstant[1] * m_global_DampingMatrix_EFG[DomainId];

		incremental_displacement_EFG[DomainId].setZero();
		velocity_EFG[DomainId].setZero();
		acceleration_EFG[DomainId].setZero();
		displacement_EFG[DomainId].setZero();
		old_acceleration_EFG[DomainId].setZero();
		old_displacement_EFG[DomainId].setZero();
		return ;
	}

	void ElasticSimulation_Couple::createGlobalMassAndStiffnessAndDampingMatrixFEM(const int DomainId)
	{
		const int nDof = m_vecLocalDof[DomainId];
		m_computeMatrix[DomainId].resize(nDof,nDof);
		m_global_MassMatrix[DomainId].resize(nDof,nDof);
		m_global_StiffnessMatrix[DomainId].resize(nDof,nDof);
		m_global_DampingMatrix[DomainId].resize(nDof,nDof);
		m_computeRhs[DomainId].resize(nDof);

		{
			R_rhs[DomainId].resize(nDof);
			R_rhs_distance[DomainId].resize(nDof);
			R_rhs_distanceForce[DomainId].resize(nDof);
			R_rhs_externalForce[DomainId].resize(nDof);
			mass_rhs[DomainId].resize(nDof);
			damping_rhs[DomainId].resize(nDof);
			displacement[DomainId].resize(nDof);
			velocity[DomainId].resize(nDof);
			acceleration[DomainId].resize(nDof);

			displacement_newmark[DomainId].resize(nDof);
			velocity_newmark[DomainId].resize(nDof);
			acceleration_newmark[DomainId].resize(nDof);

			old_acceleration[DomainId].resize(nDof);
			old_displacement[DomainId].resize(nDof);
			incremental_displacement[DomainId].resize(nDof);

			R_rhs[DomainId].setZero();
			R_rhs_distance[DomainId].setZero();
			R_rhs_externalForce[DomainId].setZero();
			R_rhs_distanceForce[DomainId].setZero();
			mass_rhs[DomainId].setZero();
			damping_rhs[DomainId].setZero();
			displacement[DomainId].setZero();
			velocity[DomainId].setZero();
			acceleration[DomainId].setZero();

			displacement_newmark[DomainId].setZero();
			velocity_newmark[DomainId].setZero();
			acceleration_newmark[DomainId].setZero();

			old_acceleration[DomainId].setZero();
			old_displacement[DomainId].setZero();
			incremental_displacement[DomainId].setZero();
		}

		m_computeMatrix[DomainId].setZero();
		m_global_MassMatrix[DomainId].setZero();
		m_global_StiffnessMatrix[DomainId].setZero();
		m_global_DampingMatrix[DomainId].setZero();
		m_computeRhs[DomainId].setZero();



		std::map<long,std::map<long,TripletNode > >& StiffTripletNodeMap = Cell::m_TripletNode_LocalStiffness[DomainId];
		std::map<long,std::map<long,TripletNode > >& MassTripletNodeMap = Cell::m_TripletNode_LocalMass[DomainId];
		std::map<long,TripletNode >& RhsTripletNode = Cell::m_TripletNode_LocalRhs[DomainId];

		for (unsigned i=0;i<Cell::getCellSize();++i)
		{
			CellPtr curCellPtr = Cell::getCell(i);
			const int curDomainId = curCellPtr->getDomainId();

			if (DomainId == curDomainId)
			{
				curCellPtr->initialize_Couple();
				curCellPtr->assembleSystemMatrix();
				/*curCellPtr->compressStiffnessMatrixFEM(StiffTripletNodeMap);
				curCellPtr->compressMassMatrixFEM(MassTripletNodeMap);
				curCellPtr->compressRhsFEM(RhsTripletNode);*/
			}
		}
		//assemble global stiffness matrix
		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
		std::map<long,std::map<long,TripletNode > >::const_iterator itr_tri =  StiffTripletNodeMap.begin();
		for (;itr_tri != StiffTripletNodeMap.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}
		m_global_StiffnessMatrix[DomainId].setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		StiffTripletNodeMap.clear();
		vec_triplet.clear();

		//assemble global mass matrix

		itr_tri =  MassTripletNodeMap.begin();
		for (;itr_tri != MassTripletNodeMap.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}
		m_global_MassMatrix[DomainId].setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		MassTripletNodeMap.clear();
		vec_triplet.clear();

		//assemble global rhs vector
		std::map<long,TripletNode >::const_iterator itrRhs = RhsTripletNode.begin();
		m_computeRhs[DomainId].setZero();
		for (;itrRhs != RhsTripletNode.end();++itrRhs)
		{
			m_computeRhs[DomainId][itrRhs->first] = (itrRhs->second).val;
		}
		RhsTripletNode.clear();

		for (MyInt c=0;c<VR_FEM::Cell::LocalDomainCount;++c)
		{
			std::vector< VertexPtr >& refVec = m_vecForceBoundaryCondition[c];
			MyVector& refRhs = R_rhs_externalForce[c];
			for (MyInt v=0;v<refVec.size();++v)
			{
				MyInt curDof = refVec[v]->getLocalDof(c).y();
				refRhs[curDof] = Material::Density * Material::GravityFactor*80*-0.0001f;
			}
		}

		m_global_DampingMatrix[DomainId] = Material::damping_alpha * m_global_MassMatrix[DomainId] + Material::damping_beta * m_global_StiffnessMatrix[DomainId];
	}

	void ElasticSimulation_Couple::createGlobalMassAndStiffnessAndDampingMatrixEFG(const int DomainId)
	{
		//MySpMat m_computeMatrix_EFG[Cell::CoupleDomainCount],m_global_MassMatrix_EFG[Cell::CoupleDomainCount],m_global_StiffnessMatrix_EFG[Cell::CoupleDomainCount],m_global_DampingMatrix_EFG[Cell::CoupleDomainCount];
		//MyVector m_computeRhs_EFG[Cell::CoupleDomainCount], R_rhs_EFG[Cell::CoupleDomainCount],mass_rhs_EFG[Cell::CoupleDomainCount],damping_rhs_EFG[Cell::CoupleDomainCount],displacement_EFG[Cell::CoupleDomainCount],velocity_EFG[Cell::CoupleDomainCount],acceleration_EFG[Cell::CoupleDomainCount],old_acceleration_EFG[Cell::CoupleDomainCount],old_displacement_EFG[Cell::CoupleDomainCount], incremental_displacement_EFG[Cell::CoupleDomainCount];

		const int nDof = m_vecCoupleDof[DomainId];
		m_computeMatrix_EFG[DomainId].resize(nDof,nDof);
		m_global_MassMatrix_EFG[DomainId].resize(nDof,nDof);
		m_global_StiffnessMatrix_EFG[DomainId].resize(nDof,nDof);
		m_global_DampingMatrix_EFG[DomainId].resize(nDof,nDof);
		m_computeRhs_EFG[DomainId].resize(nDof);

		{
			R_rhs_EFG[DomainId].resize(nDof);
			mass_rhs_EFG[DomainId].resize(nDof);
			damping_rhs_EFG[DomainId].resize(nDof);
			displacement_EFG[DomainId].resize(nDof);
			velocity_EFG[DomainId].resize(nDof);
			acceleration_EFG[DomainId].resize(nDof);
			old_acceleration_EFG[DomainId].resize(nDof);
			old_displacement_EFG[DomainId].resize(nDof);
			incremental_displacement_EFG[DomainId].resize(nDof);

			R_rhs_EFG[DomainId].setZero();
			mass_rhs_EFG[DomainId].setZero();
			damping_rhs_EFG[DomainId].setZero();
			displacement_EFG[DomainId].setZero();
			velocity_EFG[DomainId].setZero();
			acceleration_EFG[DomainId].setZero();
			old_acceleration_EFG[DomainId].setZero();
			old_displacement_EFG[DomainId].setZero();
			incremental_displacement_EFG[DomainId].setZero();
		}

		m_computeMatrix_EFG[DomainId].setZero();
		m_global_MassMatrix_EFG[DomainId].setZero();
		m_global_StiffnessMatrix_EFG[DomainId].setZero();
		m_global_DampingMatrix_EFG[DomainId].setZero();
		m_computeRhs_EFG[DomainId].setZero();

		//computeShapeFunction_Couple_EFG
		std::map<long,std::map<long,TripletNode > >& StiffTripletNodeMap = Cell::m_TripletNode_LocalStiffness_EFG[DomainId];
		std::map<long,std::map<long,TripletNode > >& MassTripletNodeMap = Cell::m_TripletNode_LocalMass_EFG[DomainId];
		std::map<long,TripletNode >& RhsTripletNode = Cell::m_TripletNode_LocalRhs_EFG[DomainId];

		for (unsigned i=0;i<Cell::getCellSize();++i)
		{
			CellPtr curCellPtr = Cell::getCell(i);
			const int curDomainId = curCellPtr->getCoupleDomainId();

			if (DomainId == curDomainId)
			{
				curCellPtr->initialize_Couple_Joint();
				//curCellPtr->assembleSystemMatrix();
			}
		}

		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
		std::map<long,std::map<long,TripletNode > >::const_iterator itr_tri =  StiffTripletNodeMap.begin();
		for (;itr_tri != StiffTripletNodeMap.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}
		m_global_StiffnessMatrix_EFG[DomainId].setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		StiffTripletNodeMap.clear();
		vec_triplet.clear();

		//assemble global mass matrix

		itr_tri =  MassTripletNodeMap.begin();
		for (;itr_tri != MassTripletNodeMap.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}
		m_global_MassMatrix_EFG[DomainId].setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		MassTripletNodeMap.clear();
		vec_triplet.clear();

		//assemble global rhs vector
		std::map<long,TripletNode >::const_iterator itrRhs = RhsTripletNode.begin();
		m_computeRhs_EFG[DomainId].setZero();
		for (;itrRhs != RhsTripletNode.end();++itrRhs)
		{
			m_computeRhs_EFG[DomainId][itrRhs->first] = (itrRhs->second).val;
		}
		RhsTripletNode.clear();

		m_global_DampingMatrix_EFG[DomainId] = Material::damping_alpha * m_global_MassMatrix_EFG[DomainId] + Material::damping_beta * m_global_StiffnessMatrix_EFG[DomainId];
	}

	void ElasticSimulation_Couple::distributeDof_local()
	{
		m_vecLocalDof.resize(Cell::LocalDomainCount);
		for (MyInt v=0;v<m_vecLocalDof.size();++v)
		{
			m_vecLocalDof[v] = Geometry::first_dof_idx;
		}

		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			MyInt nLocalDomainId = curCellPtr->getDomainId();
			MyInt& curDof = m_vecLocalDof[nLocalDomainId];
			for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
			{
				VertexPtr curVtxPtr = curCellPtr->getVertex(v);
				if ( !(curVtxPtr->isValidLocalDof(nLocalDomainId)))
				{
					curVtxPtr->setLocalDof(nLocalDomainId,curDof,curDof+1,curDof+2);
					curDof+=3;
				}
			}
		}

		for (MyInt v=0;v<m_vecLocalDof.size();++v)
		{
			printf("local [%d] : dof %d\n",v,m_vecLocalDof[v]);
		}
	}

	void ElasticSimulation_Couple::distributeDof_Couple()
	{
		m_vecCoupleDof.resize(Cell::CoupleDomainCount);
		for (MyInt v=0;v<m_vecCoupleDof.size();++v)
		{
			m_vecCoupleDof[v] = Geometry::first_dof_idx;
		}

		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			MyInt nLocalDomainId = curCellPtr->getCoupleDomainId();
			//printf("[nLocalDomainId=%d]\n",nLocalDomainId);
			if (nLocalDomainId != Invalid_Id)
			{
				//Q_ASSERT(nLocalDomainId < m_vecCoupleDof.size());
				MyInt& curDof = m_vecCoupleDof[nLocalDomainId];
				for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
				{
					VertexPtr curVtxPtr = curCellPtr->getVertex(v);
					if ( !(curVtxPtr->isValidCoupleDof(nLocalDomainId)))
					{
						curVtxPtr->setCoupleDof(nLocalDomainId,curDof,curDof+1,curDof+2);
						curDof+=3;
					}
				}
			}
		}

		for (MyInt v=0;v<m_vecCoupleDof.size();++v)
		{
			printf("Couple [%d] : dof %d.\n",v,m_vecCoupleDof[v]);
		}
	}

	void ElasticSimulation_Couple::print()
	{
		
#if 1		
		::glPushMatrix();

			glRotatef(rotate_x, 1.0, 0.0, 0.0);
			glRotatef(rotate_y, 0.0, 1.0f, 0.0);
			//glRotatef(90,0.0, 0.0f, 1.0);
			//glRotatef(90,0.0, 0.0f, 1.0);
			//glRotatef(90,0.0, 1.0f, 0.0);
			//glTranslatef(0,-0.5f,0);
			glScalef(scaled,scaled,scaled);	
			glTranslatef(-0.2f,0.25f,0);
			/*glTranslatef(-0.5,0,0);
			glScalef(2.0f,2.0f,2.0f);*/
			
			glPointSize(5.f);
			glBegin(GL_POINTS);
			const MyInt nCellSize = m_vec_cell.size();
			for (MyInt c=0;c<nCellSize;++c)
			{
				const MyInt id = m_vec_cell[c]->getDomainId();
				//if (id != 3)
				{
					glColor3f(Colors::colorTemplage[id][0],Colors::colorTemplage[id][1],Colors::colorTemplage[id][2]);

					{
						CellPtr curCellPtr = m_vec_cell[c];
						for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
						{
							VertexPtr vtxPtr = curCellPtr->getVertex(v);
							MyVectorI& dofs = vtxPtr->getLocalDof(id);
							MyPoint& pos = vtxPtr->getPos();
							//if (pos[0]<0.85f)
							{
								glVertex3f(pos[0] + incremental_displacement[id][dofs[0]],pos[1] + incremental_displacement[id][dofs[1]],pos[2] + incremental_displacement[id][dofs[2]]);
							}
							
						}
					}
				}

				const MyInt nCoupleId = m_vec_cell[c]->getCoupleDomainId();
				if (0 == nCoupleId || 1 == nCoupleId)
				{
					glColor3f(Colors::colorTemplage[7][0],Colors::colorTemplage[7][1],Colors::colorTemplage[7][2]);

					{
						CellPtr curCellPtr = m_vec_cell[c];
						for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
						{
							VertexPtr vtxPtr = curCellPtr->getVertex(v);
							MyVectorI& dofs = vtxPtr->getCoupleDof(nCoupleId);
							MyPoint& pos = vtxPtr->getPos();
							{
								glVertex3f(pos[0] + incremental_displacement_EFG[nCoupleId][dofs[0]],pos[1] + incremental_displacement_EFG[nCoupleId][dofs[1]],pos[2] + incremental_displacement_EFG[nCoupleId][dofs[2]]);
							}

						}
					}
				}
				
			}

			//cg.solve (m_computeMatrix_EFG[id], incremental_displacement_EFG[id], R_rhs_EFG[id],	preconditioner);
			
			/*for (MyInt c=0;c<nCellSize;++c)
			{
				if (0 == m_vec_cell[c]->getCoupleDomainId() )
				{
					glColor3f(Colors::colorTemplage[3][0],Colors::colorTemplage[3][1],Colors::colorTemplage[3][2]);
					CellPtr curCellPtr = m_vec_cell[c];
					for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
					{
						VertexPtr vtxPtr = curCellPtr->getVertex(v);
						MyPoint& pos = vtxPtr->getPos();
						{
							glVertex3f(pos[0] ,pos[1] ,pos[2] );
						}
					}
				}
				if (1 == m_vec_cell[c]->getCoupleDomainId())
				{
					glColor3f(Colors::colorTemplage[4][0],Colors::colorTemplage[4][1],Colors::colorTemplage[4][2]);
					CellPtr curCellPtr = m_vec_cell[c];
					for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
					{
						VertexPtr vtxPtr = curCellPtr->getVertex(v);
						MyPoint& pos = vtxPtr->getPos();
						{
							glVertex3f(pos[0] ,pos[1] ,pos[2] );
						}
					}
				}
			}

			glColor3f(Colors::colorTemplage[5][0],Colors::colorTemplage[5][1],Colors::colorTemplage[5][2]);
			for (MyInt v=0;v<m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId.size();++v)
			{
				const MyInt curVid = m_vecOutBoundary_VtxId_FEMDomainId_EFGDomainId[v][0];
				VertexPtr vtxPtr = Vertex::getVertex(curVid);
				MyPoint& pos = vtxPtr->getPos();
				{
					glVertex3f(pos[0] ,pos[1] ,pos[2] );
				}
			}*/

			glColor3f(Colors::colorTemplage[6][0],Colors::colorTemplage[6][1],Colors::colorTemplage[6][2]);
			for (MyInt v=0;v<m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId.size();++v)
			{
				const MyInt curVid = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId[v][0];
				const MyInt nEFGid = m_vecInnerBoundary_VtxId_FEMDomainId_EFGDomainId[v][2];
				VertexPtr vtxPtr = Vertex::getVertex(curVid);
				MyPoint& pos = vtxPtr->getPos();
				MyVectorI efgDofs = vtxPtr->getCoupleDof(nEFGid);
				{
					
					glVertex3f(pos[0]+incremental_displacement_EFG[nEFGid][efgDofs[0]] ,
						       pos[1]+incremental_displacement_EFG[nEFGid][efgDofs[1]] ,
							   pos[2]+incremental_displacement_EFG[nEFGid][efgDofs[2]] );
					//printf("{%f,%f,%f},",pos[0] ,pos[1] ,pos[2]);
				}
			}
			glEnd();
		glPopMatrix();
#else

		::glPushMatrix();

			glRotatef(rotate_x, 1.0, 0.0, 0.0);
			glRotatef(rotate_y, 0.0, 1.0f, 0.0);
			glScalef(scaled,scaled,scaled);

			glPointSize(5.f);
			glBegin(GL_POINTS);
			const MyInt nCellSize = m_vec_cell.size();
			for (MyInt c=0;c<nCellSize;++c)
			{
				const MyInt id = m_vec_cell[c]->getDomainId();
				if (id != Invalid_Id && id != Cell::CoupleDomainId)
				{
					glColor3f(Colors::colorTemplage[id][0],Colors::colorTemplage[id][1],Colors::colorTemplage[id][2]);
					//if (m_vec_cell[c]->getDomainId() == 0)
					{
						CellPtr curCellPtr = m_vec_cell[c];
						for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
						{
							VertexPtr vtxPtr = curCellPtr->getVertex(v);
							MyVectorI& dofs = vtxPtr->getLocalDof(id);
							MyPoint& pos = vtxPtr->getPos();
							glVertex3f(pos[0] + incremental_displacement[id][dofs[0]],pos[1] + incremental_displacement[id][dofs[1]],pos[2] + incremental_displacement[id][dofs[2]]);
						}
					}
				}
			}
			glEnd();
		glPopMatrix();
#endif
	}

	void ElasticSimulation_Couple::generateNewMarkParam()
	{
		double g_NewMark_alpha    = .25;
		double g_NewMark_delta    = .5;
		double g_NewMark_timestep = 1./64;
		double _default_delta = g_NewMark_delta;
		double _default_alpha = g_NewMark_alpha;
		double _default_timestep = g_NewMark_timestep;
		m_db_NewMarkConstant[0] = (1.0f/(_default_alpha*_default_timestep*_default_timestep));
		m_db_NewMarkConstant[1] = (_default_delta/(_default_alpha*_default_timestep));
		m_db_NewMarkConstant[2] = (1.0f/(_default_alpha*_default_timestep));
		m_db_NewMarkConstant[3] = (1.0f/(2.0f*_default_alpha)-1.0f);
		m_db_NewMarkConstant[4] = (_default_delta/_default_alpha -1.0f);
		m_db_NewMarkConstant[5] = (_default_timestep/2.0f*(_default_delta/_default_alpha-2.0f));
		m_db_NewMarkConstant[6] = (_default_timestep*(1.0f-_default_delta));
		m_db_NewMarkConstant[7] = (_default_delta*_default_timestep);

		m_2_div_timeX2 = 2.f / (_default_timestep*_default_timestep);
	}

	bool ElasticSimulation_Couple::LoadGLTextures(const char* lpszTexturePath)
	{
		glGenTextures(1, &m_texture);					// Create The Texture

		// Typical Texture Generation Using Data From The Bitmap
		glBindTexture(GL_TEXTURE_2D, m_texture);
		unsigned char* image;
		int width, height;
		image = SOIL_load_image( lpszTexturePath, &width, &height, 0, SOIL_LOAD_RGB );
		if (image)
		{
			glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image );
			//glTexImage2D(GL_TEXTURE_2D, 0, 3, TextureImage[0]->sizeX, TextureImage[0]->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, TextureImage[0]->data);
			SOIL_free_image_data( image );
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
			return true;
		}
		return false;
	}
}