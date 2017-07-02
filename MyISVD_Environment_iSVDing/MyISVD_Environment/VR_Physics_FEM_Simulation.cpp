#include "VR_Physics_FEM_Simulation.h"
#include "CG/solvercontrol.h"
#include "CG/solvercg.h"
#include "CG/preconditionssor.h"
//#include <GL/freeglut.h>
#include <fstream>



#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>

#include <boost/math/tools/precision.hpp>

#if USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/tbb_allocator.h>
#include "tbb_parallel_set_skin_displacement.h"
#endif

#include "QuaternionTools.h"

namespace YC
{
	extern float g_scriptForceFactor;
	extern int cuda_bcMinCount;
	extern int cuda_bcMaxCount;

	

	VR_Physics_FEM_Simulation::VR_Physics_FEM_Simulation(void)
	{
		m_bSimulation = true;
	}


	VR_Physics_FEM_Simulation::~VR_Physics_FEM_Simulation(void)
	{
	}

	void VR_Physics_FEM_Simulation::loadOctreeNode_Global(const int XCount, const int YCount, const int ZCount)
	{
//#define XCount (40)
//#define YCount (4)
//#define ZCount (4)
//		CellRaidus;
		MyDenseVector coord(0,0,0);
		MyDenseVector xStep(1,0,0),yStep(0,1,0),zStep(0,0,1);
		std::vector< MyVector > elementData;
		MyVector cellInfo;cellInfo.resize(4);

		
		xMin = yMin = zMin = boost::math::tools::max_value<MyFloat>();
		xMax = yMax = zMax = boost::math::tools::min_value<MyFloat>();
		for (int i=0;i<LocalDomainCount;++i)
		{
			for (int x=0;x < XCount;++x)
			{
				for (int y=0;y<YCount;++y)
				{
					for (int z=0;z<ZCount;++z)
					{
						coord = MyDenseVector((i*XCount+x)*2+1,y*2+1,z*2+1);
						//coord = xStep * (x+1) + yStep*(y+1) + zStep * (z+1);
						MyDenseVector c = coord * CellRaidus;

						cellInfo[0] = c[0];
						cellInfo[1] = c[1];
						cellInfo[2] = c[2];
						cellInfo[3] = CellRaidus;
						//elementData.push_back(cellInfo);

						m_vec_cell.push_back(Cell::makeCell(MyPoint(cellInfo[0],cellInfo[1],cellInfo[2]),cellInfo[3]));
						m_vec_cell[m_vec_cell.size()-1]->computeCellType_Global();
					}
				}
			}
		}

		const MyInt nVtxSize = Vertex::getVertexSize();
		for (int v = 0; v < nVtxSize;++v)
		{
			const MyPoint& pos = Vertex::getVertex(v)->getPos();

			if (pos.x() > xMax)
			{
				xMax = pos.x();
			}
			if (pos.x() < xMin)
			{
				xMin = pos.x();
			}

			if (pos.y() > yMax)
			{
				yMax = pos.y();
			}
			if (pos.y() < yMin)
			{
				yMin = pos.y();
			}

			if (pos.z() > zMax)
			{
				zMax = pos.z();
			}
			if (pos.z() < zMin)
			{
				zMin = pos.z();
			}
		}
		printf("FEM (%d), EFG (%d), Couple(%d)\n",Cell::s_nFEM_Cell_Count,Cell::s_nEFG_Cell_Count,Cell::s_nCOUPLE_Cell_Count);
		printf("vertex count %d [%f , %f][%f , %f][%f , %f]\n",Vertex::getVertexSize(),xMin,xMax,yMin,yMax,zMin,zMax);
		//MyPause;
	}

#if USE_MAKE_CELL_SURFACE
	void VR_Physics_FEM_Simulation::creat_Outter_Skin(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs)
	{
		const MyInt nCellSize = m_vec_cell.size();
		const MyInt nAllFaceSize = nCellSize * Geometry::faces_per_cell;
		std::vector< MyInt > vecCellFaceFlag(nAllFaceSize + 1); memset(&vecCellFaceFlag[0], '\0', (nAllFaceSize + 1)*sizeof(MyInt));
		std::vector< MyInt > vecCellFaceIndex(nAllFaceSize + 1); memset(&vecCellFaceIndex[0], '\0', (nAllFaceSize + 1)*sizeof(MyInt));

		const MyInt nVtxSize = Vertex::getVertexSize();
		std::vector< MyInt > vecVtxFaceFlag(nVtxSize + 1); memset(&vecVtxFaceFlag[0], '\0', (nVtxSize + 1)*sizeof(MyInt));
		std::vector< MyInt > vecVtxFaceIndex(nVtxSize + 1); memset(&vecVtxFaceIndex[0], '\0', (nVtxSize + 1)*sizeof(MyInt));
		for (int i = 0; i < nCellSize; ++i)
		{
			CellPtr curCellPtr = m_vec_cell[i];
			const MyInt curCellFaceBase = i * Geometry::faces_per_cell;
			for (int f = 0; f < Geometry::faces_per_cell;++f)
			{
				vecCellFaceFlag[curCellFaceBase + f] = 0;
				MyDenseVector massCenter; massCenter.setZero();
				for (int v = 0; v < Geometry::vertexs_per_face;++v)
				{
					massCenter += curCellPtr->getVertex(Order::indexQuads[f][v])->getPos();
				}
				massCenter /= Geometry::vertexs_per_face;

				if (numbers::IsEqual(massCenter.x(), xMax) || 
					numbers::IsEqual(massCenter.x(), xMin) || 
					numbers::IsEqual(massCenter.y(), yMax) ||
					numbers::IsEqual(massCenter.y(), yMin) ||
					numbers::IsEqual(massCenter.z(), zMax) ||
					numbers::IsEqual(massCenter.z(), zMin) )
				{
					vecCellFaceFlag[curCellFaceBase + f] = 1;

					for (int v = 0; v < Geometry::vertexs_per_face; ++v)
					{
						int vid = curCellPtr->getVertex(Order::indexQuads[f][v])->getId();
						vecVtxFaceFlag[vid] = 1;
					}
					
				}
			}
		}

		thrust::exclusive_scan(&vecCellFaceFlag[0], &vecCellFaceFlag[0] + nAllFaceSize + 1, &vecCellFaceIndex[0]); //generate ghost cell count
		thrust::exclusive_scan(&vecVtxFaceFlag[0], &vecVtxFaceFlag[0] + nVtxSize + 1, &vecVtxFaceIndex[0]); //generate ghost cell count

		const int OutterTriFaceSize = vecCellFaceIndex[nAllFaceSize] * 2;
		const int OutterTriVtxSize = vecVtxFaceIndex[nVtxSize];

		matV.resize(OutterTriVtxSize, myDim); matV.setZero();
		matF.resize(OutterTriFaceSize, myDim); matF.setZero();
		matV_dofs.resize(OutterTriVtxSize, myDim); matV_dofs.setZero();

		for (int v = 0; v < nVtxSize;++v)
		{
			if (vecVtxFaceFlag[v])
			{
				VertexPtr curVtxPtr = Vertex::getVertex(v);
				matV.row(vecVtxFaceIndex[v]) = curVtxPtr->getPos();
				matV_dofs.row(vecVtxFaceIndex[v]) = curVtxPtr->getGlobalDofs();
			}
		}

		for (int i = 0; i < nCellSize; ++i)
		{
			CellPtr curCellPtr = m_vec_cell[i];
			const MyInt curCellFaceBase = i * Geometry::faces_per_cell;
			for (int f = 0; f < Geometry::faces_per_cell; ++f)
			{
				if (vecCellFaceFlag[curCellFaceBase + f])
				{
					int vid0 = curCellPtr->getVertex(Order::indexQuads_Tri[f][0][0])->getId();
					int vid1 = curCellPtr->getVertex(Order::indexQuads_Tri[f][0][1])->getId();
					int vid2 = curCellPtr->getVertex(Order::indexQuads_Tri[f][0][2])->getId();
					matF.row(vecCellFaceIndex[curCellFaceBase + f] * 2) = MyVectorI(vecVtxFaceIndex[vid0], vecVtxFaceIndex[vid1], vecVtxFaceIndex[vid2]);

					vid0 = curCellPtr->getVertex(Order::indexQuads_Tri[f][1][0])->getId();
					vid1 = curCellPtr->getVertex(Order::indexQuads_Tri[f][1][1])->getId();
					vid2 = curCellPtr->getVertex(Order::indexQuads_Tri[f][1][2])->getId();
					matF.row(vecCellFaceIndex[curCellFaceBase + f] * 2 + 1) = MyVectorI(vecVtxFaceIndex[vid0], vecVtxFaceIndex[vid1], vecVtxFaceIndex[vid2]);
				}
				//vecCellFaceFlag[curCellFaceBase + f] = 0;
			}
		}

		/*std::cout << matV << std::endl;
		std::cout << matF << std::endl; 
		std::cout << "vecCellFaceIndex[nAllFaceSize] " << vecCellFaceIndex[nAllFaceSize] << std::endl;
		MyPause;*/
	}

	void VR_Physics_FEM_Simulation::getSkinDisplacement(Physic_State& currentState, YC::MyMatrix& matV, YC::MyIntMatrix& matV_dofs, YC::MyMatrix& matU)
	{
		const MyInt nRows = matV_dofs.rows();
		const MyInt nCols = matV_dofs.cols();
		matU.resize(nRows, nCols);
		matU.setZero();

		
#if USE_TBB
		using namespace tbb;
		parallel_for(blocked_range<size_t>(0, nRows), ApplySkinDisplacement(&matV, &matU, &currentState.incremental_displacement, &matV_dofs), auto_partitioner());
#else
		for (int v = 0; v < nRows; v++)
		{
			const MyVectorI& dofs = matV_dofs.row(v);
			matU.row(v) = MyDenseVector(currentState.incremental_displacement[dofs.x()],
				currentState.incremental_displacement[dofs.y()],
				currentState.incremental_displacement[dofs.z()]
				);
		}
#endif
	}
#endif

	void VR_Physics_FEM_Simulation::distributeDof_global()
	{		
		int& m_nGlobalDof = m_global_State.m_nDof;
		int& m_nDC_Dofs = m_global_State.m_nDC_Dofs;
		m_nGlobalDof = Geometry::first_dof_idx;
		m_nDC_Dofs = Geometry::first_dof_idx;
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
			{
				VertexPtr curVtxPtr = curCellPtr->getVertex(v);
				if ( !(curVtxPtr->isValidGlobalDof()) && !(curVtxPtr->isBC()))
				{
					curVtxPtr->setGlobalDof(m_nGlobalDof,m_nGlobalDof+1,m_nGlobalDof+2);
					m_nGlobalDof+=3;
				}
			}
		}
		printf("global dof %d.\n",m_nGlobalDof);

		for (int i=0;i<m_global_State.m_vecDCBoundaryCondition.size();++i)
		{
			VertexPtr curVtxPtr = m_global_State.m_vecDCBoundaryCondition[i];
			if (!(curVtxPtr->isValidGlobalDof()) && (curVtxPtr->isBC()))
			{
				curVtxPtr->setGlobalDof(m_nGlobalDof,m_nGlobalDof+1,m_nGlobalDof+2);
				m_nGlobalDof+=3;
				m_nDC_Dofs += 3;
			}
			
		}

		printf("global dof %d, %d.\n",m_nGlobalDof,m_nDC_Dofs);
	}

	void VR_Physics_FEM_Simulation::createDCBoundaryCondition_Global()
	{
		std::map<MyInt,bool> mapNouse;
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
			{
				VertexPtr curVtxPtr = curCellPtr->getVertex(v);
				const MyPoint & pos = curVtxPtr->getPos();
				if (isDCCondition_Global(pos) )
				{
					mapNouse[curVtxPtr->getId()] = true;
				}
			}
		}

		std::map<MyInt,bool>::const_iterator ci = mapNouse.begin();
		std::map<MyInt,bool>::const_iterator endc = mapNouse.end();
		for (;ci != endc; ++ci)
		{
			VertexPtr curPtr = Vertex::getVertex(ci->first);
			curPtr->setIsBC(true);
			m_global_State.m_vecDCBoundaryCondition.push_back(curPtr);
		}

		printf("Boundary Condition %d.\n",m_global_State.m_vecDCBoundaryCondition.size());//MyPause;
	}

	bool VR_Physics_FEM_Simulation::isDCCondition_Global(const MyPoint& pos)
	{
		if (pos[0] < 3*CellRaidus /*|| pos[0] > (CellRaidus * 22*2) */)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	bool VR_Physics_FEM_Simulation::isForceCondition_Global(const MyPoint& pos,const int XCount, const int YCount, const int ZCount)
	{
		//MyError("isForceCondition_Global");
		const MyFloat forceXcoor = XCount * 2 * CellRaidus - CellRaidus;
		if (pos[0] > forceXcoor)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	void VR_Physics_FEM_Simulation::createForceBoundaryCondition_Global(const int XCount, const int YCount, const int ZCount)
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
				if (isForceCondition_Global(pos,XCount,YCount,ZCount) )
				{
					mapNouse[curVtxPtr->getId()] = true;

				}
			}
		}

		std::map<MyInt,bool>::const_iterator ci = mapNouse.begin();
		std::map<MyInt,bool>::const_iterator endc = mapNouse.end();
		for (;ci != endc; ++ci)
		{
			m_global_State.m_vecForceBoundaryCondition.push_back(Vertex::getVertex(ci->first));
		}

		printf("Force Boundary Condition %d.\n",m_global_State.m_vecForceBoundaryCondition.size());
		//MyPause;
	}

	void VR_Physics_FEM_Simulation::createGlobalMassAndStiffnessAndDampingMatrix_FEM_Global()
	{
		m_global_State.resize();

		std::map<long,std::map<long,Cell::TripletNode > >& StiffTripletNodeMap = Cell::m_global_state.m_TripletNode_Stiffness;
		std::map<long,std::map<long,Cell::TripletNode > >& MassTripletNodeMap = Cell::m_global_state.m_TripletNode_Mass;
		std::map<long,Cell::TripletNode >& RhsTripletNode = Cell::m_global_state.m_TripletNode_Rhs;
		StiffTripletNodeMap.clear();
		MassTripletNodeMap.clear();
		RhsTripletNode.clear();

#if USE_MODAL_WARP
		std::map<long,std::map<long,Cell::TripletNode > >& ModalWarpMap = Cell::m_global_state.m_TripletNode_ModalWarp;
		ModalWarpMap.clear();
#endif

		std::vector< CellPtr > & curCellVec = m_vec_cell;
		for (unsigned i=0;i<curCellVec.size();++i)
		{
			CellPtr curCellPtr = curCellVec[i];

			curCellPtr->initialize_Global();
			curCellPtr->assembleSystemMatrix_Global();
		}

		Physic_State& curState = m_global_State;
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
			curState.m_global_StiffnessMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
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
			curState.m_global_MassMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
			MassTripletNodeMap.clear();
			vec_triplet.clear();
		}


		//assemble global rhs vector
		std::map<long,Cell::TripletNode >::const_iterator itrRhs = RhsTripletNode.begin();
		curState.m_computeRhs.setZero();
		for (;itrRhs != RhsTripletNode.end();++itrRhs)
		{
			curState.m_computeRhs[itrRhs->first] = (itrRhs->second).val;
		}
		RhsTripletNode.clear();

		std::vector< VertexPtr >& refVec = curState.m_vecForceBoundaryCondition;
		MyVector& refRhs = curState.R_rhs_externalForce;
		for (MyInt v=0;v<refVec.size();++v)
		{
			MyInt curDof = refVec[v]->getGlobalDofs().y();
			refRhs[curDof] = g_scriptForceFactor * Material::GravityFactor*(-1.f);

		}

		curState.m_global_DampingMatrix = Material::damping_alpha * curState.m_global_MassMatrix + Material::damping_beta * curState.m_global_StiffnessMatrix;

#if 0
		itr_tri =  ModalWarpMap.begin();
		for (;itr_tri != ModalWarpMap.end();++itr_tri)
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
			curState.m_global_ModalWarp.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
			ModalWarpMap.clear();
			vec_triplet.clear();
		}

		//printfMTX("d:\\modalWarp.mtx",curState.m_global_ModalWarp);
#endif
		
	}

	void VR_Physics_FEM_Simulation::createNewMarkMatrix_Global()
	{
		
		m_global_State.m_computeMatrix = m_global_State.m_global_StiffnessMatrix;
		m_global_State.m_computeMatrix += m_db_NewMarkConstant[0] * m_global_State.m_global_MassMatrix;
		m_global_State.m_computeMatrix += m_db_NewMarkConstant[1] * m_global_State.m_global_DampingMatrix;

		//printfMTX("d:\\massMatrix.mtx",m_global_State.m_global_MassMatrix);
		//printfMTX("d:\\stiffMatrix.mtx",m_global_State.m_global_StiffnessMatrix);

		m_global_State.incremental_displacement.setZero();
		m_global_State.velocity.setZero();
		m_global_State.acceleration.setZero();
		m_global_State.displacement.setZero();
		m_global_State.old_acceleration.setZero();
		m_global_State.old_displacement.setZero();
		return ;
	}

	void VR_Physics_FEM_Simulation::update_rhs_Global(const int nStep)
	{
		Physic_State & curState = m_global_State;
		curState.R_rhs.setZero();
		curState.mass_rhs.setZero();
		curState.damping_rhs.setZero();

		//incremental_displacement,velocity,acceleration
		curState.mass_rhs += m_db_NewMarkConstant[0] * (curState.displacement);
		curState.mass_rhs += m_db_NewMarkConstant[2] * curState.velocity;
		curState.mass_rhs += m_db_NewMarkConstant[3] * curState.acceleration;

		curState.damping_rhs += m_db_NewMarkConstant[1] * (curState.displacement);
		curState.damping_rhs += m_db_NewMarkConstant[4] * curState.velocity;
		curState.damping_rhs += m_db_NewMarkConstant[5] * curState.acceleration;

		curState.R_rhs += curState.m_computeRhs;

		curState.R_rhs += curState.m_global_MassMatrix * curState.mass_rhs;
		curState.R_rhs += curState.m_global_DampingMatrix * curState.damping_rhs;


		if (nStep >= cuda_bcMinCount && nStep < cuda_bcMaxCount)
		{
			curState.R_rhs += curState.R_rhs_externalForce;
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

	void VR_Physics_FEM_Simulation::apply_boundary_values_Global()
	{
		Physic_State & curState = m_global_State;
		std::vector< VertexPtr >& vecBoundaryVtx = curState.m_vecDCBoundaryCondition;
		MySpMat&  computeMatrix = curState.m_computeMatrix;
		MyVector curRhs = curState.R_rhs;
		MyVector curDisplacement = curState.incremental_displacement;
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
			const MyVectorI& Dofs = curVtxPtr->getGlobalDofs();
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

	void VR_Physics_FEM_Simulation::solve_linear_problem_Global()
	{
		using namespace YC;
		Physic_State & curState = m_global_State;
		SolverControl           solver_control (curState.m_nDof,  1e-3*numbers::l2_norm(curState.R_rhs));
		SolverCG              cg (solver_control);

		PreconditionSSOR preconditioner;
		preconditioner.initialize(curState.m_computeMatrix, 1.2);

		cg.solve (curState.m_computeMatrix, curState.incremental_displacement, curState.R_rhs,	preconditioner);
		//std::cout << incremental_displacement.transpose() << std::endl;//MyPause;
	}

	void VR_Physics_FEM_Simulation::solve_linear_problem_Global_Inverse()
	{
		using namespace YC;
		Physic_State & curState = m_global_State;
		static bool bFirst = true;
		static Eigen::ColPivHouseholderQR< MyMatrix > ALM_QR;
		if (bFirst)
		{
			bFirst = false;
			
			curState.m_computeMatrix.makeCompressed();
			MyMatrix tmp(curState.m_computeMatrix.rows(),curState.m_computeMatrix.cols());
			tmp.setZero();

			for (int k=0; k<curState.m_computeMatrix.outerSize(); ++k)
			{
				for (MySpMat::InnerIterator it(curState.m_computeMatrix,k); it; ++it)
				{
					tmp.coeffRef(it.row(),it.col()) = it.value();
					
				}
			}
			ALM_QR = tmp.colPivHouseholderQr();
		}

		curState.incremental_displacement = ALM_QR.solve(curState.R_rhs);
	}

	void VR_Physics_FEM_Simulation::update_u_v_a_Global()
	{
		Physic_State & curState = m_global_State;
		const MyVector& solu = curState.incremental_displacement;
		MyVector& disp_vec = curState.displacement;
		MyVector& vel_vec = curState.velocity;
		MyVector& acc_vec = curState.acceleration;
		MyVector& old_acc = curState.old_acceleration;
		MyVector& old_solu = curState.old_displacement;

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

	void VR_Physics_FEM_Simulation::simulationOnCPU_Global(const int nTimeStep)
	{
		update_rhs_Global(nTimeStep);
		apply_boundary_values_Global();
		solve_linear_problem_Global();
		update_u_v_a_Global();
	}

#if USE_MODAL_WARP
	void VR_Physics_FEM_Simulation::simulationOnCPU_Global_WithModalWrap(const int nTimeStep)
	{
		if (m_bSimulation)
		{
			update_rhs_Global(nTimeStep);
			apply_boundary_values_Global();
			solve_linear_problem_Global();
			update_u_v_a_Global();

			//return;
			compute_Local_R_Global(m_global_State);			
			update_displacement_ModalWrap(m_global_State);
		}
	}

	void VR_Physics_FEM_Simulation::update_displacement_ModalWrap(Physic_State& globalState)
	{
		globalState.incremental_displacement = globalState.m_R_hat * globalState.incremental_displacement;
	}

	void VR_Physics_FEM_Simulation::compute_Local_R_Global(Physic_State& globalState)
	{
		compute_ModalWrap_Rotation(globalState.incremental_displacement, globalState.m_R_hat);
	}

	void VR_Physics_FEM_Simulation::compute_ModalWrap_Rotation(const MyVector& globalDisplacement, MySpMat& modalWrap_R_hat)
	{
		
		MyVector w,translation;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		for (int c=0;c<m_vec_cell.size();++c)
		{
			m_vec_cell[c]->TestModalWrapMatrix(globalDisplacement,w,translation,cellDisplace);
			m_vec_cell[c]->m_vec_LocalQuat.resize(Geometry::vertexs_per_cell);
			MyDenseVector ww;ww.setZero();
			MyDenseVector W,W_hat;
			float w_norm;
			for (int m=0;m<8;++m)
			{
				W = MyDenseVector(w[3*m+0],w[3*m+1],w[3*m+2]);
				W_hat = W;
				W_hat.normalize();
				w_norm = W.norm();

				if (numbers::isZero(w_norm))
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion( );
				}
				else
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion( Vec3<float>(W_hat[0],W_hat[1],W_hat[2]),w_norm);
				}

			}
		}


		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
		modalWrap_R_hat.resize(globalDisplacement.size(),globalDisplacement.size());modalWrap_R_hat.setZero();
		MyMatrix_3X3 tmpR;
		//m_vec_VtxLocalRotaM.clear();
		for (int v=0;v<Vertex::getVertexSize();++v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec;cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getGlobalDofs();

			const int nSize = refVec.size();
			const int addAmount = nSize * Geometry::vertexs_per_cell;

			Axis::Quaternion firstQuat = refVec[0]->m_vec_LocalQuat[0];
			for (int c=0;c<refVec.size();++c)
			{
				for (int i=0;i<Geometry::vertexs_per_cell;++i)
				{
					Axis::Quaternion newRotaQuat = refVec[c]->m_vec_LocalQuat[i];
					curQuat = AverageQuaternion(cumulateVec,newRotaQuat,firstQuat,addAmount);		
				}								
			}

			Vec3<Axis::MySReal> WW = curQuat.toEulerVector();

			MyDenseVector W,W_hat;
			float w_norm;
			W = MyDenseVector(WW.x(),WW.y(),WW.z());
			W_hat = W;
			W_hat.normalize();
			w_norm = W.norm();
			MyMatrix_3X3 X;X.setZero();

			X.coeffRef(0,1) = -1.f*W_hat[2];
			X.coeffRef(0,2) = W_hat[1];

			X.coeffRef(1,0) = W_hat[2];
			X.coeffRef(1,2) = -1.f*W_hat[0];

			X.coeffRef(2,0) = -1.f * W_hat[1];
			X.coeffRef(2,1) = W_hat[0];

			if (numbers::isZero(w_norm))
			{
				//std::cout << "numbers::isZero(W_norm)" << std::endl; 
				tmpR = MyDenseMatrix::Identity(3,3);
			}
			else
			{
				tmpR = MyDenseMatrix::Identity(3,3) + (X * (1-cos(w_norm)) / w_norm ) + X * X * (1-sin(w_norm) / w_norm);				
			}
			//m_vec_VtxLocalRotaM.push_back(tmpR);
			//curQuat.toMatrix(tmpR);
			for (int r=0;r<3;++r)
			{
				for (int c=0;c<3;++c)
				{
					vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(dofs[r],dofs[c],tmpR.coeff(r,c) ));
				}
			}	
		}

		modalWrap_R_hat.setFromTriplets(vec_triplet.begin(),vec_triplet.end());;
	}

#endif

	void VR_Physics_FEM_Simulation::render_Global()
	{
		::glPushMatrix();
		glScalef(8,8,8);
		glTranslatef(0,2,0);
		glPointSize(5.f);
		glBegin(GL_POINTS);
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			const MyInt id = Colors::red;
			glColor3f(Colors::colorTemplage[Colors::red][0],Colors::colorTemplage[Colors::red][1],Colors::colorTemplage[Colors::red][2]);

			{
				CellPtr curCellPtr = m_vec_cell[c];
				for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
				{
					VertexPtr vtxPtr = curCellPtr->getVertex(v);
					MyVectorI& dofs = vtxPtr->getGlobalDofs();
					MyPoint& pos = vtxPtr->getPos();
					glVertex3f(pos[0] + m_global_State.incremental_displacement[dofs[0]],pos[1] + m_global_State.incremental_displacement[dofs[1]],pos[2] + m_global_State.incremental_displacement[dofs[2]]);
					//glVertex3f(pos[0] ,pos[1] ,pos[2]);
				}
			}
		}
		glEnd();

		glPopMatrix();
	}

	void VR_Physics_FEM_Simulation::printfMTX(const char* lpszFileName, const MySpMat& sparseMat)
	{
		std::ofstream outfile(lpszFileName);
		const int nDofs = sparseMat.rows();
		const int nValCount = sparseMat.nonZeros();
		int nValCount_0 = 0;

		outfile << nDofs << " " << nDofs << " " << nValCount << std::endl;
		for (int k=0; k<sparseMat.outerSize(); ++k)
		{
			for (MySpMat::InnerIterator it(sparseMat,k); it; ++it)
			{
				if (   !numbers::isEqual( it.value() , sparseMat.coeff(it.col(),it.row()),0.00000001 )   )
				{
					//printf("(%d,%d) (%f)----(%f)\n",it.row(),it.col(),it.value(),obj.coeff(it.col(),it.row()));
					++nValCount_0;
					outfile << it.row() << " " << it.col() << " " << it.value() << std::endl;
				}
			}
		}
		Q_ASSERT(nValCount_0 == nValCount);
		outfile.close();
	}

	void VR_Physics_FEM_Simulation::TestVtxCellId()
	{
		for (int v=0;v<Vertex::getVertexSize();++v)
		{
			if ( v != (Vertex::getVertex(v)->getId()) )
			{
				printf("index %d; Id %d \n",v,(Vertex::getVertex(v)->getId()));
			}
		}

		for (int c=0;c<m_vec_cell.size();++c)
		{
			if (c != m_vec_cell[c]->getID())
			{
				printf("index %d; cell Id %d \n",c,(m_vec_cell[c]->getID()));
			}
		}

		for (int v=0;v<Vertex::getVertexSize();++v)
		{
			VertexPtr vtxPtr = Vertex::getVertex(v);

			if ( (vtxPtr->getShareCellCount()) != (vtxPtr->m_vec_ShareCell.size()) )
			{
				printf("getShareCellCount %d; m_vec_ShareCell.size() %d \n",vtxPtr->getShareCellCount(),(vtxPtr->m_vec_ShareCell.size()));
			}
		}
		//MyPause;
	}

#if USE_MODAL_WARP
	void VR_Physics_FEM_Simulation::solve_linear_Subspace_problem(Vibrate_State& curState)
	{
		//R_MR_rhs = m_BasisU.transpose() * R_rhs;
		curState.MR_incremental_displacement = curState.m_MR_computeMatrix_Inverse * curState.R_MR_rhs;
		//std::cout << curState.MR_incremental_displacement << std::endl;MyPause;
		/*SolverControl           solver_control (nSubspaceDofs(),  1e-3*numbers::l2_norm(R_MR_rhs));
		SolverCG              cg (solver_control);

		PreconditionSSOR preconditioner;
		preconditioner.initialize(m_MR_computeMatrix, 1.2);

		cg.solve (m_MR_computeMatrix, incremental_displacement, R_rhs,	preconditioner);*/
	}

	void VR_Physics_FEM_Simulation::computeReferenceDisplace_ModalWarp(Physic_State& curRefState, Vibrate_State&  curState)
	{
		curRefState.incremental_displacement = curState.m_BasisU_hat * curState.MR_incremental_displacement;
	}

	Axis::Quaternion VR_Physics_FEM_Simulation::covert_matrix2Quaternion(const MyDenseMatrix& nodeRotationMatrix)
	{
		Axis::Quaternion quater;
		Matrix3<float> mat;
		mat.matrix[0].elems[0] = nodeRotationMatrix(0,0);
		mat.matrix[0].elems[1] = nodeRotationMatrix(0,1);
		mat.matrix[0].elems[2] = nodeRotationMatrix(0,2);

		mat.matrix[1].elems[0] = nodeRotationMatrix(1,0);
		mat.matrix[1].elems[1] = nodeRotationMatrix(1,1);
		mat.matrix[1].elems[2] = nodeRotationMatrix(1,2);

		mat.matrix[2].elems[0] = nodeRotationMatrix(2,0);
		mat.matrix[2].elems[1] = nodeRotationMatrix(2,1);
		mat.matrix[2].elems[2] = nodeRotationMatrix(2,2);
		quater.fromMatrix(mat);
		return quater;
	}

	void VR_Physics_FEM_Simulation::compute_modalWarp_R(Vibrate_State& subspaceState)
	{
		MyVector W_global = subspaceState.m_ModalWarp_Basis * subspaceState.MR_incremental_displacement;
		//W_global *= -1.f;
		const int nVtxSize = subspaceState.m_globalDofs / 3;
		MyDenseMatrix X(3,3);X.setZero();
		MyDenseMatrix tmpR;
		MyFloat W_norm;
		subspaceState.m_R_hat.setZero();
		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
		m_vec_frame_quater.clear();
		for (int m=0;m<nVtxSize;++m)
		{
			MyVectorI dofs = Vertex::getVertex(m)->getGlobalDofs();

			MyDenseVector W=MyDenseVector( W_global[dofs[0]], W_global[dofs[1]], W_global[dofs[2]]);
			MyDenseVector W_hat = W;
			W_hat.normalize();

			X.setZero();
			X.coeffRef(0,1) = -1.f*W_hat[2];
			X.coeffRef(0,2) = W_hat[1];

			X.coeffRef(1,0) = W_hat[2];
			X.coeffRef(1,2) = -1.f*W_hat[0];

			X.coeffRef(2,0) = -1.f * W_hat[1];
			X.coeffRef(2,1) = W_hat[0];

			W_norm = W.norm();

			if (numbers::isZero(W_norm))
			{
				tmpR = MyDenseMatrix::Identity(3,3);
			}
			else
			{
				tmpR = MyDenseMatrix::Identity(3,3) + (X * (1-cos(W_norm)) / W_norm ) + X * X * (1-sin(W_norm) / W_norm);				
			}

			m_vec_frame_quater.push_back(covert_matrix2Quaternion(tmpR));

			for (int r=0;r<3;++r)
			{
				for (int c=0;c<3;++c)
				{
					vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(dofs[r],dofs[c],tmpR.coeff(r,c) ));
					//vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(i*3+r,i*3+c,tmpR.coeff(r,c) )); std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
				}
			}	
		}

		subspaceState.m_R_hat.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
	}

	void VR_Physics_FEM_Simulation::compute_R_Basis(Vibrate_State& subspaceState)
	{
		subspaceState.m_BasisU_hat = subspaceState.m_R_hat * subspaceState.m_BasisU;
	}

	void VR_Physics_FEM_Simulation::compute_rhs_ModalWrap(const Physic_State& globalState, Vibrate_State& subspaceState)
	{
		subspaceState.R_MR_rhs = subspaceState.m_BasisU.transpose() * globalState.R_rhs;
	}

	void VR_Physics_FEM_Simulation::compute_Local_R(Vibrate_State& subspaceState)
	{
		m_global_State.incremental_displacement = subspaceState.m_BasisU * subspaceState.MR_incremental_displacement;

		//std::cout << m_global_State.incremental_displacement.transpose() << std::endl;MyPause;

		MyVector w,translation;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		for (int c=0;c<m_vec_cell.size();++c)
		{
			m_vec_cell[c]->TestModalWrapMatrix(m_global_State.incremental_displacement,w,translation,cellDisplace);
			m_vec_cell[c]->m_vec_LocalQuat.resize(Geometry::vertexs_per_cell);
			MyDenseVector ww;ww.setZero();
			MyDenseVector W,W_hat;
			float w_norm;
			for (int m=0;m<8;++m)
			{
				W = MyDenseVector(w[3*m+0],w[3*m+1],w[3*m+2]);
				W_hat = W;
				W_hat.normalize();
				w_norm = W.norm();

				if (numbers::isZero(w_norm))
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion( );
				}
				else
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion( Vec3<float>(W_hat[0],W_hat[1],W_hat[2]),w_norm);
				}

			}
		}


		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;subspaceState.m_R_hat.setZero();
		MyMatrix_3X3 tmpR;
		m_vec_VtxLocalRotaM.clear();
		for (int v=0;v<Vertex::getVertexSize();++v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec;cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getGlobalDofs();

			const int nSize = refVec.size();
			const int addAmount = nSize * Geometry::vertexs_per_cell;

			Axis::Quaternion firstQuat = refVec[0]->m_vec_LocalQuat[0];
			for (int c=0;c<refVec.size();++c)
			{
				for (int i=0;i<Geometry::vertexs_per_cell;++i)
				{
					Axis::Quaternion newRotaQuat = refVec[c]->m_vec_LocalQuat[i];
					curQuat = AverageQuaternion(cumulateVec,newRotaQuat,firstQuat,addAmount);		
				}								
			}

			Vec3<Axis::MySReal> WW = curQuat.toEulerVector();

			MyDenseVector W,W_hat;
			float w_norm;
			W = MyDenseVector(WW.x(),WW.y(),WW.z());
			W_hat = W;
			W_hat.normalize();
			w_norm = W.norm();
			MyMatrix_3X3 X;X.setZero();

			X.coeffRef(0,1) = -1.f*W_hat[2];
			X.coeffRef(0,2) = W_hat[1];

			X.coeffRef(1,0) = W_hat[2];
			X.coeffRef(1,2) = -1.f*W_hat[0];

			X.coeffRef(2,0) = -1.f * W_hat[1];
			X.coeffRef(2,1) = W_hat[0];

			if (numbers::isZero(w_norm))
			{
				//std::cout << "numbers::isZero(W_norm)" << std::endl; 
				tmpR = MyDenseMatrix::Identity(3,3);
			}
			else
			{
				tmpR = MyDenseMatrix::Identity(3,3) + (X * (1-cos(w_norm)) / w_norm ) + X * X * (1-sin(w_norm) / w_norm);				
			}
			m_vec_VtxLocalRotaM.push_back(tmpR);
			//curQuat.toMatrix(tmpR);
			for (int r=0;r<3;++r)
			{
				for (int c=0;c<3;++c)
				{
					vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(dofs[r],dofs[c],tmpR.coeff(r,c) ));


					//vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(i*3+r,i*3+c,tmpR.coeff(r,c) )); 
				}
			}	
		}
		//subspaceState.m_R_hat.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		subspaceState.m_R_hat.setFromTriplets(vec_triplet.begin(), vec_triplet.end());;

		
		subspaceState.m_BasisU_hat = subspaceState.m_R_hat * subspaceState.m_BasisU;
	}

	void VR_Physics_FEM_Simulation::compute_CellRotation(Vibrate_State& subspaceState)
	{
		m_global_State.incremental_displacement = subspaceState.m_BasisU * subspaceState.MR_incremental_displacement;
		m_testCellRotation.clear();

		MyMatrix_3X3 RMatrix;
		MyDenseVector translationVec;
		MyVector translation;
		MyVector w;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		MyVector cellTranslation;
		for (int c=0;c<m_vec_cell.size();++c)
		{
			m_vec_cell[c]->TestCellRotationMatrix(m_global_State.incremental_displacement,RMatrix,translationVec);
			m_testCellRotation.push_back(std::make_pair(translationVec,covert_matrix2Quaternion(RMatrix)));
			//continue;
		}

		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
		vecShareCellQuaternion.clear();
		m_vec_CelLocalRotaM.clear();
		MyMatrix_3X3 tmpR;
		for (int v=0;v<Vertex::getVertexSize();++v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec;cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getGlobalDofs();

			const int nSize = refVec.size();
			if (1 == nSize)
			{
				int id = refVec[0]->getID();
				float * quatPtr = m_testCellRotation[id].second.ptr();
				curQuat = NormalizeQuaternion(quatPtr[0],quatPtr[1],quatPtr[2],quatPtr[3]);
			}
			else
			{
				int id = refVec[0]->getID();
				Axis::Quaternion firstQuat = m_testCellRotation[id].second;
				for (int c=0;c<refVec.size();++c)
				{
					int id = refVec[c]->getID();
					Axis::Quaternion newRotaQuat = m_testCellRotation[id].second;
					curQuat = AverageQuaternion(cumulateVec,newRotaQuat,firstQuat,nSize);						
				}
			}


			//curQuat = InverseSignQuaternion(curQuat);
			vecShareCellQuaternion.push_back(curQuat);

			curQuat.toMatrix(tmpR);
			m_vec_CelLocalRotaM.push_back(tmpR);
			for (int r=0;r<3;++r)
			{
				for (int c=0;c<3;++c)
				{
					vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(dofs[r],dofs[c],tmpR.coeff(r,c) ));
					//vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(i*3+r,i*3+c,tmpR.coeff(r,c) )); 
				}
			}	
		}
		//Q_ASSERT(vecShareCellQuaternion.size() == m_vec_frame_quater.size());
		/*for (int m=0;m<vecShareCellQuaternion.size();++m)
		{
		MyMatrix_3X3 shareCellRot,vtxRot;
		vecShareCellQuaternion[m].toMatrix(shareCellRot);
		m_vec_frame_quater[m].toMatrix(vtxRot);
		std::cout << "ShareRot " << std::endl << shareCellRot << std::endl << "Vtx Rot " << std::endl << vtxRot << std::endl;
		MyPause;
		}*/


		subspaceState.m_R_hat.setFromTriplets(vec_triplet.begin(), vec_triplet.end());;
		subspaceState.m_BasisU_hat = subspaceState.m_R_hat * subspaceState.m_BasisU;
		//m_global_State.incremental_displacement = m_global_subspace_State.m_BasisU_hat * m_global_subspace_State.MR_incremental_displacement;
	}

	void VR_Physics_FEM_Simulation::printRotationFrame()
	{
		return ;
		for (int i=0;i<Vertex::getVertexSize()/4;++i)		
		{
			//int i = Vertex::getVertexSize()-232;
			VertexPtr curVtxPtr = Vertex::getVertex(i);
			MyVectorI dofs = curVtxPtr->getGlobalDofs();
			MyPoint pos = curVtxPtr->getPos();
			MyPoint translationPos = MyPoint( pos[0] + m_global_State.incremental_displacement[dofs[0]],
				pos[1]+ m_global_State.incremental_displacement[dofs[1]],
				pos[2]+ m_global_State.incremental_displacement[dofs[2]] );

			Axis::draw(Vec3<float>(translationPos[0],translationPos[1],translationPos[2]), covert_matrix2Quaternion(m_vec_VtxLocalRotaM[i]), .1);

			/*std::vector< CellPtr >& refShareCell = curVtxPtr->m_vec_ShareCell;
			std::cout << "Share Cell Count : " <<refShareCell.size() << std::endl;
			for (int c=0;c<refShareCell.size();++c)
			{
			const int cIdx = refShareCell[c]->getID();
			MyDenseVector& translationPos = m_testCellRotation[cIdx].first;
			Axis::draw(Vec3<float>(translationPos[0],translationPos[1],translationPos[2]), m_testCellRotation[cIdx].second, 0.1f);
			}*/
		}
		return ;
		for (int i=m_testCellRotation.size()-1;i<m_testCellRotation.size();++i)
		{
			MyDenseVector& translationPos = m_testCellRotation[i].first;

			Axis::draw(Vec3<float>(translationPos[0],translationPos[1],translationPos[2]), m_testCellRotation[i].second, 0.25);

			for (int v=0;v<8;++v)
			{
				VertexPtr curVtxPtr = m_vec_cell[i]->getVertex(v);
				MyVectorI dofs = curVtxPtr->getGlobalDofs();
				MyPoint pos = curVtxPtr->getPos();

				MyPoint translationPos = MyPoint( pos[0] + m_global_State.incremental_displacement[dofs[0]],
					pos[1]+ m_global_State.incremental_displacement[dofs[1]],
					pos[2]+ m_global_State.incremental_displacement[dofs[2]] );

				Axis::draw(Vec3<float>(translationPos[0],translationPos[1],translationPos[2]), vecShareCellQuaternion[curVtxPtr->getId()], .1);
			}
		}

		return ;
		const int nVtxSize = Vertex::getVertexSize();
		for (int i=nVtxSize-10;i<nVtxSize;++i)
			//int i=800;
		{

			VertexPtr curVtxPtr = Vertex::getVertex(i);
			MyVectorI dofs = curVtxPtr->getGlobalDofs();
			MyPoint pos = curVtxPtr->getPos();
			MyPoint translationPos = MyPoint( pos[0] + m_global_State.incremental_displacement[dofs[0]],
				pos[1]+ m_global_State.incremental_displacement[dofs[1]],
				pos[2]+ m_global_State.incremental_displacement[dofs[2]] );
			int idx = dofs[0] / 3;

			//Axis::draw(Vec3<float>(translationPos[0],translationPos[1],translationPos[2]), m_vec_frame_quater[idx], .1);
			Axis::draw(Vec3<float>(translationPos[0],translationPos[1],translationPos[2]), vecShareCellQuaternion[i], .1);
		}
		//float * ptr = m_vec_frame_quater[i].ptr();
		//printf("quater[%f,%f,%f,%f]\n",ptr[0],ptr[1],ptr[2],ptr[3]);
	}
#endif//USE_MODAL_WARP

#if USE_SUBSPACE_Whole_Spectrum
	void VR_Physics_FEM_Simulation::create_Whole_Spectrum(const Physic_State& globalState, Vibrate_State& subspaceState)
	{
		using namespace std;
		const MyInt nNativeDofs = globalState.m_nDof;
		const MyInt nDCDofs = globalState.m_nDC_Dofs;
		const MyInt nSubspaceModeNum = nNativeDofs - nDCDofs;
		const int nDofs = nSubspaceModeNum;//K.rows()-nBCSize;

		const MySpMat& matK = globalState.m_global_StiffnessMatrix;
		const MySpMat& matM = globalState.m_global_MassMatrix;

		subspaceState.m_nSubspaceDofs = nSubspaceModeNum;
		const int nEigenValueCount = subspaceState.m_nSubspaceDofs;
		
		
		MyMatrix tmpK(nDofs, nDofs);
		tmpK = matK.block(0, 0, nDofs, nDofs);

		MyMatrix tmpM(nDofs, nDofs);
		tmpM = matM.block(0, 0, nDofs, nDofs);
		
		cout << "GeneralizedSelfAdjointEigenSolver start ..." << endl;
		Eigen::GeneralizedSelfAdjointEigenSolver< MyMatrix > es(tmpK, tmpM);
		cout << "GeneralizedSelfAdjointEigenSolver complete" << endl;
		//LinearModeAnalysis_Global(nEigenValueCount, _eigenValueList, _eigenVector);

		subspaceState.m_EigenValuesVector.resize(nSubspaceModeNum);
		subspaceState.m_EigenValuesVector = es.eigenvalues();

		subspaceState.m_BasisU.resize(nNativeDofs, nSubspaceModeNum);
		subspaceState.m_BasisU.setZero();
		subspaceState.m_BasisU.block(0, 0, nSubspaceModeNum, nSubspaceModeNum) = es.eigenvectors();

		cout << "createSubspaceMatrix start ..." << endl;
		subspaceState.createSubspaceMatrix(m_global_State, m_db_NewMarkConstant[0], m_db_NewMarkConstant[1]);
		cout << "createSubspaceMatrix complete" << endl;
	}

	void VR_Physics_FEM_Simulation::simulationWholeSpectrumOnCPU(const int nTimeStep)
	{
#if USE_SUBSPACE_Whole_Spectrum
		if (m_bSimulation)
		{
			//m_bSimulation = false;
			update_rhs_Global(nTimeStep);
			compute_rhs_ModalWrap(m_global_State, m_global_Whole_Spectrum_State);
			solve_linear_Subspace_problem(m_global_Whole_Spectrum_State);

			//compute_modalWarp_R(m_global_subspace_State);
			//compute_R_Basis(m_global_subspace_State);
			compute_Local_R(m_global_Whole_Spectrum_State);
			//compute_CellRotation();

			/*for (int i=500;i<Vertex::getVertexSize()/2;++i)
			{
			std::cout << "Vtx Mat" << std::endl << m_vec_VtxLocalRotaM[i] << std::endl << "Cell Mat "<< std::endl << m_vec_CelLocalRotaM[i] << std::endl;
			MyPause;
			}*/
			//
			update_u_v_a_Global();
			computeReferenceDisplace_ModalWarp(m_global_State, m_global_Whole_Spectrum_State);
		}
#else
		update_Subspace_rhs(nTimeStep);
		//update_rhs(nTimeStep);
		solve_linear_Subspace_problem();
		computeReferenceDisplace();
		update_Subspace_u_v_a();
#endif
	}

#endif//USE_SUBSPACE_Whole_Spectrum

#if USE_SUBSPACE_SVD

#if SingleDomainISVD
	void VR_Physics_FEM_Simulation::init_iSVD_updater(const Physic_State& globalState, MyInt nPoolCapacity)
	{
	}
#else
	void VR_Physics_FEM_Simulation::initSVD(const Physic_State& globalState, MyInt nPoolCapacity)
	{
		m_usingSubspaceSimulation = false;
		m_global_SVD_State.m_globalDofs = globalState.m_nDof;
		m_global_SVD_State.m_nDC_Dofs4SVD = globalState.m_nDC_Dofs;
		m_global_SVD_State.m_matDisplacePoolCapacity = nPoolCapacity;
		m_global_SVD_State.m_matDisplacePoolSize = 0;
		m_global_SVD_State.m_matDisplacePool.resize(globalState.m_nDof, nPoolCapacity);
	}

	void VR_Physics_FEM_Simulation::appendGlobalDisplace2Pool(const MyVector& global_displacement)
	{
		using namespace std;
		Vibrate_State& svdState = m_global_SVD_State;
		const MyInt validRows = svdState.m_globalDofs - svdState.m_nDC_Dofs4SVD;
		Q_ASSERT(svdState.m_matDisplacePool.rows() == global_displacement.rows());
		svdState.m_matDisplacePool.block(0, svdState.m_matDisplacePoolSize, validRows, 1) = global_displacement.block(0, 0, validRows, 1);
		++svdState.m_matDisplacePoolSize;
		if (svdState.m_matDisplacePoolSize >= svdState.m_matDisplacePoolCapacity)
		{
			m_usingSubspaceSimulation = true;

			Eigen::JacobiSVD< MyMatrix > svds(svdState.m_matDisplacePool, Eigen::ComputeThinU);
			svdState.m_BasisU = svds.matrixU();

			//cout << "svdState.m_nSubspaceDofs=" << svdState.m_nSubspaceDofs << "  svdState.m_BasisU.cols()=" << svdState.m_BasisU.cols() << endl; MyPause;
			svdState.m_nSubspaceDofs = svdState.m_BasisU.cols();

			cout << "createSubspaceMatrix start ..." << endl;
			svdState.createSubspaceMatrix(m_global_State, m_db_NewMarkConstant[0], m_db_NewMarkConstant[1]);
			cout << "createSubspaceMatrix complete" << endl;
		}
	}

	void VR_Physics_FEM_Simulation::simulationSVDOnCPU(const int nTimeStep)
	{
		Vibrate_State& svdState = m_global_SVD_State;
		static MyInt sSubspaceDoneCount = 0;
		static MyInt sSubspaceDoneSize = nSampleInterval * svdState.m_matDisplacePoolCapacity;

		if (!m_usingSubspaceSimulation)
		{
			simulationOnCPU_Global_WithModalWrap(nTimeStep);
			if ((nSampleInterval - 1) == (nTimeStep%nSampleInterval))
			{
				appendGlobalDisplace2Pool(m_global_State.incremental_displacement);
			}
			sSubspaceDoneCount = 0;
		}
		else
		{

			simulationSVDOnCPU_Subspace_WithModalWrap(nTimeStep);
			++sSubspaceDoneCount;
			if (sSubspaceDoneCount >= sSubspaceDoneSize)
			{
				m_usingSubspaceSimulation = false;
				svdState.m_matDisplacePoolSize = 0;
				svdState.m_matDisplacePool.setZero();
			}

		}
	}

	void VR_Physics_FEM_Simulation::simulationSVDOnCPU_Subspace_WithModalWrap(const int nTimeStep)
	{
		if (m_bSimulation)
		{
			Vibrate_State& svdState = m_global_SVD_State;
			update_rhs_Global(nTimeStep);
			compute_rhs_ModalWrap(m_global_State, svdState);
			solve_linear_Subspace_problem(svdState);

			compute_Local_R(svdState);

			update_u_v_a_Global();
			//return;
			computeReferenceDisplace_ModalWarp(m_global_State, svdState);
		}
	}
#endif//SingleDomainISVD
	

	

	

#endif//USE_SUBSPACE_SVD
}//namespace YC
