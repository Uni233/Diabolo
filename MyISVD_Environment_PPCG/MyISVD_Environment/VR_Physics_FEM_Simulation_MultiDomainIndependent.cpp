#include "VR_Physics_FEM_Simulation_MultiDomainIndependent.h"
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

#include <set>
#include "MyIterMacro.h"
#include "QuaternionTools.h"
#include "MyIterMacro.h"

#if USE_PPCG
#include "PCG_with_residual_update.h"
#endif

namespace YC
{
	extern float g_scriptForceFactor;
	extern int cuda_bcMinCount;
	extern int cuda_bcMaxCount;

	VR_Physics_FEM_Simulation_MultiDomainIndependent::VR_Physics_FEM_Simulation_MultiDomainIndependent(void)
	{
		m_bSimulation = true;
		m_nGlobalDofs = Invalid_Id;
	}

	VR_Physics_FEM_Simulation_MultiDomainIndependent::~VR_Physics_FEM_Simulation_MultiDomainIndependent(void)
	{}

	bool VR_Physics_FEM_Simulation_MultiDomainIndependent::loadOctreeNode_MultidomainIndependent(const int nXCount, const int nYCount, const int nZCount)
	{
		std::vector< MyInt > XCount(LocalDomainCount, nXCount);
		std::vector< MyInt > YCount(LocalDomainCount, nYCount);
		std::vector< MyInt > ZCount(LocalDomainCount, nZCount);

		xMin = yMin = zMin = boost::math::tools::max_value<MyFloat>();
		xMax = yMax = zMax = boost::math::tools::min_value<MyFloat>();

		MyDenseVector coord(0, 0, 0);
		MyDenseVector xStep(1, 0, 0), yStep(0, 1, 0), zStep(0, 0, 1);
		std::vector< MyVector > elementData;
		MyVector cellInfo; cellInfo.resize(4);

		int xCountSum = 0;
		for (int i = 0; i < LocalDomainCount; ++i)
		{
			for (int x = 0; x < XCount[i]; ++x)
			{
				for (int y = 0; y < YCount[i]; ++y)
				{
					for (int z = 0; z < ZCount[i]; ++z)
					{
						coord = MyDenseVector((xCountSum + x) * 2 + 1, y * 2 + 1, z * 2 + 1);
						MyDenseVector c = coord * CellRaidus;

						cellInfo[0] = c[0];
						cellInfo[1] = c[1];
						cellInfo[2] = c[2];
						cellInfo[3] = CellRaidus;

						m_vec_cell.push_back(Cell::makeCell_DMI(MyPoint(cellInfo[0], cellInfo[1], cellInfo[2]), cellInfo[3], i));
						m_vec_cell[m_vec_cell.size() - 1]->computeCellType_DMI(i);
						m_vecLocalCellPool[i].push_back(m_vec_cell[m_vec_cell.size() - 1]);
					}
				}
			}
			xCountSum += XCount[i];
		}

#if USE_MODAL_WARP

		for (iterAllOf(itr, Vertex::s_vector_pair_ALM))
		{
			VertexPtr leftVtxPtr = (*itr).first;
			std::vector< CellPtr > tmpLeftShareCell = leftVtxPtr->m_vec_ShareCell;
			VertexPtr rightVtxPtr = (*itr).second;
			std::vector< CellPtr > tmpRightShareCell = rightVtxPtr->m_vec_ShareCell;

			leftVtxPtr->m_vec_ShareCell.insert((leftVtxPtr->m_vec_ShareCell).end(), tmpRightShareCell.begin(), tmpRightShareCell.end());
			rightVtxPtr->m_vec_ShareCell.insert((rightVtxPtr->m_vec_ShareCell).end(), tmpLeftShareCell.begin(), tmpLeftShareCell.end());
		}

		/*for (iterAllOf(itr, Vertex::s_vector_pair_ALM))
		{
			VertexPtr leftVtxPtr = (*itr).first;
			VertexPtr rightVtxPtr = (*itr).second;
			printf("Left %d , %d  Right %d , %d\n", leftVtxPtr->getId(), leftVtxPtr->m_vec_ShareCell.size(), rightVtxPtr->getId(), rightVtxPtr->m_vec_ShareCell.size());
		}
		MyExit;*/
#endif//USE_MODAL_WARP

		const MyInt nVtxSize = Vertex::getVertexSize();
		for (int v = 0; v < nVtxSize; ++v)
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

		printf("FEM (%d), EFG (%d), Couple(%d)\n", Cell::s_nFEM_Cell_Count, Cell::s_nEFG_Cell_Count, Cell::s_nCOUPLE_Cell_Count);
		printf("vertex count %d [%f , %f][%f , %f][%f , %f]\n", Vertex::getVertexSize(), xMin, xMax, yMin, yMax, zMin, zMax);
		printf("pair alm %d\n", Vertex::s_vector_pair_ALM.size());
		for (int i = 0; i < LocalDomainCount;++i)
		{
			printf("domain [%d] cell count [%d]\n",i,m_vecLocalCellPool[i].size());
		}
		return true;
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::distributeDof_local_Independent()
	{
		m_nDof_Q = Geometry::first_dof_idx;
		for (int i = 0; i < LocalDomainCount;++i)
		{
			std::vector< CellPtr >& refCurLocalCellPool = m_vecLocalCellPool[i];
			MyInt & curDofs = m_nLocalDofs[i] = Geometry::first_dof_idx;			
			const MyInt nCellSize = refCurLocalCellPool.size();

			for (MyInt c = 0; c < nCellSize; ++c)
			{
				CellPtr curCellPtr = refCurLocalCellPool[c];
				for (MyInt v = 0; v < Geometry::vertexs_per_cell; ++v)
				{
					VertexPtr curVtxPtr = curCellPtr->getVertex(v);
					if ((curVtxPtr->hasALM_Mate()))
					{
						curCellPtr->setLM_Boundary(v);
						if (!(curVtxPtr->isValidDof_Q()))
						{
							curVtxPtr->setDof_Q(m_nDof_Q, m_nDof_Q + 1, m_nDof_Q + 2);
							curVtxPtr->getALMPtr()->setDof_Q(m_nDof_Q, m_nDof_Q + 1, m_nDof_Q + 2);
							m_nDof_Q += 3;

							Q_ASSERT(!(curVtxPtr->isValidDof_DMI()));
							curVtxPtr->setDof_DMI(curDofs, curDofs + 1, curDofs + 2);
							curDofs += 3;
						}
					}
				}
			}
			for (MyInt c = 0; c < nCellSize; ++c)
			{
				CellPtr curCellPtr = refCurLocalCellPool[c];
				for (MyInt v = 0; v < Geometry::vertexs_per_cell; ++v)
				{
					VertexPtr curVtxPtr = curCellPtr->getVertex(v);
					if (!(curVtxPtr->isValidDof_DMI()))
					{
						curVtxPtr->setDof_DMI(curDofs, curDofs + 1, curDofs + 2);
						curDofs += 3;
					}
				}
			}

			m_localPhysicDomain[i].m_nLocalDomainDof = curDofs;
			printf("Local Dof %d.\n", curDofs);
			printf("m_nDof_Q %d.\n", m_nDof_Q); 
		}

		for (int i = 0; i < LocalDomainCount;++i)
		{
			m_localPhysicDomain[i].m_nGlobal_Dofs_Q = m_nDof_Q;
		}
		MyPause;
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::distributeDof_global()
	{
		MyVectorI dofsBase[LocalDomainCount];
		dofsBase[0] = MyVectorI(0,0,0);
		m_localPhysicDomain[0].m_nGlobalDofsBase = 0;
		
		for (int i = 1 MyNotice; i < LocalDomainCount;++i)
		{
			dofsBase[i] = dofsBase[i - 1] + MyVectorI(m_nLocalDofs[i-1], m_nLocalDofs[i-1], m_nLocalDofs[i-1]);
			m_localPhysicDomain[i].m_nGlobalDofsBase = dofsBase[i][0];
		}

		m_nGlobalDofs = Geometry::first_dof_idx;
		for (int i = 0; i < LocalDomainCount; ++i)
		{
			m_nGlobalDofs += m_nLocalDofs[i];
		}

		const MyInt& nVtxSize = Vertex::getVertexSize();
		for (int v = 0; v < nVtxSize;++v)
		{
			MyInt did = Vertex::getVertex(v)->getFromDomainId();
			MyVectorI curLocalDofs = Vertex::getVertex(v)->getDofs_DMI();
			Vertex::getVertex(v)->setDof_DMI_Global(curLocalDofs + dofsBase[did]);
		}

		for (int i = 0; i < LocalDomainCount; ++i)
		{
			m_localPhysicDomain[i].m_nGlobalDofs = m_nGlobalDofs;
			printf("Global Dofs Base [%d]\n", m_localPhysicDomain[i].m_nGlobalDofsBase);
		}
		printf("Global Dofs [%d]\n", m_nGlobalDofs); MyPause;

		/*for (int v = 0; v < Vertex::getVertexSize(); ++v)
		{
			const MyVectorI& nDofs = Vertex::getVertex(v)->getDofs_DMI_Global();
			printf("[%d,%d,%d,%d],", v, nDofs[0], nDofs[1], nDofs[2]);
		}
		printf("\n"); MyPause;*/
	}

	bool VR_Physics_FEM_Simulation_MultiDomainIndependent::isDCCondition_DMI(const MyPoint& pos)
	{
		if (pos[0] < CellRaidus /*|| pos[0] > (CellRaidus * 22*2) */)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::createDCBoundaryCondition_Independent()
	{
		for (int i = 0; i < LocalDomainCount; ++i)
		{
			Physic_State_DomainIndependent& refDomain = m_localPhysicDomain[i];
			refDomain.m_vecDCBoundaryCondition.clear();
			refDomain.m_vecForceBoundaryCondition.clear();
		}

		const MyInt nVtxSize = Vertex::getVertexSize();
		for (int v = 0; v < nVtxSize;++v)
		{
			VertexPtr curVtxPtr = Vertex::getVertex(v);
			const MyPoint & pos = curVtxPtr->getPos();
			const MyInt did = curVtxPtr->getFromDomainId();
			if (isDCCondition_DMI(pos))
			{
				m_localPhysicDomain[did].m_vecDCBoundaryCondition.push_back(curVtxPtr);
			}
		}
		for (int i = 0; i < LocalDomainCount; ++i)
		{
			Physic_State_DomainIndependent& refDomain = m_localPhysicDomain[i];
			refDomain.m_nLocalDC_Dofs = refDomain.m_vecDCBoundaryCondition.size() * MyDIM;
			printf("Domain[%d] DC Condition Count [%d]\n", i, refDomain.m_nLocalDC_Dofs);
		}
		MyPause;
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::setFromTriplets(MyVector & vec, const std::map<long, Cell::TripletNode >& RhsTripletNode)
	{
		std::map<long, Cell::TripletNode >::const_iterator itrRhs = RhsTripletNode.begin();
		vec.setZero();
		for (; itrRhs != RhsTripletNode.end(); ++itrRhs)
		{
			vec[itrRhs->first] = (itrRhs->second).val;
		}
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::setFromTriplets(MyMatrix &mat, const std::map<long, std::map<long, Cell::TripletNode > >& TripletNodeMap)
	{
		std::map<long, std::map<long, Cell::TripletNode > >::const_iterator itr_tri = TripletNodeMap.begin();
		for (; itr_tri != TripletNodeMap.end(); ++itr_tri)
		{
			const std::map<long, Cell::TripletNode >&  ref_map = itr_tri->second;
			std::map<long, Cell::TripletNode >::const_iterator itr_2 = ref_map.begin();
			for (; itr_2 != ref_map.end(); ++itr_2)
			{
				mat.coeffRef(itr_tri->first, itr_2->first) = (itr_2->second).val;
			}
		}
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::createGlobalMassAndStiffnessAndDampingMatrix_Independent()
	{
		for (int i = 0; i < LocalDomainCount;++i)
		{
			Physic_State_DomainIndependent& curDomainState = m_localPhysicDomain[i];
			curDomainState.resize();
			Cell::s_TripletNode_LocalMass_DMI[i].clear();
			Cell::s_TripletNode_LocalStiffness_DMI[i].clear();
			Cell::s_TripletNode_LocalRhs_DMI[i].clear();
			/*Cell::s_TripletNode_ModalWarp_DIM[i].clear();*/

			std::vector< CellPtr > & curCellVec = m_vecLocalCellPool[i];
			for (unsigned i = 0; i < curCellVec.size(); ++i)
			{
				CellPtr curCellPtr = curCellVec[i];
				curCellPtr->initialize_DMI();
				curCellPtr->assembleSystemMatrix_DMI();
			}

			setFromTriplets(curDomainState.m_LocalMatrix[PMI::StiffMat], Cell::s_TripletNode_LocalStiffness_DMI[i]);
			setFromTriplets(curDomainState.m_LocalMatrix[PMI::MassMat], Cell::s_TripletNode_LocalMass_DMI[i]);
			setFromTriplets(curDomainState.m_LocalVector[PMI::ComputeRhs_Vec], Cell::s_TripletNode_LocalRhs_DMI[i]);
			
		}
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::createNewMarkMatrix_DMI()
	{
		for (int i = 0; i < LocalDomainCount;++i)
		{
			Physic_State_DomainIndependent& curDomainState = m_localPhysicDomain[i];
			curDomainState.m_LocalMatrix[PMI::DampingMat] = Material::damping_alpha * curDomainState.m_LocalMatrix[PMI::MassMat]
				+ Material::damping_beta * curDomainState.m_LocalMatrix[PMI::StiffMat];

			curDomainState.m_LocalMatrix[PMI::ComputeMat] = curDomainState.m_LocalMatrix[PMI::StiffMat];
			curDomainState.m_LocalMatrix[PMI::ComputeMat] += m_db_NewMarkConstant[0] * curDomainState.m_LocalMatrix[PMI::MassMat];
			curDomainState.m_LocalMatrix[PMI::ComputeMat] += m_db_NewMarkConstant[1] * curDomainState.m_LocalMatrix[PMI::DampingMat];
		}
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::createConstraintMatrix_DMI()
	{
		MyMatrix globalConstraintMat(m_nDof_Q,m_nGlobalDofs);
		globalConstraintMat.setZero();

		MyFloat first_nonzero_diagonal_entry = 1;
		const unsigned int n_dofs = m_nGlobalDofs;
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);
		MyMatrix& ref_SampleMatrix = m_localPhysicDomain[0].m_LocalMatrix[PMI::ComputeMat];
		for (int v = 0; v < n_dofs; ++v)
		{
			if (!numbers::isZero(ref_SampleMatrix.coeff(v, v)))
			{
				first_nonzero_diagonal_entry = ref_SampleMatrix.coeff(v, v);
				break;
			}
		}

		std::set< std::pair< MyInt, MyInt > > vertexConstraintPairSet;
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c = 0; c<nCellSize; ++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];

			if ((curCellPtr->m_vec_index.size()) > 0)
			{
				MyMatrix Q/*,V*/;
				int faceId;
				VertexPtr curVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[0]);
				Q_ASSERT((curVtxPtr->getCreateOrder()) != Invalid_Id);
				if (0 == (curVtxPtr->getCreateOrder()))
				{
					for (int m = 0; m < (curCellPtr->m_vec_index.size()); ++m)
					{
						VertexPtr tmpVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[m]);
						vertexConstraintPairSet.insert(std::make_pair(tmpVtxPtr->getId(), tmpVtxPtr->getALMPtr()->getId()));
					}
				}
				else
				{
					for (int m = 0; m < (curCellPtr->m_vec_index.size()); ++m)
					{
						VertexPtr tmpVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[m]);
						vertexConstraintPairSet.insert(std::make_pair(tmpVtxPtr->getId(), tmpVtxPtr->getALMPtr()->getId()));
					}
				}

			}
		}

		const float scalefff = 1.f * first_nonzero_diagonal_entry;//100000.01f;//50000.01f;

		for (iterAllOf(itr, vertexConstraintPairSet))
		{
			VertexPtr leftVtxPtr = Vertex::getVertex((*itr).first);
			VertexPtr rightVtxPtr = Vertex::getVertex((*itr).second);
			const MyPoint& leftPos = leftVtxPtr->getPos();
			const MyPoint& rightPos = rightVtxPtr->getPos();
			Q_ASSERT(numbers::IsEqual(leftPos.x(), rightPos.x()) && numbers::IsEqual(leftPos.y(), rightPos.y()) && numbers::IsEqual(leftPos.z(), rightPos.z()));

			const MyVectorI& left_dofs_Q = leftVtxPtr->getDofs_Q();
			const MyVectorI& right_dofs_Q = rightVtxPtr->getDofs_Q();
			Q_ASSERT(left_dofs_Q.x() == right_dofs_Q.x() && left_dofs_Q.y() == right_dofs_Q.y() && left_dofs_Q.z() == right_dofs_Q.z());

			const MyVectorI& left_dofs_DMI = leftVtxPtr->getDofs_DMI_Global();
			const MyVectorI& right_dofs_DMI = rightVtxPtr->getDofs_DMI_Global();

			for (int i = 0; i < myDim; ++i)
			{
				globalConstraintMat.coeffRef(left_dofs_Q[i], left_dofs_DMI[i]) = -1.f*scalefff;
				globalConstraintMat.coeffRef(left_dofs_Q[i], right_dofs_DMI[i]) = 1.f*scalefff;
			}
		}

		for (int i = 0; i < LocalDomainCount;++i)
		{
			m_localPhysicDomain[i].getConstraintMat(globalConstraintMat);
		}

		/*printfMTX("d:\\globalConstraintMat.txt", globalConstraintMat);
		MyExit;*/
	}

#if USE_PPCG_Permutation
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::premutationGlobalMatrix()
	{
		MyMatrix& refA = m_globalPPCGDomain.m_LocalMatrix[PPCGI::B];
		MyMatrix& paifloat = m_globalPPCGDomain.m_matPermutation;
		Eigen::FullPivLU<MyMatrix> lu(refA.transpose());
		Eigen::MatrixXi pai = lu.permutationP().toDenseMatrix();

		paifloat.resize(pai.rows(), pai.cols()); 
		paifloat.setZero();
		for (int r = 0; r < pai.rows();++r)
		{
			for (int c = 0; c < pai.cols();++c)
			{
				if (pai.coeff(r,c))
				{
					paifloat.coeffRef(r, c) = 1.0;
				}
			}
		}

		refA *= paifloat;
		m_globalPPCGDomain.m_LocalMatrix[PPCGI::A] *= paifloat;
		m_globalPPCGDomain.m_LocalMatrix[PPCGI::A] = paifloat.transpose() * m_globalPPCGDomain.m_LocalMatrix[PPCGI::A];

		m_globalPPCGDomain.m_vecBoundaryFlag.resize(paifloat.cols());
		m_globalPPCGDomain.m_vecBoundaryFlag.setZero();
		const int nDCSize = m_globalPPCGDomain.m_vecDCBoundaryCondition.size();
		
		for (int i = 0; i < nDCSize;++i)
		{
			const MyVectorI& refDofs = m_globalPPCGDomain.m_vecDCBoundaryCondition[i]->getDofs_DMI_Global();
			m_globalPPCGDomain.m_vecBoundaryFlag[refDofs[0]] = 1.0;
			m_globalPPCGDomain.m_vecBoundaryFlag[refDofs[1]] = 1.0;
			m_globalPPCGDomain.m_vecBoundaryFlag[refDofs[2]] = 1.0;
		}

		m_globalPPCGDomain.m_vecBoundaryFlag = paifloat.transpose() * m_globalPPCGDomain.m_vecBoundaryFlag;
		//printf("A [%d, %d]\n", refA.rows(), refA.cols());
		/*printfMTX("d:\\refA.txt", refA);
		printfMTX("d:\\refA_hat.txt", refA * paifloat);		
		MyPause;*/

		

	}
#endif//USE_PPCG_Permutation

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::assembleGlobalMatrix_Independent()
	{
#if USE_PPCG
		m_globalPPCGDomain.m_nGlobalDofs = m_nGlobalDofs;
		m_globalPPCGDomain.m_nGlobal_Dofs_Q = m_nDof_Q;
		const MyInt n = m_nGlobalDofs;
		const MyInt m = m_nDof_Q;
		const MyInt n_m = n - m;
		m_globalPPCGDomain.resize();
		for (int i = 0; i < LocalDomainCount; ++i)
		{
			m_localPhysicDomain[i].assembleMatrixLocalToGlobal_PPCG(m_globalPPCGDomain);
			m_globalPPCGDomain.m_vecDCBoundaryCondition.insert(m_globalPPCGDomain.m_vecDCBoundaryCondition.end(),
				m_localPhysicDomain[i].m_vecDCBoundaryCondition.begin(), m_localPhysicDomain[i].m_vecDCBoundaryCondition.end());
		}

#if USE_PPCG_Permutation
		premutationGlobalMatrix();
#endif//USE_PPCG_Permutation
		//G22, , C, R_hat;
		m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1] = m_globalPPCGDomain.m_LocalMatrix[PPCGI::B].block(0,0,m,m);
		m_globalPPCGDomain.m_LocalMatrix[PPCGI::B2] = m_globalPPCGDomain.m_LocalMatrix[PPCGI::B].block(0, m, m, n_m);
		for (int i=0;i<m;++i)
		{
			if (numbers::isZero(m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1].coeff(i, i)))
			{
				MyError("numbers::isZero(m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1].coeff(i, i))");
			}
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1_Inv].coeffRef(i,i) = 1.f / m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1].coeff(i,i);
		}

		MyMatrix& refG22 = m_globalPPCGDomain.m_LocalMatrix[PPCGI::G22];
		MyMatrix& refA = m_globalPPCGDomain.m_LocalMatrix[PPCGI::A];
		for (int i=0;i<n_m;++i)
		{
			refG22.coeffRef(i, i) = 1.0 / refA.coeff(i+m,i+m);
		}
		/*std::cout << m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1_Inv] * m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1] << std::endl;
		MyPause;*/
#else//USE_PPCG
		m_globalPhysicDomain.m_nGlobalDofsBase = 0;
		m_globalPhysicDomain.m_nGlobalDofs = m_nGlobalDofs + m_nDof_Q;
		m_globalPhysicDomain.m_nGlobal_Dofs_Q = m_nDof_Q;
		m_globalPhysicDomain.m_nLocalDomainDof = m_globalPhysicDomain.m_nGlobalDofs;

		m_globalPhysicDomain.resize();
		m_globalPhysicDomain.printInfo();
		for (int i = 0; i < LocalDomainCount; ++i)
		{
			m_localPhysicDomain[i].assembleMatrixLocalToGlobal(m_globalPhysicDomain);
			m_globalPhysicDomain.m_vecDCBoundaryCondition.insert(m_globalPhysicDomain.m_vecDCBoundaryCondition.end(),
				m_localPhysicDomain[i].m_vecDCBoundaryCondition.begin(), m_localPhysicDomain[i].m_vecDCBoundaryCondition.end());
		}
#endif//USE_PPCG

		/*printfMTX("d:\\ComputeMat.txt", m_globalPhysicDomain.m_LocalMatrix[PMI::ComputeMat]);
		MyExit;*/
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::TestVtxCellId()
	{
		for (int v = 0; v<Vertex::getVertexSize(); ++v)
		{
			if (v != (Vertex::getVertex(v)->getId()))
			{
				printf("index %d; Id %d \n", v, (Vertex::getVertex(v)->getId()));
			}
		}

		for (int c = 0; c<m_vec_cell.size(); ++c)
		{
			if (c != m_vec_cell[c]->getID())
			{
				printf("index %d; cell Id %d \n", c, (m_vec_cell[c]->getID()));
			}
		}

		for (size_t i = 0; i < LocalDomainCount; i++)
		{
			const MyInt nCellSize = m_vecLocalCellPool[i].size();
			for (size_t c = 0; c < nCellSize; c++)
			{
				if (i != m_vecLocalCellPool[i][c]->getDomainId())
				{
					printf("local domain %d; cell domain id %d \n", c, (m_vec_cell[c]->getDomainId()));
				}
			}
		}

		for (int v = 0; v<Vertex::getVertexSize(); ++v)
		{
			VertexPtr vtxPtr = Vertex::getVertex(v);

			if ((vtxPtr->getShareCellCount()) != (vtxPtr->m_vec_ShareCell.size()))
			{
				printf("getShareCellCount %d; m_vec_ShareCell.size() %d \n", vtxPtr->getShareCellCount(), (vtxPtr->m_vec_ShareCell.size()));
			}
		}
		//MyPause;
	}

#if USE_MAKE_CELL_SURFACE
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::creat_Outter_Skin(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs, YC::MyIntVector& vecF_ColorId)
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
			for (int f = 0; f < Geometry::faces_per_cell; ++f)
			{
				vecCellFaceFlag[curCellFaceBase + f] = 0;
				MyDenseVector massCenter; massCenter.setZero();
				for (int v = 0; v < Geometry::vertexs_per_face; ++v)
				{
					massCenter += curCellPtr->getVertex(Order::indexQuads[f][v])->getPos();
				}
				massCenter /= Geometry::vertexs_per_face;

				if (numbers::IsEqual(massCenter.x(), xMax) ||
					numbers::IsEqual(massCenter.x(), xMin) ||
					numbers::IsEqual(massCenter.y(), yMax) ||
					numbers::IsEqual(massCenter.y(), yMin) ||
					numbers::IsEqual(massCenter.z(), zMax) ||
					numbers::IsEqual(massCenter.z(), zMin))
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
		vecF_ColorId.resize(OutterTriFaceSize); vecF_ColorId.setZero();

		for (int v = 0; v < nVtxSize; ++v)
		{
			if (vecVtxFaceFlag[v])
			{
				VertexPtr curVtxPtr = Vertex::getVertex(v);
				matV.row(vecVtxFaceIndex[v]) = curVtxPtr->getPos();
				matV_dofs.row(vecVtxFaceIndex[v]) = curVtxPtr->getDofs_DMI_Global();
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
					vecF_ColorId[(vecCellFaceIndex[curCellFaceBase + f] * 2)] = curCellPtr->getDomainId();

					vid0 = curCellPtr->getVertex(Order::indexQuads_Tri[f][1][0])->getId();
					vid1 = curCellPtr->getVertex(Order::indexQuads_Tri[f][1][1])->getId();
					vid2 = curCellPtr->getVertex(Order::indexQuads_Tri[f][1][2])->getId();
					matF.row(vecCellFaceIndex[curCellFaceBase + f] * 2 + 1) = MyVectorI(vecVtxFaceIndex[vid0], vecVtxFaceIndex[vid1], vecVtxFaceIndex[vid2]);
					vecF_ColorId[(vecCellFaceIndex[curCellFaceBase + f] * 2 + 1)] = curCellPtr->getDomainId();
				}
				//vecCellFaceFlag[curCellFaceBase + f] = 0;
			}
		}

		/*std::cout << matV << std::endl;
		std::cout << matF << std::endl;
		std::cout << "vecCellFaceIndex[nAllFaceSize] " << vecCellFaceIndex[nAllFaceSize] << std::endl;
		MyPause;*/
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::getSkinDisplacement(YC::MyVector& displacement, YC::MyMatrix& matV, YC::MyIntMatrix& matV_dofs, YC::MyMatrix& matU)
	{
		const MyInt nRows = matV_dofs.rows();
		const MyInt nCols = matV_dofs.cols();
		matU.resize(nRows, nCols);
		matU.setZero();


#if USE_TBB
		using namespace tbb;
		parallel_for(blocked_range<size_t>(0, nRows), ApplySkinDisplacement(&matV, &matU, &displacement, &matV_dofs), auto_partitioner());
#else
		for (int v = 0; v < nRows; v++)
		{
			const MyVectorI& dofs = matV_dofs.row(v);
			matU.row(v) = MyDenseVector(currentState.incremental_displacement[dofs.x()],
				currentState.incremental_displacement[dofs.y()],
				currentState.incremental_displacement[dofs.z()]
				);
		}
#endif//USE_TBB
	}
#endif//USE_MAKE_CELL_SURFACE

#if USE_PPCG
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::simulationOnCPU_DMI_Global_PPCG(const int nTimeStep)
	{
		update_rhs_DMI_Global_PPCG(nTimeStep);
		apply_boundary_values_DMI_Global_PPCG();
		solve_linear_problem_DMI_Global_PPCG();
		update_u_v_a_DMI_Global_PPCG();
		compute_Local_R_DMI_PPCG(getGlobalState_DMI_PPCG());
		update_displacement_ModalWrap_DMI_PPCG(getGlobalState_DMI_PPCG());
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::update_rhs_DMI_Global_PPCG(const int nStep)
	{
		Physic_PPCG_State_Independent & curState = m_globalPPCGDomain;
		MyVector& R_rhs = curState.m_LocalVector[PPCGI::R_rhs];
		MyVector& damping_rhs = curState.m_LocalVector[PPCGI::damping_rhs];
		MyVector& mass_rhs = curState.m_LocalVector[PPCGI::mass_rhs];
		MyVector& displacement = curState.m_LocalVector[PPCGI::displacement];
		MyVector& velocity = curState.m_LocalVector[PPCGI::velocity];
		MyVector& acceleration = curState.m_LocalVector[PPCGI::acceleration];
		MyVector&  computeRhs = curState.m_LocalVector[PPCGI::computeRhs];
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

		R_rhs += computeRhs;

		R_rhs += curState.m_LocalMatrix[PPCGI::A_mass] * mass_rhs;
		R_rhs += curState.m_LocalMatrix[PPCGI::A_damping] * damping_rhs;


		if (nStep >= cuda_bcMinCount && nStep < cuda_bcMaxCount)
		{
			R_rhs += curState.m_LocalVector[PPCGI::R_rhs_externalForce];
			//MyPause;
		}

#if USE_PPCG_Permutation
		R_rhs = m_globalPPCGDomain.m_matPermutation.transpose() * R_rhs;
#endif
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::apply_boundary_values_DMI_Global_PPCG()
	{
		Physic_PPCG_State_Independent & curState = m_globalPPCGDomain;
		std::vector< VertexPtr >& vecBoundaryVtx = curState.m_vecDCBoundaryCondition;
		MyMatrix&  computeMatrix = curState.m_LocalMatrix[PPCGI::A];
		MyVector& curRhs = curState.m_LocalVector[PPCGI::R_rhs];
		MyVector& curDisplacement = curState.m_LocalVector[PPCGI::incremental_displacement];
		if (vecBoundaryVtx.size() == 0)
			return;


		const unsigned int n_dofs = curState.m_nGlobalDofs - curState.m_nGlobal_Dofs_Q;
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);

		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = computeMatrix.coeff(v, v);
			//printf("numbers::isZero(diagnosticValue[v]) %f\n", diagnosticValue[v]);
			Q_ASSERT(!numbers::isZero(diagnosticValue[v]));
		}

		MyFloat first_nonzero_diagonal_entry = 1;
		for (unsigned int i = 0; i < n_dofs; ++i)
		{
			if (!numbers::isZero(diagnosticValue[i]))
			{
				first_nonzero_diagonal_entry = diagnosticValue[i];
				break;
			}
		}
#if USE_PPCG_Permutation
		MyVector& dcflag = curState.m_vecBoundaryFlag;
		const int dcSize = dcflag.size();
		for (int i = 0; i < dcSize;++i)
		{
			if (dcflag[i]>0)
			{
				const unsigned int dof_number = i;
				setMatrixRowZeroWithoutDiag(computeMatrix, curState.m_nGlobalDofs - curState.m_nGlobal_Dofs_Q, dof_number);

				MyFloat new_rhs;
				if (!numbers::isZero(diagnosticValue[dof_number]))
				{
					new_rhs = 0 * diagnosticValue[dof_number];
					curRhs(dof_number) = new_rhs;
				}
				else
				{
					computeMatrix.coeffRef(dof_number, dof_number) = first_nonzero_diagonal_entry;
					new_rhs = 0 * first_nonzero_diagonal_entry;
					curRhs(dof_number) = new_rhs;
				}
				curDisplacement(dof_number) = 0;
			}
		}
#else//USE_PPCG_Permutation
		for (unsigned i = 0; i < vecBoundaryVtx.size(); ++i)
		{
			VertexPtr curVtxPtr = vecBoundaryVtx[i];
			const MyVectorI& Dofs = curVtxPtr->getDofs_DMI_Global();
			for (unsigned c = 0; c < MyDIM; ++c)
			{
				const unsigned int dof_number = Dofs[c];
				setMatrixRowZeroWithoutDiag(computeMatrix, curState.m_nGlobalDofs - curState.m_nGlobal_Dofs_Q, dof_number);

				MyFloat new_rhs;
				if (!numbers::isZero(diagnosticValue[dof_number]))
				{
					new_rhs = 0 * diagnosticValue[dof_number];
					curRhs(dof_number) = new_rhs;
				}
				else
				{
					computeMatrix.coeffRef(dof_number, dof_number) = first_nonzero_diagonal_entry;
					new_rhs = 0 * first_nonzero_diagonal_entry;
					curRhs(dof_number) = new_rhs;
				}
				curDisplacement(dof_number) = 0;
			}
		}
#endif//USE_PPCG_Permutation
	}
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::solve_linear_problem_DMI_Global_PPCG()
	{
		Physic_PPCG_State_Independent  & curState_0 = m_globalPPCGDomain;
		
#define CGSolverTolerance (1e-6f)

		const int n = m_globalPPCGDomain.m_nGlobalDofs;
		const int m = m_globalPPCGDomain.m_nGlobal_Dofs_Q;
		
		static float r_nrm2;
		r_nrm2 = m_globalPPCGDomain.m_LocalVector[PPCGI::R_rhs].norm();
		PCG_RU::MyMonitor<MyFloat> monitor(r_nrm2, n + m, CGSolverTolerance);

		MyVector& displace = m_globalPPCGDomain.m_LocalVector[PPCGI::incremental_displacement];



		displace = PCG_RU::PCG_Residual_Update_Dense_Schilders(
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::A], 
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::B],
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1], 
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::B2], 
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::C],
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::G22],
			m_globalPPCGDomain.m_LocalMatrix[PPCGI::B1_Inv],
			m_globalPPCGDomain.m_LocalVector[PPCGI::R_rhs],
			m_globalPPCGDomain.m_LocalVector[PPCGI::d], monitor);

#if USE_PPCG_Permutation
		displace = m_globalPPCGDomain.m_matPermutation * displace;
#endif
		return;
		using namespace YC;
		Physic_PPCG_State_Independent & curState = m_globalPPCGDomain;
		static bool bFirst = true;
		static Eigen::ColPivHouseholderQR< MyMatrix > ALM_QR;

		MyMatrix&  computeMatrix = curState.m_LocalMatrix[PPCGI::A];
		MyVector& curRhs = curState.m_LocalVector[PPCGI::R_rhs];
		MyVector& curDisplacement = curState.m_LocalVector[PPCGI::incremental_displacement];
		
		if (bFirst)
		{
			bFirst = false;
			ALM_QR = computeMatrix.colPivHouseholderQr();
		}

		curDisplacement = ALM_QR.solve(curRhs);
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::update_u_v_a_DMI_Global_PPCG()
	{
		Physic_PPCG_State_Independent & curState = m_globalPPCGDomain;
		const MyVector& solu = curState.m_LocalVector[PPCGI::incremental_displacement];
		MyVector& disp_vec = curState.m_LocalVector[PPCGI::displacement];
		MyVector& vel_vec = curState.m_LocalVector[PPCGI::velocity];
		MyVector& acc_vec = curState.m_LocalVector[PPCGI::acceleration];
		MyVector& old_acc = curState.m_LocalVector[PPCGI::old_acceleration];
		MyVector& old_solu = curState.m_LocalVector[PPCGI::old_displacement];

		old_solu = disp_vec;
		disp_vec = solu;
		old_acc = acc_vec;

		acc_vec *= (-1 * m_db_NewMarkConstant[3]);//    acc_vec.scale(-1 * m_db_NewMarkConstant[3]);
		acc_vec += (disp_vec * m_db_NewMarkConstant[0]); //acc_vec.add(m_db_NewMarkConstant[0], disp_vec);
		acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));//acc_vec.add(-1 * m_db_NewMarkConstant[0], old_solu);
		acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));//acc_vec.add(-1 * m_db_NewMarkConstant[2],vel_vec);

		vel_vec += (old_acc * m_db_NewMarkConstant[6]);//vel_vec.add(m_db_NewMarkConstant[6],old_acc);
		vel_vec += (acc_vec * m_db_NewMarkConstant[7]);//vel_vec.add(m_db_NewMarkConstant[7],acc_vec);

	}
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::update_displacement_ModalWrap_DMI_PPCG(Physic_PPCG_State_Independent& dmiState)
	{
		//printfMTX("d:\\error_R_hat.txt", dmiState.m_LocalMatrix[PMI::RhatMat]); MyExit;
		dmiState.m_LocalVector[PPCGI::incremental_displacement] = dmiState.m_LocalMatrix[PPCGI::R_hat] * dmiState.m_LocalVector[PPCGI::incremental_displacement];
		/*std::ofstream outfile("d:\\error_vec.txt");
		outfile << dmiState.m_LocalVector[PMI::incremental_displacement_Vec] << std::endl;
		outfile.close();
		MyExit;*/
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::compute_Local_R_DMI_PPCG(Physic_PPCG_State_Independent& dmiState)
	{
		compute_ModalWrap_Rotation_DMI_PPCG(dmiState.m_LocalVector[PPCGI::incremental_displacement], dmiState.m_LocalMatrix[PPCGI::R_hat]);
		//printfMTX("d:\\error_R_hat.txt", dmiState.m_LocalMatrix[PMI::RhatMat]); MyExit;
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::compute_ModalWrap_Rotation_DMI_PPCG(const MyVector& globalDisplacement, MyMatrix& modalWrap_R_hat)
	{

		MyVector w, translation;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		for (int c = 0; c < m_vec_cell.size(); ++c)
		{
			m_vec_cell[c]->TestModalWrapMatrix_DMI(globalDisplacement, w, translation, cellDisplace);
			m_vec_cell[c]->m_vec_LocalQuat.resize(Geometry::vertexs_per_cell);
			MyDenseVector ww; ww.setZero();
			MyDenseVector W, W_hat;
			float w_norm;
			for (int m = 0; m < 8; ++m)
			{
				W = MyDenseVector(w[3 * m + 0], w[3 * m + 1], w[3 * m + 2]);
				W_hat = W;
				W_hat.normalize();
				w_norm = W.norm();

				if (numbers::isZero(w_norm))
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion();
				}
				else
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion(Vec3<float>(W_hat[0], W_hat[1], W_hat[2]), w_norm);
				}

			}
		}


		std::vector< Eigen::Triplet<MyFloat, long> > vec_triplet;
		MySpMat tmpSpMat(globalDisplacement.size(), globalDisplacement.size());
		modalWrap_R_hat.resize(globalDisplacement.size(), globalDisplacement.size()); modalWrap_R_hat.setZero();
		MyMatrix_3X3 tmpR;
		//m_vec_VtxLocalRotaM.clear();
		//for (int v = 0; v < Vertex::getVertexSize(); ++v)
		for (int v = Vertex::getVertexSize() - 1; v >= 0; --v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec; cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getDofs_DMI_Global();

			const int nSize = refVec.size();
			const int addAmount = nSize * Geometry::vertexs_per_cell;

			Axis::Quaternion firstQuat = refVec[0]->m_vec_LocalQuat[0];
			for (int c = 0; c < refVec.size(); ++c)
			{
				for (int i = 0; i < Geometry::vertexs_per_cell; ++i)
				{
					Axis::Quaternion newRotaQuat = refVec[c]->m_vec_LocalQuat[i];
					curQuat = AverageQuaternion(cumulateVec, newRotaQuat, firstQuat, addAmount);
				}
			}

			Vec3<Axis::MySReal> WW = curQuat.toEulerVector();

			MyDenseVector W, W_hat;
			float w_norm;
			W = MyDenseVector(WW.x(), WW.y(), WW.z());
			W_hat = W;
			W_hat.normalize();
			w_norm = W.norm();
			MyMatrix_3X3 X; X.setZero();

			X.coeffRef(0, 1) = -1.f*W_hat[2];
			X.coeffRef(0, 2) = W_hat[1];

			X.coeffRef(1, 0) = W_hat[2];
			X.coeffRef(1, 2) = -1.f*W_hat[0];

			X.coeffRef(2, 0) = -1.f * W_hat[1];
			X.coeffRef(2, 1) = W_hat[0];

			if (numbers::isZero(w_norm))
			{
				//std::cout << "numbers::isZero(W_norm)" << std::endl; 
				tmpR = MyDenseMatrix::Identity(3, 3);
			}
			else
			{
				tmpR = MyDenseMatrix::Identity(3, 3) + (X * (1 - cos(w_norm)) / w_norm) + X * X * (1 - sin(w_norm) / w_norm);

			}
			//m_vec_VtxLocalRotaM.push_back(tmpR);
			//curQuat.toMatrix(tmpR);

			//std::cout << tmpR << std::endl; MyPause;
			for (int r = 0; r < 3; ++r)
			{
				for (int c = 0; c < 3; ++c)
				{
					vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(dofs[r], dofs[c], tmpR.coeff(r, c)));
					//modalWrap_R_hat.coeffRef(dofs[r], dofs[c]) += tmpR.coeff(r, c);
					//printf("[%d,%d][%f],", dofs[r], dofs[c], tmpR.coeff(r, c));
				}
			}
			//printf("\n"); MyPause;
		}

		tmpSpMat.setFromTriplets(vec_triplet.begin(), vec_triplet.end());
		modalWrap_R_hat = tmpSpMat;
	}
#else
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::simulationOnCPU_DMI_Global(const int nTimeStep)
	{
		update_rhs_DMI_Global(nTimeStep);
		apply_boundary_values_DMI_Global();
		solve_linear_problem_DMI_Global();
		update_u_v_a_DMI_Global();
		compute_Local_R_DMI(getGlobalState_DMI());
		update_displacement_ModalWrap_DMI(getGlobalState_DMI());
	}
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::update_rhs_DMI_Global(const int nStep)
	{
		Physic_State_DomainIndependent & curState = m_globalPhysicDomain;
		MyVector& R_rhs = curState.m_LocalVector[PMI::R_rhs_Vec];
		MyVector& damping_rhs = curState.m_LocalVector[PMI::damping_rhs_Vec];
		MyVector& mass_rhs = curState.m_LocalVector[PMI::mass_rhs_Vec];
		MyVector& displacement = curState.m_LocalVector[PMI::displacement_Vec];
		MyVector& velocity = curState.m_LocalVector[PMI::velocity_Vec];
		MyVector& acceleration = curState.m_LocalVector[PMI::acceleration_Vec];
		MyVector&  computeRhs = curState.m_LocalVector[PMI::ComputeRhs_Vec];
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

		R_rhs += computeRhs;

		R_rhs += curState.m_LocalMatrix[PMI::MassMat] * mass_rhs;
		R_rhs += curState.m_LocalMatrix[PMI::DampingMat] * damping_rhs;


		if (nStep >= cuda_bcMinCount && nStep < cuda_bcMaxCount)
		{
			R_rhs += curState.m_LocalVector[PMI::R_rhs_externalForce_Vec];
			//MyPause;
		}
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::apply_boundary_values_DMI_Global()
	{
		Physic_State_DomainIndependent & curState = m_globalPhysicDomain;
		std::vector< VertexPtr >& vecBoundaryVtx = curState.m_vecDCBoundaryCondition;
		MyMatrix&  computeMatrix = curState.m_LocalMatrix[PMI::ComputeMat];
		MyVector& curRhs = curState.m_LocalVector[PMI::R_rhs_Vec];
		MyVector& curDisplacement = curState.m_LocalVector[PMI::incremental_displacement_Vec];
		if (vecBoundaryVtx.size() == 0)
			return;


		const unsigned int n_dofs = curState.m_nGlobalDofs - curState.m_nGlobal_Dofs_Q;
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);

		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = computeMatrix.coeff(v, v);
			//printf("numbers::isZero(diagnosticValue[v]) %f\n", diagnosticValue[v]);
			Q_ASSERT(!numbers::isZero(diagnosticValue[v]));
		}

		MyFloat first_nonzero_diagonal_entry = 1;
		for (unsigned int i = 0; i < n_dofs; ++i)
		{
			if (!numbers::isZero(diagnosticValue[i]))
			{
				first_nonzero_diagonal_entry = diagnosticValue[i];
				break;
			}
		}

		for (unsigned i = 0; i < vecBoundaryVtx.size(); ++i)
		{
			VertexPtr curVtxPtr = vecBoundaryVtx[i];
			const MyVectorI& Dofs = curVtxPtr->getDofs_DMI_Global();
			for (unsigned c = 0; c < MyDIM; ++c)
			{
				const unsigned int dof_number = Dofs[c];
				setMatrixRowZeroWithoutDiag(computeMatrix, curState.m_nGlobalDofs - curState.m_nGlobal_Dofs_Q, dof_number);

				MyFloat new_rhs;
				if (!numbers::isZero(diagnosticValue[dof_number]))
				{
					new_rhs = 0 * diagnosticValue[dof_number];
					curRhs(dof_number) = new_rhs;
				}
				else
				{
					computeMatrix.coeffRef(dof_number, dof_number) = first_nonzero_diagonal_entry;
					new_rhs = 0 * first_nonzero_diagonal_entry;
					curRhs(dof_number) = new_rhs;
				}
				curDisplacement(dof_number) = 0;
			}
		}
	}
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::solve_linear_problem_DMI_Global()
	{
		using namespace YC;
		Physic_State_DomainIndependent & curState = m_globalPhysicDomain;
		static bool bFirst = true;
		static Eigen::ColPivHouseholderQR< MyMatrix > ALM_QR;

		MyMatrix&  computeMatrix = curState.m_LocalMatrix[PMI::ComputeMat];
		MyVector& curRhs = curState.m_LocalVector[PMI::R_rhs_Vec];
		MyVector& curDisplacement = curState.m_LocalVector[PMI::incremental_displacement_Vec];

		/*for (int i = 0; i < 864; ++i)
		{
		curDisplacement[i] = displaceData[i];
		}
		return;*/

		if (bFirst)
		{
			bFirst = false;
			ALM_QR = computeMatrix.colPivHouseholderQr();
		}

		curDisplacement = ALM_QR.solve(curRhs);
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::update_u_v_a_DMI_Global()
	{
		Physic_State_DomainIndependent & curState = m_globalPhysicDomain;
		const MyVector& solu = curState.m_LocalVector[PMI::incremental_displacement_Vec];
		MyVector& disp_vec = curState.m_LocalVector[PMI::displacement_Vec];
		MyVector& vel_vec = curState.m_LocalVector[PMI::velocity_Vec];
		MyVector& acc_vec = curState.m_LocalVector[PMI::acceleration_Vec];
		MyVector& old_acc = curState.m_LocalVector[PMI::old_acceleration_Vec];
		MyVector& old_solu = curState.m_LocalVector[PMI::old_displacement_Vec];

		old_solu = disp_vec;
		disp_vec = solu;
		old_acc = acc_vec;

		acc_vec *= (-1 * m_db_NewMarkConstant[3]);//    acc_vec.scale(-1 * m_db_NewMarkConstant[3]);
		acc_vec += (disp_vec * m_db_NewMarkConstant[0]); //acc_vec.add(m_db_NewMarkConstant[0], disp_vec);
		acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));//acc_vec.add(-1 * m_db_NewMarkConstant[0], old_solu);
		acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));//acc_vec.add(-1 * m_db_NewMarkConstant[2],vel_vec);

		vel_vec += (old_acc * m_db_NewMarkConstant[6]);//vel_vec.add(m_db_NewMarkConstant[6],old_acc);
		vel_vec += (acc_vec * m_db_NewMarkConstant[7]);//vel_vec.add(m_db_NewMarkConstant[7],acc_vec);

	}
	void VR_Physics_FEM_Simulation_MultiDomainIndependent::update_displacement_ModalWrap_DMI(Physic_State_DomainIndependent& dmiState)
	{
		//printfMTX("d:\\error_R_hat.txt", dmiState.m_LocalMatrix[PMI::RhatMat]); MyExit;
		dmiState.m_LocalVector[PMI::incremental_displacement_Vec] = dmiState.m_LocalMatrix[PMI::RhatMat] * dmiState.m_LocalVector[PMI::incremental_displacement_Vec];
		/*std::ofstream outfile("d:\\error_vec.txt");
		outfile << dmiState.m_LocalVector[PMI::incremental_displacement_Vec] << std::endl;
		outfile.close();
		MyExit;*/
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::compute_Local_R_DMI(Physic_State_DomainIndependent& dmiState)
	{
		compute_ModalWrap_Rotation_DMI(dmiState.m_LocalVector[PMI::incremental_displacement_Vec], dmiState.m_LocalMatrix[PMI::RhatMat]);
		//printfMTX("d:\\error_R_hat.txt", dmiState.m_LocalMatrix[PMI::RhatMat]); MyExit;
	}

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::compute_ModalWrap_Rotation_DMI(const MyVector& globalDisplacement, MyMatrix& modalWrap_R_hat)
	{

		MyVector w, translation;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		for (int c = 0; c < m_vec_cell.size(); ++c)
		{
			m_vec_cell[c]->TestModalWrapMatrix_DMI(globalDisplacement, w, translation, cellDisplace);
			m_vec_cell[c]->m_vec_LocalQuat.resize(Geometry::vertexs_per_cell);
			MyDenseVector ww; ww.setZero();
			MyDenseVector W, W_hat;
			float w_norm;
			for (int m = 0; m < 8; ++m)
			{
				W = MyDenseVector(w[3 * m + 0], w[3 * m + 1], w[3 * m + 2]);
				W_hat = W;
				W_hat.normalize();
				w_norm = W.norm();

				if (numbers::isZero(w_norm))
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion();
				}
				else
				{
					m_vec_cell[c]->m_vec_LocalQuat[m] = Axis::Quaternion(Vec3<float>(W_hat[0], W_hat[1], W_hat[2]), w_norm);
				}

			}
		}


		std::vector< Eigen::Triplet<MyFloat, long> > vec_triplet;
		MySpMat tmpSpMat(globalDisplacement.size(), globalDisplacement.size());
		modalWrap_R_hat.resize(globalDisplacement.size(), globalDisplacement.size()); modalWrap_R_hat.setZero();
		MyMatrix_3X3 tmpR;
		//m_vec_VtxLocalRotaM.clear();
		//for (int v = 0; v < Vertex::getVertexSize(); ++v)
		for (int v = Vertex::getVertexSize() - 1; v >= 0; --v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec; cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getDofs_DMI_Global();

			const int nSize = refVec.size();
			const int addAmount = nSize * Geometry::vertexs_per_cell;

			Axis::Quaternion firstQuat = refVec[0]->m_vec_LocalQuat[0];
			for (int c = 0; c < refVec.size(); ++c)
			{
				for (int i = 0; i < Geometry::vertexs_per_cell; ++i)
				{
					Axis::Quaternion newRotaQuat = refVec[c]->m_vec_LocalQuat[i];
					curQuat = AverageQuaternion(cumulateVec, newRotaQuat, firstQuat, addAmount);
				}
			}

			Vec3<Axis::MySReal> WW = curQuat.toEulerVector();

			MyDenseVector W, W_hat;
			float w_norm;
			W = MyDenseVector(WW.x(), WW.y(), WW.z());
			W_hat = W;
			W_hat.normalize();
			w_norm = W.norm();
			MyMatrix_3X3 X; X.setZero();

			X.coeffRef(0, 1) = -1.f*W_hat[2];
			X.coeffRef(0, 2) = W_hat[1];

			X.coeffRef(1, 0) = W_hat[2];
			X.coeffRef(1, 2) = -1.f*W_hat[0];

			X.coeffRef(2, 0) = -1.f * W_hat[1];
			X.coeffRef(2, 1) = W_hat[0];

			if (numbers::isZero(w_norm))
			{
				//std::cout << "numbers::isZero(W_norm)" << std::endl; 
				tmpR = MyDenseMatrix::Identity(3, 3);
			}
			else
			{
				tmpR = MyDenseMatrix::Identity(3, 3) + (X * (1 - cos(w_norm)) / w_norm) + X * X * (1 - sin(w_norm) / w_norm);

			}
			//m_vec_VtxLocalRotaM.push_back(tmpR);
			//curQuat.toMatrix(tmpR);

			//std::cout << tmpR << std::endl; MyPause;
			for (int r = 0; r < 3; ++r)
			{
				for (int c = 0; c < 3; ++c)
				{
					vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(dofs[r], dofs[c], tmpR.coeff(r, c)));
					//modalWrap_R_hat.coeffRef(dofs[r], dofs[c]) += tmpR.coeff(r, c);
					//printf("[%d,%d][%f],", dofs[r], dofs[c], tmpR.coeff(r, c));
				}
			}
			//printf("\n"); MyPause;
		}

		tmpSpMat.setFromTriplets(vec_triplet.begin(), vec_triplet.end());
		modalWrap_R_hat = tmpSpMat;
	}
#endif


	void VR_Physics_FEM_Simulation_MultiDomainIndependent::setMatrixRowZeroWithoutDiag(MyMatrix& matrix, const int nDofs, const int  rowIdx)
	{
		
		for (size_t c = 0; c < nDofs; c++)
		{
			if (rowIdx != c)
			{
				matrix.coeffRef(rowIdx, c) = MyNull;
			}
		}
	}
	

	void VR_Physics_FEM_Simulation_MultiDomainIndependent::printfMTX(const char* lpszFileName, const MyMatrix& Mat)
	{
		std::ofstream outfile(lpszFileName);
		const int nRows = Mat.rows();
		const int nCols = Mat.cols();
		const int nValCount = nRows * nCols;

		outfile << nRows << " " << nCols << " " << nValCount << std::endl;
		for (size_t r = 0; r < nRows; r++)
		{
			for (size_t c = 0; c < nCols; c++)
			{
				if (!numbers::isZero(Mat.coeff(r, c)))
				{
					outfile << r << " " << c << " " << Mat.coeff(r, c) << std::endl;
				}
				
			}
		}
		outfile.close();
	}

}//namespace YC