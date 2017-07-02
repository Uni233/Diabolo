#include "VR_Physics_FEM_Simulation_MultiDomain.h"
#include "CG/solvercontrol.h"
#include "CG/solvercg.h"
#include "CG/preconditionssor.h"
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

#include "PCG_with_residual_update.h"
#include <set>
#include "MyIterMacro.h"
#include "GaussElimination.h"

#include "QuaternionTools.h"

namespace YC
{
#if USE_MultiDomain
	extern float g_scriptForceFactor;
	extern int cuda_bcMinCount;
	extern int cuda_bcMaxCount;

	VR_Physics_FEM_Simulation_MultiDomain::VR_Physics_FEM_Simulation_MultiDomain(void)
	{
		m_bSimulation = true;
	}

	VR_Physics_FEM_Simulation_MultiDomain::~VR_Physics_FEM_Simulation_MultiDomain(void)
	{}

	void VR_Physics_FEM_Simulation_MultiDomain::printMatrix(const MySpMat& spMat, const std::string& lpszFile)
	{
		return;
		MyMatrix tmp(spMat.rows(), spMat.cols());
		tmp.setZero();
		tmp = spMat;
		std::ofstream outfileBefore(std::string("d:\\") + lpszFile + std::string(".txt"));
		outfileBefore << tmp << std::endl;
		outfileBefore.close();
	}

	void VR_Physics_FEM_Simulation_MultiDomain::printVector(const MyVector& vec, const std::string& lpszFile)
	{
		return;
		std::ofstream outfileBefore(std::string("d:\\") + lpszFile + std::string(".txt"));
		outfileBefore << vec << std::endl;
		outfileBefore.close();
	}

	void VR_Physics_FEM_Simulation_MultiDomain::setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx)
	{
		{
			for (MySpMat::InnerIterator it(matrix, rowIdx); it; ++it)
			{
				Q_ASSERT(rowIdx == it.row());
				const int r = it.row();
				const int c = it.col();
				if (r == rowIdx && (r != c))
				{
					it.valueRef() = MyNull;
				}
			}
		}
	}

	bool VR_Physics_FEM_Simulation_MultiDomain::loadOctreeNode_MultiDomain(const int nXCount, const int nYCount, const int nZCount)
	{
		std::vector< MyInt > XCount(LocalDomainCount,nXCount);
		std::vector< MyInt > YCount(LocalDomainCount, nYCount);
		std::vector< MyInt > ZCount(LocalDomainCount, nZCount);

		xMin = yMin = zMin = boost::math::tools::max_value<MyFloat>();
		xMax = yMax = zMax = boost::math::tools::min_value<MyFloat>();

		MyDenseVector coord(0, 0, 0);
		MyDenseVector xStep(1, 0, 0), yStep(0, 1, 0), zStep(0, 0, 1);
		std::vector< MyVector > elementData;
		MyVector cellInfo; cellInfo.resize(4);

		int xCountSum = 0;
		for (int i = 0; i<LocalDomainCount; ++i)
		{
			for (int x = 0; x < XCount[i]; ++x)
			{
				for (int y = 0; y<YCount[i]; ++y)
				{
					for (int z = 0; z<ZCount[i]; ++z)
					{
						coord = MyDenseVector((xCountSum/*i*XCount[i]*/ + x) * 2 + 1, y * 2 + 1, z * 2 + 1);
						//coord = xStep * (x+1) + yStep*(y+1) + zStep * (z+1);
						MyDenseVector c = coord * CellRaidus;

						cellInfo[0] = c[0];
						cellInfo[1] = c[1];
						cellInfo[2] = c[2];
						cellInfo[3] = CellRaidus;
						//elementData.push_back(cellInfo);

						m_vec_cell.push_back(Cell::makeCell_ALM(MyPoint(cellInfo[0], cellInfo[1], cellInfo[2]), cellInfo[3], i));
						m_vec_cell[m_vec_cell.size() - 1]->computeCellType_ALM(i);
					}
				}
			}
			xCountSum += XCount[i];
		}

#if USE_MODAL_WARP
		
		/*for (iterAllOf(itr, Vertex::s_vector_pair_ALM))
		{
			VertexPtr leftVtxPtr = (*itr).first;
			std::vector< CellPtr > tmpLeftShareCell = leftVtxPtr->m_vec_ShareCell;
			VertexPtr rightVtxPtr = (*itr).second;
			std::vector< CellPtr > tmpRightShareCell = rightVtxPtr->m_vec_ShareCell;

			leftVtxPtr->m_vec_ShareCell.insert((leftVtxPtr->m_vec_ShareCell).end(), tmpRightShareCell.begin(), tmpRightShareCell.end());
			rightVtxPtr->m_vec_ShareCell.insert((rightVtxPtr->m_vec_ShareCell).end(), tmpLeftShareCell.begin(), tmpLeftShareCell.end());
		}*/
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

		return true;
	}

	bool VR_Physics_FEM_Simulation_MultiDomain::isLagrangeMultiplierCell(CellPtr curCellPtr)
	{
		return true;
#define LMStep (CellRaidus)
		static MyPoint constraintPos[4] = { MyPoint(0.f, yMin + LMStep, zMin + LMStep), MyPoint(0.f, yMin + LMStep, zMax - LMStep), MyPoint(0.f, yMax - LMStep, zMin + LMStep), MyPoint(0.f, yMax - LMStep, zMax - LMStep) };
		const MyPoint& center = curCellPtr->getCellCenter();
		for (int i = 0; i < 4;++i)
		{
			if (numbers::IsEqual(center.y(), constraintPos[i].y()) && numbers::isEqual(center.z(), constraintPos[i].z()))
			{
				return true;
			}
		}
		return false;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::distributeDof_ALM_PPCG()
	{
		m_nDof_ALM = Geometry::first_dof_idx;
		m_nDof_Q = Geometry::first_dof_idx;
		const MyInt nCellSize = m_vec_cell.size();

		{
			for (MyInt c = 0; c<nCellSize; ++c)
			{
				CellPtr curCellPtr = m_vec_cell[c];
				if (isLagrangeMultiplierCell(curCellPtr))
				{
					for (MyInt v = 0; v < Geometry::vertexs_per_cell; ++v)
					{
						VertexPtr curVtxPtr = curCellPtr->getVertex(v);
						if ((curVtxPtr->hasALM_Mate()))
						{
							curCellPtr->setLM_Boundary(v);
							if (!(curVtxPtr->isValidDof_Q()))
							{
								curVtxPtr->setDof_Q(m_nDof_Q, m_nDof_Q + 1, m_nDof_Q + 2);
								curVtxPtr->m_ALM_Ptr->setDof_Q(m_nDof_Q, m_nDof_Q + 1, m_nDof_Q + 2);
								m_nDof_Q += 3;

								Q_ASSERT(!(curVtxPtr->isValidDof_ALM()));
								curVtxPtr->setDof_ALM(m_nDof_ALM, m_nDof_ALM + 1, m_nDof_ALM + 2);
								m_nDof_ALM += 3;

							}
						}
					}
				}
			}
		}
		for (MyInt c = 0; c<nCellSize; ++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			for (MyInt v = 0; v<Geometry::vertexs_per_cell; ++v)
			{
				VertexPtr curVtxPtr = curCellPtr->getVertex(v);
				if (!(curVtxPtr->isValidDof_ALM()))
				{
					curVtxPtr->setDof_ALM(m_nDof_ALM, m_nDof_ALM + 1, m_nDof_ALM + 2);
					m_nDof_ALM += 3;
				}
			}
		}
		printf("global dof %d.\n", m_nDof_ALM);
		printf("m_nDof_Q %d.\n", m_nDof_Q); //MyPause;
		return;


		printf("alm pair %d\n", Vertex::s_vector_pair_ALM.size());

		for (MyInt c = 0; c<nCellSize; ++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			for (MyInt v = 0; v<Geometry::vertexs_per_cell; ++v)
			{
				VertexPtr curVtxPtr = curCellPtr->getVertex(v);
				if (curVtxPtr->hasALM_Mate())
				{
					//is Lagrange Multipl boundary
					curCellPtr->setLM_Boundary(v);
					if (!(curVtxPtr->isValidDof_Q()))
					{
						curVtxPtr->setDof_Q(m_nDof_Q, m_nDof_Q + 1, m_nDof_Q + 2);
						curVtxPtr->m_ALM_Ptr->setDof_Q(m_nDof_Q, m_nDof_Q + 1, m_nDof_Q + 2);
						m_nDof_Q += 3;
					}
				}
			}
		}

		printf("m_nDof_Q %d.\n", m_nDof_Q); MyPause;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::distributeDof_local()
	{
		m_vecLocalDof.resize(LocalDomainCount);
		for (MyInt v = 0; v<m_vecLocalDof.size(); ++v)
		{
			m_vecLocalDof[v] = Geometry::first_dof_idx;
		}

		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c = 0; c<nCellSize; ++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			MyInt nLocalDomainId = curCellPtr->getDomainId();
			MyInt& curDof = m_vecLocalDof[nLocalDomainId];
			for (MyInt v = 0; v<Geometry::vertexs_per_cell; ++v)
			{
				VertexPtr curVtxPtr = curCellPtr->getVertex(v);
				if (!(curVtxPtr->isValidLocalDof(nLocalDomainId)))
				{
					curVtxPtr->setLocalDof(nLocalDomainId, curDof, curDof + 1, curDof + 2);
					curDof += 3;
				}
			}
		}

		for (MyInt v = 0; v<m_vecLocalDof.size(); ++v)
		{
			printf("local [%d] : dof %d\n", v, m_vecLocalDof[v]);
		}
	}

	bool VR_Physics_FEM_Simulation_MultiDomain::isDCCondition_ALM(const MyPoint& pos)
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

	void VR_Physics_FEM_Simulation_MultiDomain::createDCBoundaryCondition_ALM()
	{
		m_vecDCBoundaryCondition_ALM.clear();
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c = 0; c<nCellSize; ++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];
			for (MyInt v = 0; v<Geometry::vertexs_per_cell; ++v)
			{
				VertexPtr curVtxPtr = curCellPtr->getVertex(v);
				const MyPoint & pos = curVtxPtr->getPos();
				if (isDCCondition_ALM(pos))
				{
					m_vecDCBoundaryCondition_ALM.push_back(curVtxPtr);
				}
			}
		}

		printf("ALM Boundary Condition %d.\n", m_vecDCBoundaryCondition_ALM.size());//MyPause;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::createGlobalMassAndStiffnessAndDampingMatrix_ALM()
	{
		const int ALM_Size = m_nDof_Q;

		global_ppcg_state.nDof_ALM = m_nDof_ALM;
		global_ppcg_state.nDof_Q = m_nDof_Q;
		global_ppcg_state.resize();

		global_alm_state.m_nDof = m_nDof_ALM + ALM_Size;
		global_alm_state.resize();

		std::map<long, std::map<long, Cell::TripletNode > >& StiffTripletNodeMap = Cell::m_TripletNode_LocalStiffness_ALM;
		std::map<long, std::map<long, Cell::TripletNode > >& MassTripletNodeMap = Cell::m_TripletNode_LocalMass_ALM;
		std::map<long, Cell::TripletNode >& RhsTripletNode = Cell::m_TripletNode_LocalRhs_ALM;
		StiffTripletNodeMap.clear();
		MassTripletNodeMap.clear();
		RhsTripletNode.clear();
#if USE_MODAL_WARP
		std::map<long, std::map<long, Cell::TripletNode > >& ModalWarpMap = Cell::m_global_state.m_TripletNode_ModalWarp;
		ModalWarpMap.clear();
#endif

		std::vector< CellPtr > & curCellVec = m_vec_cell;
		for (unsigned i = 0; i<curCellVec.size(); ++i)
		{
			CellPtr curCellPtr = curCellVec[i];
			//Q_ASSERT(curCellPtr);
			curCellPtr->initialize_ALM();
			curCellPtr->assembleSystemMatrix_ALM();
			//curCellPtr->assembleSystemMatrix();
		}

		std::vector< Eigen::Triplet<MyFloat, long> > vec_triplet;
		std::map<long, std::map<long, Cell::TripletNode > >::const_iterator itr_tri = StiffTripletNodeMap.begin();
		for (; itr_tri != StiffTripletNodeMap.end(); ++itr_tri)
		{
			const std::map<long, Cell::TripletNode >&  ref_map = itr_tri->second;
			std::map<long, Cell::TripletNode >::const_iterator itr_2 = ref_map.begin();
			for (; itr_2 != ref_map.end(); ++itr_2)
			{
				vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(itr_tri->first, itr_2->first, (itr_2->second).val));
			}
		}
		
		global_alm_state.m_global_StiffnessMatrix.setFromTriplets(vec_triplet.begin(), vec_triplet.end()); 
		printMatrix(global_alm_state.m_global_StiffnessMatrix, "m_global_StiffnessMatrix_ALM");
#if USE_PPCG
		global_ppcg_state.m_sp_ppcg_A_stiff.setFromTriplets(vec_triplet.begin(), vec_triplet.end());
#endif
		StiffTripletNodeMap.clear();
		vec_triplet.clear();

		itr_tri = MassTripletNodeMap.begin();
		for (; itr_tri != MassTripletNodeMap.end(); ++itr_tri)
		{
			const std::map<long, Cell::TripletNode >&  ref_map = itr_tri->second;
			std::map<long, Cell::TripletNode >::const_iterator itr_2 = ref_map.begin();
			for (; itr_2 != ref_map.end(); ++itr_2)
			{
				vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(itr_tri->first, itr_2->first, (itr_2->second).val));
			}
		}
		global_alm_state.m_global_MassMatrix.setFromTriplets(vec_triplet.begin(), vec_triplet.end()); 
		printMatrix(global_alm_state.m_global_MassMatrix, "m_global_MassMatrix_ALM");
#if USE_PPCG
		global_ppcg_state.m_sp_ppcg_A_mass.setFromTriplets(vec_triplet.begin(), vec_triplet.end());
#endif		
		MassTripletNodeMap.clear();
		vec_triplet.clear();

		std::map<long, Cell::TripletNode >::const_iterator itrRhs = RhsTripletNode.begin();
		global_alm_state.m_computeRhs.setZero();
		for (; itrRhs != RhsTripletNode.end(); ++itrRhs)
		{
			global_alm_state.m_computeRhs[itrRhs->first] = (itrRhs->second).val;
#if USE_PPCG
			global_ppcg_state.m_computeRhs_PPCG[itrRhs->first] = (itrRhs->second).val;
#endif	
		}
		RhsTripletNode.clear(); printVector(global_alm_state.m_computeRhs, "m_computeRhs_ALM");
	}

	void VR_Physics_FEM_Simulation_MultiDomain::createNewMarkMatrix_ALM()
	{
		global_alm_state.m_global_DampingMatrix = Material::damping_alpha * global_alm_state.m_global_MassMatrix + Material::damping_beta * global_alm_state.m_global_StiffnessMatrix;

		
		global_alm_state.m_computeMatrix = global_alm_state.m_global_StiffnessMatrix;
		global_alm_state.m_computeMatrix += m_db_NewMarkConstant[0] * global_alm_state.m_global_MassMatrix;
		global_alm_state.m_computeMatrix += m_db_NewMarkConstant[1] * global_alm_state.m_global_DampingMatrix;

#if USE_PPCG
		global_ppcg_state.m_sp_ppcg_A_damping = Material::damping_alpha * global_ppcg_state.m_sp_ppcg_A_mass + Material::damping_beta * global_ppcg_state.m_sp_ppcg_A_stiff;
		global_ppcg_state.m_sp_ppcg_A = global_ppcg_state.m_sp_ppcg_A_stiff;
		global_ppcg_state.m_sp_ppcg_A += m_db_NewMarkConstant[0] * global_ppcg_state.m_sp_ppcg_A_mass;
		global_ppcg_state.m_sp_ppcg_A += m_db_NewMarkConstant[1] * global_ppcg_state.m_sp_ppcg_A_damping;
#endif

		global_alm_state.incremental_displacement.setZero();
		global_alm_state.velocity.setZero();
		global_alm_state.acceleration.setZero();
		global_alm_state.displacement.setZero();
		global_alm_state.old_acceleration.setZero();
		global_alm_state.old_displacement.setZero();

		/*std::ofstream outfile("d:\\aa.txt");
		outfile << m_computeMatrix_ALM;
		outfile.close();*/
		return;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::makeALM_PPCG()
	{
		
		MySpMat& ref_GlobalMatrix = global_alm_state.m_computeMatrix;//m_global_StiffnessMatrix_ALM;

		MySpMat& ref_B = global_ppcg_state.m_sp_ppcg_B;
		MySpMat& ref_B1 = global_ppcg_state.m_sp_ppcg_B1;
		MySpMat& ref_B2 = global_ppcg_state.m_sp_ppcg_B2;
		MySpMat& ref_B1_Inv = global_ppcg_state.m_sp_ppcg_B1_Inv;
		MySpMat& ref_C = global_ppcg_state.m_sp_ppcg_C;
		MySpMat& ref_PreconMatrix = global_ppcg_state.m_sp_ppcg_P;
		const int n = m_nDof_ALM;
		const int m = m_nDof_Q;
		printf("PPCG n(%d) m(%d)\n", n, m);
		ref_PreconMatrix.resize(n + m, n + m); ref_PreconMatrix.setZero();
		ref_B.resize(m, n); ref_B.setZero();
		ref_B1.resize(m, m); ref_B1.setZero();
		ref_B2.resize(m, n - m); ref_B2.setZero();

		ref_C.resize(m, m); ref_C.setZero();

		std::vector< int > vecDofs, vecDofs_Q;
		const MyInt nCellSize = m_vec_cell.size();
		float n_flag = 1.f;

		MyFloat first_nonzero_diagonal_entry = 1;
		const unsigned int n_dofs = m_nDof_ALM;
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);
		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = ref_GlobalMatrix.coeff(v, v);
			//Q_ASSERT(!numbers::isZero( diagnosticValue[v]));
			ref_PreconMatrix.coeffRef(v, v) = ref_GlobalMatrix.coeff(v, v);
		}

		for (unsigned int i = 0; i<n_dofs; ++i)
		{
			if (!numbers::isZero(diagnosticValue[i]))
			{
				first_nonzero_diagonal_entry = diagnosticValue[i];
				break;
			}
		}

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
					n_flag = -1.f;

					//Q1
					faceId = 1;
					Q = curCellPtr->get_LM_ShapeFunction_Q1() * n_flag;
					//V = curCellPtr->get_LM_ShapeFunction_V1();

				}
				else
				{
					n_flag = 1.f;

					//Q2
					faceId = 0;
					Q = curCellPtr->get_LM_ShapeFunction_Q2()* n_flag;
					//V = curCellPtr->get_LM_ShapeFunction_V2();

				}

				const float scalefff = 100.f;//100000.01f;//50000.01f;
				Q *= scalefff*first_nonzero_diagonal_entry;
#if !USE_PPCG
				Q *= scalefff*first_nonzero_diagonal_entry;
				V *= -1 * 0;
#endif
				curCellPtr->get_dof_indices_Local_ALM(vecDofs);
				curCellPtr->get_dof_indices_Local_ALM_Q_New(vecDofs_Q, faceId);
				Q_ASSERT(Q.rows() == vecDofs.size());
				Q_ASSERT(Q.cols() == vecDofs_Q.size());
				Q_ASSERT(vecDofs_Q.size() == 12);

				for (int r = 0; r< vecDofs.size(); ++r)
				{
					for (int c = 0; c<vecDofs_Q.size(); ++c)
					{
						//printf("m_computeMatrix_ALM[%d][%d] = %f\n",vecDofs[r], vecDofs_Q[c] + m_nDof_ALM,Q.coeff(r,c));
						ref_GlobalMatrix.coeffRef(vecDofs[r], vecDofs_Q[c] + m_nDof_ALM) += Q.coeff(r, c);
						ref_GlobalMatrix.coeffRef(vecDofs_Q[c] + m_nDof_ALM, vecDofs[r]) += Q.coeff(r, c);

						ref_PreconMatrix.coeffRef(vecDofs[r], vecDofs_Q[c] + m_nDof_ALM) += Q.coeff(r, c);
						ref_PreconMatrix.coeffRef(vecDofs_Q[c] + m_nDof_ALM, vecDofs[r]) += Q.coeff(r, c);

						ref_B.coeffRef(vecDofs_Q[c], vecDofs[r]) += Q.coeff(r, c);

						if (vecDofs[r] < m)
						{
							ref_B1.coeffRef(vecDofs_Q[c], vecDofs[r]) += Q.coeff(r, c);
						}
						else
						{
							ref_B2.coeffRef(vecDofs_Q[c], vecDofs[r] - m) += Q.coeff(r, c);
						}
					}
				}

				/*for (int r=0;r<vecDofs_Q.size();++r)
				{
				for (int c=0;c<vecDofs_Q.size();++c)
				{
				if (!numbers::isZero(V.coeff(r,c)))
				{
				ref_GlobalMatrix.coeffRef(vecDofs_Q[r] + m_nDof_ALM,vecDofs_Q[c] + m_nDof_ALM) += V.coeff(r,c);
				}
				}
				}*/
			}
		}
		ref_GlobalMatrix.makeCompressed();
		ref_PreconMatrix.makeCompressed();
		ref_B1.makeCompressed();
		ref_B2.makeCompressed();
	}

	void VR_Physics_FEM_Simulation_MultiDomain::makeALM_PPCG_PerVertex()
	{
		MySpMat& ref_GlobalMatrix = global_alm_state.m_computeMatrix;//m_global_StiffnessMatrix_ALM;

		MySpMat& ref_B = global_ppcg_state.m_sp_ppcg_B;
		MySpMat& ref_B1 = global_ppcg_state.m_sp_ppcg_B1;
		MySpMat& ref_B2 = global_ppcg_state.m_sp_ppcg_B2;
		MySpMat& ref_B1_Inv = global_ppcg_state.m_sp_ppcg_B1_Inv;
		MySpMat& ref_C = global_ppcg_state.m_sp_ppcg_C;
		MySpMat& ref_PreconMatrix = global_ppcg_state.m_sp_ppcg_P;
		const int n = m_nDof_ALM;
		const int m = m_nDof_Q;
		printf("PPCG n(%d) m(%d)\n", n, m);
		ref_PreconMatrix.resize(n + m, n + m); ref_PreconMatrix.setZero();
		ref_B.resize(m, n); ref_B.setZero();
		ref_B1.resize(m, m); ref_B1.setZero();
		ref_B2.resize(m, n - m); ref_B2.setZero();

		ref_C.resize(m, m); ref_C.setZero();

		std::vector< int > vecDofs, vecDofs_Q;
		const MyInt nCellSize = m_vec_cell.size();
		float n_flag = 1.f;

		MyFloat first_nonzero_diagonal_entry = 1;
		const unsigned int n_dofs = m_nDof_ALM;
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);
		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = ref_GlobalMatrix.coeff(v, v);
			//Q_ASSERT(!numbers::isZero( diagnosticValue[v]));
			ref_PreconMatrix.coeffRef(v, v) = ref_GlobalMatrix.coeff(v, v);
		}

		for (unsigned int i = 0; i<n_dofs; ++i)
		{
			if (!numbers::isZero(diagnosticValue[i]))
			{
				first_nonzero_diagonal_entry = diagnosticValue[i];
				break;
			}
		}

		std::set< std::pair< MyInt, MyInt > > vertexConstraintPairSet;

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
						vertexConstraintPairSet.insert(std::make_pair(tmpVtxPtr->getId(), tmpVtxPtr->m_ALM_Ptr->getId()));
					}
				}
				else
				{
					for (int m = 0; m < (curCellPtr->m_vec_index.size()); ++m)
					{
						VertexPtr tmpVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[m]);
						vertexConstraintPairSet.insert(std::make_pair(tmpVtxPtr->getId(), tmpVtxPtr->m_ALM_Ptr->getId()));
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

			const MyVectorI& left_dofs_ALM = leftVtxPtr->getDofs_ALM();
			const MyVectorI& right_dofs_ALM = rightVtxPtr->getDofs_ALM();

			for (int i = 0; i < myDim; ++i)
			{
				ref_GlobalMatrix.coeffRef(left_dofs_Q[i] + m_nDof_ALM, left_dofs_ALM[i]) = -1.f*scalefff;
				ref_GlobalMatrix.coeffRef(left_dofs_Q[i] + m_nDof_ALM, right_dofs_ALM[i]) = 1.f*scalefff;

				ref_GlobalMatrix.coeffRef(left_dofs_ALM[i], left_dofs_Q[i] + m_nDof_ALM) = -1.f*scalefff;
				ref_GlobalMatrix.coeffRef(right_dofs_ALM[i], left_dofs_Q[i] + m_nDof_ALM) = 1.f*scalefff;

				ref_PreconMatrix.coeffRef(left_dofs_Q[i] + m_nDof_ALM, left_dofs_ALM[i]) = -1.f*scalefff;
				ref_PreconMatrix.coeffRef(left_dofs_Q[i] + m_nDof_ALM, right_dofs_ALM[i]) = 1.f*scalefff;

				ref_PreconMatrix.coeffRef(left_dofs_ALM[i], left_dofs_Q[i] + m_nDof_ALM) = -1.f*scalefff;
				ref_PreconMatrix.coeffRef(right_dofs_ALM[i], left_dofs_Q[i] + m_nDof_ALM) = 1.f*scalefff;
				

				ref_B.coeffRef(left_dofs_Q[i], left_dofs_ALM[i]) = -1.f*scalefff;
				ref_B.coeffRef(left_dofs_Q[i], right_dofs_ALM[i]) = 1.f*scalefff;

				if (left_dofs_ALM[i] < m)
				{
					ref_B1.coeffRef(left_dofs_Q[i], left_dofs_ALM[i]) = -1.f*scalefff;
				}
				else
				{
					ref_B2.coeffRef(left_dofs_Q[i], left_dofs_ALM[i] - m) = -1.f*scalefff;
				}

				if (right_dofs_ALM[i] < m)
				{
					ref_B1.coeffRef(left_dofs_Q[i], right_dofs_ALM[i]) = 1.f*scalefff;
				}
				else
				{
					ref_B2.coeffRef(left_dofs_Q[i], right_dofs_ALM[i] - m) = 1.f*scalefff;
				}
			}

		}
		ref_GlobalMatrix.makeCompressed();
		ref_PreconMatrix.makeCompressed();
		ref_B1.makeCompressed();
		ref_B2.makeCompressed();
	}

	void VR_Physics_FEM_Simulation_MultiDomain::makeALM_PerVertex()
	{
		MySpMat& ref_GlobalMatrix = global_alm_state.m_computeMatrix;//m_global_StiffnessMatrix_ALM;
		std::vector< int > vecDofs, vecDofs_Q;
		const MyInt nCellSize = m_vec_cell.size();
		float n_flag = 1.f;

		MyFloat first_nonzero_diagonal_entry = 1;
		const unsigned int n_dofs = m_nDof_ALM;
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);
		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = ref_GlobalMatrix.coeff(v, v);
			//Q_ASSERT(!numbers::isZero( diagnosticValue[v]));
		}


		for (unsigned int i = 0; i < n_dofs; ++i)
		{
			if (!numbers::isZero(diagnosticValue[i]))
			{
				first_nonzero_diagonal_entry = diagnosticValue[i];
				break;
			}
		}

		std::set< std::pair< MyInt, MyInt > > vertexConstraintPairSet;

		for (MyInt c = 0; c<nCellSize; ++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];

			if ((curCellPtr->m_vec_index.size()) > 0)
			{
				MyMatrix Q, V;
				int faceId;
				VertexPtr curVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[0]);
				Q_ASSERT((curVtxPtr->getCreateOrder()) != Invalid_Id);
				if (0 == (curVtxPtr->getCreateOrder()))
				{
					n_flag = -1.f;

					for (int m = 0; m < (curCellPtr->m_vec_index.size());++m)
					{
						VertexPtr tmpVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[m]);
						vertexConstraintPairSet.insert(std::make_pair(tmpVtxPtr->getId(), tmpVtxPtr->m_ALM_Ptr->getId()));
					}
					
				}
				else
				{
					for (int m = 0; m < (curCellPtr->m_vec_index.size()); ++m)
					{
						VertexPtr tmpVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[m]);
						vertexConstraintPairSet.insert(std::make_pair(tmpVtxPtr->getId(), tmpVtxPtr->m_ALM_Ptr->getId()));
					}
				}
			}
		}
		std::cout << "vertexConstraintPairSet.size = " << vertexConstraintPairSet.size() << std::endl;
		//MyPause;
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
			Q_ASSERT(left_dofs_Q.x() == right_dofs_Q.x() && left_dofs_Q.y() == right_dofs_Q.y() && left_dofs_Q.z() == right_dofs_Q.z() );

			const MyVectorI& left_dofs_ALM = leftVtxPtr->getDofs_ALM();
			const MyVectorI& right_dofs_ALM = rightVtxPtr->getDofs_ALM();

			for (int i = 0; i < myDim;++i)
			{
				ref_GlobalMatrix.coeffRef(left_dofs_Q[i] + m_nDof_ALM, left_dofs_ALM[i]) = -1.f*scalefff;
				ref_GlobalMatrix.coeffRef(left_dofs_Q[i] + m_nDof_ALM, right_dofs_ALM[i]) = 1.f*scalefff;

				ref_GlobalMatrix.coeffRef(left_dofs_ALM[i], left_dofs_Q[i] + m_nDof_ALM) = -1.f*scalefff;
				ref_GlobalMatrix.coeffRef(right_dofs_ALM[i], left_dofs_Q[i] + m_nDof_ALM) = 1.f*scalefff;
			}
			
		}		
		ref_GlobalMatrix.makeCompressed();

		/*std::ofstream outfile("d:\\ref_GlobalMatrix.txt");
		outfile << ref_GlobalMatrix;
		outfile.close();
		MyExit;*/
	}

	void VR_Physics_FEM_Simulation_MultiDomain::makeALM_New()
	{
		MySpMat& ref_GlobalMatrix = global_alm_state.m_computeMatrix;//m_global_StiffnessMatrix_ALM;
		std::vector< int > vecDofs, vecDofs_Q;
		const MyInt nCellSize = m_vec_cell.size();
		float n_flag = 1.f;

		MyFloat first_nonzero_diagonal_entry = 1;
		const unsigned int n_dofs = m_nDof_ALM;
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);
		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = ref_GlobalMatrix.coeff(v, v);
			//Q_ASSERT(!numbers::isZero( diagnosticValue[v]));
		}


		for (unsigned int i = 0; i<n_dofs; ++i)
		{
			if (!numbers::isZero(diagnosticValue[i]))
			{
				first_nonzero_diagonal_entry = diagnosticValue[i];
				break;
			}
		}

		for (MyInt c = 0; c<nCellSize; ++c)
		{
			CellPtr curCellPtr = m_vec_cell[c];

			if ((curCellPtr->m_vec_index.size()) > 0)
			{
				MyMatrix Q, V;
				int faceId;
				VertexPtr curVtxPtr = curCellPtr->getVertex((curCellPtr->m_vec_index)[0]);
				Q_ASSERT((curVtxPtr->getCreateOrder()) != Invalid_Id);
				if (0 == (curVtxPtr->getCreateOrder()))
				{
					n_flag = -1.f;

					//Q1
					faceId = 1;
					Q = curCellPtr->get_LM_ShapeFunction_Q1() * n_flag;
					V = curCellPtr->get_LM_ShapeFunction_V1();

				}
				else
				{
					n_flag = 1.f;

					//Q2
					faceId = 0;
					Q = curCellPtr->get_LM_ShapeFunction_Q2()* n_flag;
					V = curCellPtr->get_LM_ShapeFunction_V2();

				}

				const float scalefff = 50000.f;//100000.01f;//50000.01f;

#if USE_PPCG
				Q *= scalefff*first_nonzero_diagonal_entry;
				V *= -1 * 0;
#endif

				//V *= scalefff*first_nonzero_diagonal_entry*1/100;
				/*MyMatrix LM_ShapeFunction = curCellPtr->get_LM_ShapeFunction();

				MyMatrix V = LM_ShapeFunction.transpose() * LM_ShapeFunction * (0.000244140625f) * (-1.f) * (1.01f) * first_nonzero_diagonal_entry * scalefff;

				MyMatrix Q = ((curCellPtr->s_shapeFunction).transpose()) * LM_ShapeFunction * (n_flag) * (0.000244140625f)* first_nonzero_diagonal_entry * scalefff;*/

				curCellPtr->get_dof_indices_Local_ALM(vecDofs);
				curCellPtr->get_dof_indices_Local_ALM_Q_New(vecDofs_Q, faceId);
				Q_ASSERT(Q.rows() == vecDofs.size());
				Q_ASSERT(Q.cols() == vecDofs_Q.size());
				Q_ASSERT(vecDofs_Q.size() == 12);

				//m_global_DampingMatrix_ALM = Material::damping_alpha * m_global_MassMatrix_ALM + Material::damping_beta * m_global_StiffnessMatrix_ALM;


				for (int r = 0; r< vecDofs.size(); ++r)
				{
					for (int c = 0; c<vecDofs_Q.size(); ++c)
					{
						//printf("m_computeMatrix_ALM[%d][%d] = %f\n",vecDofs[r], vecDofs_Q[c] + m_nDof_ALM,Q.coeff(r,c));
						ref_GlobalMatrix.coeffRef(vecDofs[r], vecDofs_Q[c] + m_nDof_ALM) += Q.coeff(r, c);
						ref_GlobalMatrix.coeffRef(vecDofs_Q[c] + m_nDof_ALM, vecDofs[r]) += Q.coeff(r, c);

						/*m_global_MassMatrix_ALM.coeffRef( vecDofs[r], vecDofs_Q[c] + m_nDof_ALM ) += Q.coeff(r,c);
						m_global_MassMatrix_ALM.coeffRef( vecDofs_Q[c] + m_nDof_ALM, vecDofs[r] ) += Q.coeff(r,c);

						m_global_StiffnessMatrix_ALM.coeffRef( vecDofs[r], vecDofs_Q[c] + m_nDof_ALM ) += Q.coeff(r,c);
						m_global_StiffnessMatrix_ALM.coeffRef( vecDofs_Q[c] + m_nDof_ALM, vecDofs[r] ) += Q.coeff(r,c);*/


					}
				}

				for (int r = 0; r<vecDofs_Q.size(); ++r)
				{
					for (int c = 0; c<vecDofs_Q.size(); ++c)
					{
						if (!numbers::isZero(V.coeff(r, c)))
						{
							ref_GlobalMatrix.coeffRef(vecDofs_Q[r] + m_nDof_ALM, vecDofs_Q[c] + m_nDof_ALM) += V.coeff(r, c);
						}
					}
				}
			}
		}
		ref_GlobalMatrix.makeCompressed();
	}

	void VR_Physics_FEM_Simulation_MultiDomain::simulation_PPCG()
	{
		update_rhs_PPCG();
		apply_boundary_values_PPCG();
		solve_linear_problem_PPCG();
		update_u_v_a_PPCG();
		//return;
		compute_Local_R_ALM(Global_PPCG_State());
		update_displacement_ModalWrap_ALM(Global_PPCG_State());
	}

	void VR_Physics_FEM_Simulation_MultiDomain::update_rhs_PPCG()
	{
		
		global_ppcg_state.R_rhs_PPCG.setZero();
		global_ppcg_state.mass_rhs_PPCG.setZero();
		global_ppcg_state.damping_rhs_PPCG.setZero();

		//incremental_displacement,velocity,acceleration
		global_ppcg_state.mass_rhs_PPCG += m_db_NewMarkConstant[0] * global_ppcg_state.displacement_PPCG;
		global_ppcg_state.mass_rhs_PPCG += m_db_NewMarkConstant[2] * global_ppcg_state.velocity_PPCG;
		global_ppcg_state.mass_rhs_PPCG += m_db_NewMarkConstant[3] * global_ppcg_state.acceleration_PPCG;

		global_ppcg_state.damping_rhs_PPCG += m_db_NewMarkConstant[1] * global_ppcg_state.displacement_PPCG;
		global_ppcg_state.damping_rhs_PPCG += m_db_NewMarkConstant[4] * global_ppcg_state.velocity_PPCG;
		global_ppcg_state.damping_rhs_PPCG += m_db_NewMarkConstant[5] * global_ppcg_state.acceleration_PPCG;

		global_ppcg_state.R_rhs_PPCG = global_ppcg_state.m_computeRhs_PPCG;

		global_ppcg_state.R_rhs_PPCG += global_ppcg_state.m_sp_ppcg_A_mass * global_ppcg_state.mass_rhs_PPCG;
		global_ppcg_state.R_rhs_PPCG += global_ppcg_state.m_sp_ppcg_A_damping * global_ppcg_state.damping_rhs_PPCG;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::apply_boundary_values_PPCG()
	{
		std::vector< VertexPtr >& vecBoundaryVtx = m_vecDCBoundaryCondition_ALM;
		MySpMat&  computeMatrix = global_ppcg_state.m_sp_ppcg_A;
		MyVector& curRhs = global_ppcg_state.R_rhs_PPCG;
		MyVector& curDisplacement = global_ppcg_state.incremental_displacement_PPCG;
		printf("apply_boundary_values begin \n");
		if (vecBoundaryVtx.size() == 0)
			return;


		const unsigned int n_dofs = computeMatrix.rows();
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);

		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = computeMatrix.coeff(v, v);
			//Q_ASSERT(!numbers::isZero( diagnosticValue[v]));
		}

		MyFloat first_nonzero_diagonal_entry = 1;
		for (unsigned int i = 0; i<n_dofs; ++i)
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
			const MyVectorI& Dofs = curVtxPtr->getDofs_ALM();
			for (unsigned c = 0; c<MyDIM; ++c)
			{
				const unsigned int dof_number = Dofs[c];
				setMatrixRowZeroWithoutDiag(computeMatrix, dof_number);

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

	void VR_Physics_FEM_Simulation_MultiDomain::solve_linear_problem_PPCG()
	{
		
#define CGSolverTolerance (1e-6f)

		const int n = m_nDof_ALM;
		const int m = m_nDof_Q;
		MyVector d(m);
		MySpMat C(m, m);
		C.setZero();
		d.setZero();
		static float r_nrm2;
		r_nrm2 = global_ppcg_state.R_rhs_PPCG.norm();
		PCG_RU::MyMonitor<MyFloat> monitor(r_nrm2, n + m, CGSolverTolerance);

		global_ppcg_state.incremental_displacement_PPCG = PCG_RU::PCG_Residual_Update_Sparse_Schilders(global_ppcg_state.m_sp_ppcg_A, global_ppcg_state.m_sp_ppcg_B, global_ppcg_state.m_sp_ppcg_B1, global_ppcg_state.m_sp_ppcg_B2, global_ppcg_state.m_sp_ppcg_C, global_ppcg_state.R_rhs_PPCG, d, monitor);
		//incremental_displacement_PPCG = PCG_RU::PCG_Residual_Update_Sparse(m_sp_ppcg_A,m_sp_ppcg_B,C,R_rhs_PPCG,d,m_sp_ppcg_P,monitor);

		/*for (int i = 0; i<Vertex::s_vector_pair_ALM.size(); ++i)
		{
			VertexPtr v0 = Vertex::s_vector_pair_ALM[i].first;
			VertexPtr v1 = Vertex::s_vector_pair_ALM[i].second;
			MyVectorI dofs_0 = v0->getDofs_ALM();
			MyVectorI dofs_1 = v1->getDofs_ALM();

			for (int d = 0; d<myDim; ++d)
			{
				float vals = (global_ppcg_state.incremental_displacement_PPCG[dofs_0[d]] + global_ppcg_state.incremental_displacement_PPCG[dofs_1[d]]) / 2;
				global_ppcg_state.incremental_displacement_PPCG[dofs_0[d]] = vals;
				global_ppcg_state.incremental_displacement_PPCG[dofs_1[d]] = vals;
			}

		}*/
	}

	void VR_Physics_FEM_Simulation_MultiDomain::update_u_v_a_PPCG()
	{
		const MyVector& solu = global_ppcg_state.incremental_displacement_PPCG;
		MyVector& disp_vec = global_ppcg_state.displacement_PPCG;
		MyVector& vel_vec = global_ppcg_state.velocity_PPCG;
		MyVector& acc_vec = global_ppcg_state.acceleration_PPCG;
		MyVector& old_acc = global_ppcg_state.old_acceleration_PPCG;
		MyVector& old_solu = global_ppcg_state.old_displacement_PPCG;

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

	void VR_Physics_FEM_Simulation_MultiDomain::simulation_ALM()
	{
		update_rhs_ALM();

		apply_boundary_values_ALM();
		//printf("solve_linear_problem\n");
		solve_linear_problem_ALM();
		//printf("update_u_v_a\n");
		update_u_v_a_ALM();
	}

	void VR_Physics_FEM_Simulation_MultiDomain::apply_boundary_values_ALM()
	{
		std::vector< VertexPtr >& vecBoundaryVtx = m_vecDCBoundaryCondition_ALM;
		MySpMat&  computeMatrix = global_alm_state.m_computeMatrix;
		MyVector& curRhs = global_alm_state.R_rhs;
		MyVector& curDisplacement = global_alm_state.incremental_displacement;
		printf("apply_boundary_values begin \n");
		if (vecBoundaryVtx.size() == 0)
			return;


		const unsigned int n_dofs = computeMatrix.rows();
		std::vector<MyFloat> diagnosticValue(n_dofs, 0.0);

		for (int v = 0; v < n_dofs; ++v)
		{
			diagnosticValue[v] = computeMatrix.coeff(v, v);
			//Q_ASSERT(!numbers::isZero( diagnosticValue[v]));
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
			const MyVectorI& Dofs = curVtxPtr->getDofs_ALM();
			for (unsigned c = 0; c < myDim; ++c)
			{
				const unsigned int dof_number = Dofs[c];
				setMatrixRowZeroWithoutDiag(computeMatrix, dof_number);

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

	void VR_Physics_FEM_Simulation_MultiDomain::update_rhs_ALM()
	{
		global_alm_state.R_rhs.setZero();
		global_alm_state.mass_rhs.setZero();
		global_alm_state.damping_rhs.setZero();

		//incremental_displacement,velocity,acceleration
		global_alm_state.mass_rhs += m_db_NewMarkConstant[0] * global_alm_state.displacement;
		global_alm_state.mass_rhs += m_db_NewMarkConstant[2] * global_alm_state.velocity;
		global_alm_state.mass_rhs += m_db_NewMarkConstant[3] * global_alm_state.acceleration;

		global_alm_state.damping_rhs += m_db_NewMarkConstant[1] * global_alm_state.displacement;
		global_alm_state.damping_rhs += m_db_NewMarkConstant[4] * global_alm_state.velocity;
		global_alm_state.damping_rhs += m_db_NewMarkConstant[5] * global_alm_state.acceleration;

		global_alm_state.R_rhs = global_alm_state.m_computeRhs;

		global_alm_state.R_rhs += global_alm_state.m_global_MassMatrix * global_alm_state.mass_rhs;
		global_alm_state.R_rhs += global_alm_state.m_global_DampingMatrix * global_alm_state.damping_rhs;


	}

	void VR_Physics_FEM_Simulation_MultiDomain::update_u_v_a_ALM()
	{
		const MyVector& solu = global_alm_state.incremental_displacement;
		MyVector& disp_vec = global_alm_state.displacement;
		MyVector& vel_vec = global_alm_state.velocity;
		MyVector& acc_vec = global_alm_state.acceleration;
		MyVector& old_acc = global_alm_state.old_acceleration;
		MyVector& old_solu = global_alm_state.old_displacement;

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

	void VR_Physics_FEM_Simulation_MultiDomain::solve_linear_problem_ALM()
	{
		using namespace Eigen;
#if 1
		static bool bFirst = true;
		static ColPivHouseholderQR< MyMatrix > ALM_QR;
		if (bFirst)
		{
			bFirst = false;
			MyMatrix tmp(global_alm_state.m_computeMatrix.rows(), global_alm_state.m_computeMatrix.cols());
			tmp.setZero();

			for (int r = 0; r < global_alm_state.m_computeMatrix.rows(); ++r)
			{
				for (int c = 0; c < global_alm_state.m_computeMatrix.cols(); ++c)
				{
					tmp.coeffRef(r, c) = global_alm_state.m_computeMatrix.coeff(r, c);
				}
			}
			ALM_QR = tmp.colPivHouseholderQr();
		}

		global_alm_state.incremental_displacement = ALM_QR.solve(global_alm_state.R_rhs);
		return ;
		//incremental_displacement_ALM = tmp.ldlt().solve(R_rhs_ALM);
		for (int i = 0; i < Vertex::s_vector_pair_ALM.size(); ++i)
		{
			VertexPtr v0 = Vertex::s_vector_pair_ALM[i].first;
			VertexPtr v1 = Vertex::s_vector_pair_ALM[i].second;
			MyVectorI dofs_0 = v0->getDofs_ALM();
			MyVectorI dofs_1 = v1->getDofs_ALM();

			for (int d = 0; d < myDim; ++d)
			{
				float vals = (global_alm_state.incremental_displacement[dofs_0[d]] + global_alm_state.incremental_displacement[dofs_1[d]]) / 2;
				global_alm_state.incremental_displacement[dofs_0[d]] = vals;
				global_alm_state.incremental_displacement[dofs_1[d]] = vals;
			}

		}
		//MyPause;
		return;

#else
		global_alm_state.incremental_displacement = MyGaussElimination::GaussElimination(global_alm_state.m_computeMatrix, global_alm_state.R_rhs);
#endif	
	}

#if USE_MAKE_CELL_SURFACE
	void VR_Physics_FEM_Simulation_MultiDomain::creat_Outter_Skin(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs, YC::MyIntVector& vecF_ColorId)
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
				matV_dofs.row(vecVtxFaceIndex[v]) = curVtxPtr->getDofs_ALM();
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

	void VR_Physics_FEM_Simulation_MultiDomain::getSkinDisplacement(YC::MyVector& displacement, YC::MyMatrix& matV, YC::MyIntMatrix& matV_dofs, YC::MyMatrix& matU)
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

#if USE_MODAL_WARP
	

	void VR_Physics_FEM_Simulation_MultiDomain::update_displacement_ModalWrap_ALM(Physic_PPCG_State& ppcgState)
	{
		ppcgState.incremental_displacement_PPCG = ppcgState.m_R_hat * ppcgState.incremental_displacement_PPCG;
		//std::cout << globalState.incremental_displacement.transpose() << std::endl; MyPause;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::compute_Local_R_ALM(Physic_PPCG_State& ppcgState)
	{
		compute_ModalWrap_Rotation_ALM(ppcgState.incremental_displacement_PPCG, ppcgState.m_R_hat);
	}

	void VR_Physics_FEM_Simulation_MultiDomain::compute_ModalWrap_Rotation_ALM(const MyVector& globalDisplacement, MySpMat& modalWrap_R_hat)
	{

		MyVector w, translation;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		for (int c = 0; c < m_vec_cell.size(); ++c)
		{
			m_vec_cell[c]->TestModalWrapMatrix_ALM(globalDisplacement, w, translation, cellDisplace); 
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
		modalWrap_R_hat.resize(globalDisplacement.size(), globalDisplacement.size()); modalWrap_R_hat.setZero();
		MyMatrix_3X3 tmpR;
		//m_vec_VtxLocalRotaM.clear();
		for (int v = 0; v < Vertex::getVertexSize(); ++v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec; cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getDofs_ALM();

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
			for (int r = 0; r < 3; ++r)
			{
				for (int c = 0; c < 3; ++c)
				{
					vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(dofs[r], dofs[c], tmpR.coeff(r, c)));
				}
			}
		}

		modalWrap_R_hat.setFromTriplets(vec_triplet.begin(), vec_triplet.end());;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::solve_linear_Subspace_problem(Vibrate_State& curState)
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

	void VR_Physics_FEM_Simulation_MultiDomain::computeReferenceDisplace_ModalWarp(Physic_State& curRefState, Vibrate_State&  curState)
	{
		curRefState.incremental_displacement = curState.m_BasisU_hat * curState.MR_incremental_displacement;
	}

	Axis::Quaternion VR_Physics_FEM_Simulation_MultiDomain::covert_matrix2Quaternion(const MyDenseMatrix& nodeRotationMatrix)
	{
		Axis::Quaternion quater;
		Matrix3<float> mat;
		mat.matrix[0].elems[0] = nodeRotationMatrix(0, 0);
		mat.matrix[0].elems[1] = nodeRotationMatrix(0, 1);
		mat.matrix[0].elems[2] = nodeRotationMatrix(0, 2);

		mat.matrix[1].elems[0] = nodeRotationMatrix(1, 0);
		mat.matrix[1].elems[1] = nodeRotationMatrix(1, 1);
		mat.matrix[1].elems[2] = nodeRotationMatrix(1, 2);

		mat.matrix[2].elems[0] = nodeRotationMatrix(2, 0);
		mat.matrix[2].elems[1] = nodeRotationMatrix(2, 1);
		mat.matrix[2].elems[2] = nodeRotationMatrix(2, 2);
		quater.fromMatrix(mat);
		return quater;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::compute_modalWarp_R(Vibrate_State& subspaceState)
	{
		MyVector W_global = subspaceState.m_ModalWarp_Basis * subspaceState.MR_incremental_displacement;
		//W_global *= -1.f;
		const int nVtxSize = subspaceState.m_globalDofs / 3;
		MyDenseMatrix X(3, 3); X.setZero();
		MyDenseMatrix tmpR;
		MyFloat W_norm;
		subspaceState.m_R_hat.setZero();
		std::vector< Eigen::Triplet<MyFloat, long> > vec_triplet;
		m_vec_frame_quater.clear();
		for (int m = 0; m<nVtxSize; ++m)
		{
			MyVectorI dofs = Vertex::getVertex(m)->getDofs_ALM();

			MyDenseVector W = MyDenseVector(W_global[dofs[0]], W_global[dofs[1]], W_global[dofs[2]]);
			MyDenseVector W_hat = W;
			W_hat.normalize();

			X.setZero();
			X.coeffRef(0, 1) = -1.f*W_hat[2];
			X.coeffRef(0, 2) = W_hat[1];

			X.coeffRef(1, 0) = W_hat[2];
			X.coeffRef(1, 2) = -1.f*W_hat[0];

			X.coeffRef(2, 0) = -1.f * W_hat[1];
			X.coeffRef(2, 1) = W_hat[0];

			W_norm = W.norm();

			if (numbers::isZero(W_norm))
			{
				tmpR = MyDenseMatrix::Identity(3, 3);
			}
			else
			{
				tmpR = MyDenseMatrix::Identity(3, 3) + (X * (1 - cos(W_norm)) / W_norm) + X * X * (1 - sin(W_norm) / W_norm);
			}

			m_vec_frame_quater.push_back(covert_matrix2Quaternion(tmpR));

			for (int r = 0; r<3; ++r)
			{
				for (int c = 0; c<3; ++c)
				{
					vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(dofs[r], dofs[c], tmpR.coeff(r, c)));
					//vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(i*3+r,i*3+c,tmpR.coeff(r,c) )); std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
				}
			}
		}

		subspaceState.m_R_hat.setFromTriplets(vec_triplet.begin(), vec_triplet.end());
	}

	void VR_Physics_FEM_Simulation_MultiDomain::compute_R_Basis(Vibrate_State& subspaceState)
	{
		subspaceState.m_BasisU_hat = subspaceState.m_R_hat * subspaceState.m_BasisU;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::compute_rhs_ModalWrap(const Physic_State& globalState, Vibrate_State& subspaceState)
	{
		subspaceState.R_MR_rhs = subspaceState.m_BasisU.transpose() * globalState.R_rhs;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::compute_Local_R(Vibrate_State& subspaceState)
	{
		global_alm_state.incremental_displacement = subspaceState.m_BasisU * subspaceState.MR_incremental_displacement;

		//std::cout << m_global_State.incremental_displacement.transpose() << std::endl;MyPause;

		MyVector w, translation;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		for (int c = 0; c<m_vec_cell.size(); ++c)
		{
			m_vec_cell[c]->TestModalWrapMatrix(global_alm_state.incremental_displacement, w, translation, cellDisplace);
			m_vec_cell[c]->m_vec_LocalQuat.resize(Geometry::vertexs_per_cell);
			MyDenseVector ww; ww.setZero();
			MyDenseVector W, W_hat;
			float w_norm;
			for (int m = 0; m<8; ++m)
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


		std::vector< Eigen::Triplet<MyFloat, long> > vec_triplet; subspaceState.m_R_hat.setZero();
		MyMatrix_3X3 tmpR;
		m_vec_VtxLocalRotaM.clear();
		for (int v = 0; v<Vertex::getVertexSize(); ++v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec; cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getDofs_ALM();

			const int nSize = refVec.size();
			const int addAmount = nSize * Geometry::vertexs_per_cell;

			Axis::Quaternion firstQuat = refVec[0]->m_vec_LocalQuat[0];
			for (int c = 0; c<refVec.size(); ++c)
			{
				for (int i = 0; i<Geometry::vertexs_per_cell; ++i)
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
			m_vec_VtxLocalRotaM.push_back(tmpR);
			//curQuat.toMatrix(tmpR);
			for (int r = 0; r<3; ++r)
			{
				for (int c = 0; c<3; ++c)
				{
					vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(dofs[r], dofs[c], tmpR.coeff(r, c)));


					//vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(i*3+r,i*3+c,tmpR.coeff(r,c) )); 
				}
			}
		}
		//subspaceState.m_R_hat.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		subspaceState.m_R_hat.setFromTriplets(vec_triplet.begin(), vec_triplet.end());;


		subspaceState.m_BasisU_hat = subspaceState.m_R_hat * subspaceState.m_BasisU;
	}

	void VR_Physics_FEM_Simulation_MultiDomain::compute_CellRotation(Vibrate_State& subspaceState)
	{
		global_alm_state.incremental_displacement = subspaceState.m_BasisU * subspaceState.MR_incremental_displacement;
		m_testCellRotation.clear();

		MyMatrix_3X3 RMatrix;
		MyDenseVector translationVec;
		MyVector translation;
		MyVector w;
		MyVector cellDisplace(Geometry::dofs_per_cell);
		MyVector cellTranslation;
		for (int c = 0; c<m_vec_cell.size(); ++c)
		{
			m_vec_cell[c]->TestCellRotationMatrix(global_alm_state.incremental_displacement, RMatrix, translationVec);
			m_testCellRotation.push_back(std::make_pair(translationVec, covert_matrix2Quaternion(RMatrix)));
			//continue;
		}

		std::vector< Eigen::Triplet<MyFloat, long> > vec_triplet;
		vecShareCellQuaternion.clear();
		m_vec_CelLocalRotaM.clear();
		MyMatrix_3X3 tmpR;
		for (int v = 0; v<Vertex::getVertexSize(); ++v)
		{
			Axis::Quaternion curQuat;
			Eigen::Vector4f cumulateVec; cumulateVec.setZero();
			std::vector< CellPtr >& refVec = Vertex::getVertex(v)->m_vec_ShareCell;
			MyVectorI dofs = Vertex::getVertex(v)->getDofs_ALM();

			const int nSize = refVec.size();
			if (1 == nSize)
			{
				int id = refVec[0]->getID();
				float * quatPtr = m_testCellRotation[id].second.ptr();
				curQuat = NormalizeQuaternion(quatPtr[0], quatPtr[1], quatPtr[2], quatPtr[3]);
			}
			else
			{
				int id = refVec[0]->getID();
				Axis::Quaternion firstQuat = m_testCellRotation[id].second;
				for (int c = 0; c<refVec.size(); ++c)
				{
					int id = refVec[c]->getID();
					Axis::Quaternion newRotaQuat = m_testCellRotation[id].second;
					curQuat = AverageQuaternion(cumulateVec, newRotaQuat, firstQuat, nSize);
				}
			}


			//curQuat = InverseSignQuaternion(curQuat);
			vecShareCellQuaternion.push_back(curQuat);

			curQuat.toMatrix(tmpR);
			m_vec_CelLocalRotaM.push_back(tmpR);
			for (int r = 0; r<3; ++r)
			{
				for (int c = 0; c<3; ++c)
				{
					vec_triplet.push_back(Eigen::Triplet<MyFloat, long>(dofs[r], dofs[c], tmpR.coeff(r, c)));
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

	void VR_Physics_FEM_Simulation_MultiDomain::printRotationFrame()
	{
		return;
	}
#endif//USE_MODAL_WARP

#endif //#if USE_MultiDomain
}//namespace YC