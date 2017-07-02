#include "stdafx.h"
#include "ElasticSimulation_Couple.h"
#include "MeshParser_Obj/objParser.h"
#include "SOIL.h"
#include "CG/solvercontrol.h"
#include "CG/solvercg.h"
#include "CG/preconditionssor.h"
extern float element_steak[3136][4];
namespace VR_FEM
{
	ElasticSimulation_Couple::ElasticSimulation_Couple()
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


		printf("FEM (%d), EFG (%d), Couple(%d)\n",Cell::s_nFEM_Cell_Count,Cell::s_nEFG_Cell_Count,Cell::s_nCOUPLE_Cell_Count);
		//steak : FEM (1712), EFG (680), Couple(744)
		return true;
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

	void ElasticSimulation_Couple::simulation()
	{
		//MyInt id = 0;
		for (unsigned id=0;id < 2;++id)
		{
			//printf("update_rhs\n");
			update_rhs(id);

			//apply_boundary_values(id);
			//printf("solve_linear_problem\n");
			solve_linear_problem (id);
			//printf("update_u_v_a\n");
			update_u_v_a(id);
		}
	}

	void ElasticSimulation_Couple::update_rhs(MyInt id)
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

	void ElasticSimulation_Couple::solve_linear_problem(MyInt id)
	{
		SolverControl           solver_control (m_vecLocalDof[id],  1e-3*numbers::l2_norm(R_rhs[id]));
		SolverCG              cg (solver_control);

		PreconditionSSOR preconditioner;
		preconditioner.initialize(m_computeMatrix[id], 1.2);

		cg.solve (m_computeMatrix[id], incremental_displacement[id], R_rhs[id],	preconditioner);

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
			mass_rhs[DomainId].resize(nDof);
			damping_rhs[DomainId].resize(nDof);
			displacement[DomainId].resize(nDof);
			velocity[DomainId].resize(nDof);
			acceleration[DomainId].resize(nDof);
			old_acceleration[DomainId].resize(nDof);
			old_displacement[DomainId].resize(nDof);
			incremental_displacement[DomainId].resize(nDof);

			R_rhs[DomainId].setZero();
			mass_rhs[DomainId].setZero();
			damping_rhs[DomainId].setZero();
			displacement[DomainId].setZero();
			velocity[DomainId].setZero();
			acceleration[DomainId].setZero();
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

		m_global_DampingMatrix[DomainId] = Material::damping_alpha * m_global_MassMatrix[DomainId] + Material::damping_beta * m_global_StiffnessMatrix[DomainId];
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
			if (nLocalDomainId != Invalid_Id)
			{
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
		glPointSize(5.f);
		glBegin(GL_POINTS);
		const MyInt nCellSize = m_vec_cell.size();
		for (MyInt c=0;c<nCellSize;++c)
		{
			const MyInt id = m_vec_cell[c]->getDomainId();
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
				/*const MyPoint& center = m_vec_cell[c]->getCenterPoint();
				const MyFloat r = m_vec_cell[c]->getRadius()*2; 
				MyInt type = m_vec_cell[c]->getDomainId();

				glPushMatrix();
				glColor3f(Colors::colorTemplage[type][0],Colors::colorTemplage[type][1],Colors::colorTemplage[type][2]);
				glTranslatef(center[0],center[1],center[2]);
				glutSolidCube(r);
				glPopMatrix();*/
			}
		}
		glEnd();
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