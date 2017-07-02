#include "stdafx.h"
#include "Cell.h"
#include "globalrowsfromlocal.h"
#include "BlasOperator.h"
#include <Eigen/Cholesky>
#include <iterator>     // std::back_inserter
#include <vector>       // std::vector
#include <algorithm>    // std::copy
//#include <Eigen/Dense>
#include "constant_numbers.h"

namespace VR_FEM
{
	std::vector< CellPtr > Cell::s_Cell_Cache;
	MyPoint Cell::GaussVertex[Geometry::vertexs_per_cell];
	MyFloat Cell::GaussVertexWeight[Geometry::vertexs_per_cell];
	MyMatrix Cell::MaterialMatrix_6_5[Cell::LocalDomainCount];
	bool Cell::bMaterialMatrixInitial[Cell::LocalDomainCount]={false,false/*,false,false*/};
	std::map<long,std::map<long,TripletNode > > Cell::m_TripletNode_Mass;
	std::map<long,std::map<long,TripletNode > > Cell::m_TripletNode_Stiffness;
	std::map< unsigned,bool > Cell::s_map_InfluncePointSize;

	std::map<long,std::map<long,TripletNode > > Cell::m_TripletNode_LocalMass[LocalDomainCount];
	std::map<long,std::map<long,TripletNode > > Cell::m_TripletNode_LocalStiffness[LocalDomainCount];
	std::map<long,TripletNode >				   Cell::m_TripletNode_LocalRhs[LocalDomainCount];

	std::map<long,std::map<long,TripletNode > > Cell::m_TripletNode_LocalMass_EFG[CoupleDomainCount];
	std::map<long,std::map<long,TripletNode > > Cell::m_TripletNode_LocalStiffness_EFG[CoupleDomainCount];
	std::map<long,TripletNode >				    Cell::m_TripletNode_LocalRhs_EFG[CoupleDomainCount];

	//MyDenseVector Cell::externalForce(Material::Density * Material::GravityFactor*50,0.f,0.f);//X
	//MyDenseVector Cell::externalForce(0,-1*Material::Density * Material::GravityFactor*1000,0.f);//Y
	MyDenseVector Cell::externalForce(0,Material::Density * Material::GravityFactor*8,0.f);//Z
	std::map<long,TripletNode >					Cell::m_TripletNode_Rhs;

	int Cell::s_vertexOutgoingEdgePoints[8][3] = {{1,2,4},{0,3,5},{0,3,6},{1,2,7},{0,5,6},{1,4,7},{2,4,7},{3,5,6}};

	std::vector< tuple_matrix > Cell::vec_cell_stiffness_matrix;
	std::vector< tuple_matrix > Cell::vec_cell_mass_matrix;
	std::vector< tuple_vector > Cell::vec_cell_rhs_matrix;
	std::vector< FEMShapeValue > Cell::vec_FEM_ShapeValue;

	Cell::Cell(MyPoint center, MyFloat radius, VertexPtr vertexes[])
		:m_center(center),m_radius(radius),m_CellType(INVALIDTYPE),m_nID(Invalid_Id),m_nRhsIdx(Invalid_Id),m_nMassMatrixIdx(Invalid_Id),m_nStiffnessMatrixIdx(Invalid_Id)
	{
		m_CellType = FEM;
		m_mapSupportPtsIdMap.clear();
		for (unsigned i=0; i < Geometry::vertexs_per_cell; ++i)
		{
			m_elem_vertex[i] = vertexes[i];
			m_mapSupportPtsIdMap[m_elem_vertex[i]->getId()] = true;
		}
		Q_ASSERT(Geometry::vertexs_per_cell == m_mapSupportPtsIdMap.size());

		shapeFunctionValue_8_8.resize(Geometry::vertexs_per_cell,Geometry::vertexs_per_cell);
		memset(&JxW_values[0],'\0',sizeof(JxW_values));
		memset(&GaussVertexWeight[0],'\0',sizeof(GaussVertexWeight));
		for (int v=0; v<Geometry::vertexs_per_cell;++v)
		{
			GaussVertex[v].setZero();
			contravariant[v].setZero();

			for (int w=0;w<Geometry::vertexs_per_cell;++w)
			{
				shapeDerivativeValue_8_8_3[v][w].setZero();
				shapeSecondDerivativeValue_8_8_3_3[v][w].setZero();
			}
		}
		shapeFunctionValue_8_8.setZero();

		//*************   EFG ********************
		for (unsigned i=0; i < sizeof(gaussPointInfluncePointCount_8)/sizeof(gaussPointInfluncePointCount_8[0]);++i)
		{
			gaussPointInfluncePointCount_8[i]=0;
		}
		//*************   EFG ********************
		incremental_displacement.resize(Geometry::dofs_per_cell);
		incremental_displacement.setZero();
	}

	CellPtr Cell::makeCell(MyPoint point, MyFloat radius)
	{
		//printf("{%f,%f,%f} radius(%f)\n",point[0],point[1],point[2],radius);
		std::vector< CellPtr >::reverse_iterator itr = std::find_if(s_Cell_Cache.rbegin(),s_Cell_Cache.rend(),CellCompare(point[0],point[1],point[2]));
		if ( s_Cell_Cache.rend() == itr )
		{
			//no find
			VertexPtr vertexes[Geometry::vertexs_per_cell];
			Vertex::makeCellVertex(point,radius,vertexes);

			/*
			vertex(0.000000,0.000000,0.000000)
			vertex(0.015625,0.000000,0.000000)
			vertex(0.000000,0.015625,0.000000)
			vertex(0.015625,0.015625,0.000000)
			vertex(0.000000,0.000000,0.015625)
			vertex(0.015625,0.000000,0.015625)
			vertex(0.000000,0.015625,0.015625)
			vertex(0.015625,0.015625,0.015625)
			*/

			for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
			{
				MyDenseVector pos = vertexes[v]->getPos();
				
				if ( !(vertexes[v]->isValidOutgoingEdge()) )
				{
					std::vector< VertexPtr > params;
					params.push_back(vertexes[s_vertexOutgoingEdgePoints[v][0]]);
					params.push_back(vertexes[s_vertexOutgoingEdgePoints[v][1]]);
					params.push_back(vertexes[s_vertexOutgoingEdgePoints[v][2]]);
					vertexes[v]->initializeRotationMatrixInRest(params);
				}
			}

			s_Cell_Cache.push_back( CellPtr(new Cell(point,radius,vertexes)) );
			s_Cell_Cache[s_Cell_Cache.size()-1]->setId(s_Cell_Cache.size()-1);
			return s_Cell_Cache[s_Cell_Cache.size()-1];
		}
		else
		{
			//find it
			//Q_ASSERT(false);
			return (*itr);
		}
	}

	void Cell::get_dof_indices(std::vector<int> &vecDofs)
	{
		vecDofs.clear();
		if (FEM == m_CellType)
		{
			for (int i=0;i<Geometry::vertexs_per_cell;++i)
			{
				MyVectorI& ref_Dof = m_elem_vertex[i]->getDofs();
				vecDofs.push_back(ref_Dof[0]);
				vecDofs.push_back(ref_Dof[1]);
				vecDofs.push_back(ref_Dof[2]);
			}
		}
	}

	void Cell::get_dof_indices_Local_FEM(MyInt nLocalDomainId,std::vector<int> &vecDofs)
	{
		vecDofs.clear();
		if (FEM == m_CellType)
		{
			for (int i=0;i<Geometry::vertexs_per_cell;++i)
			{
				MyVectorI& ref_Dof = m_elem_vertex[i]->getLocalDof(m_nDomainId);
				vecDofs.push_back(ref_Dof[0]);
				vecDofs.push_back(ref_Dof[1]);
				vecDofs.push_back(ref_Dof[2]);
			}
		}
		else
		{
			printf("FEM != m_CellType\n");
			MyPause;
		}
	}

	void Cell::get_dof_indices_Couple_EFG(unsigned gaussIdx,std::vector<int> &vecDofs)
	{
		vecDofs.clear();
		const MyInt nCoupleDomainId = getCoupleDomainId();
		const unsigned nInflPts = InflPts_8_27[gaussIdx].size();
		for (unsigned i = 0 ; i < nInflPts;++i)
		{
			VertexPtr  curVertex = InflPts_8_27[gaussIdx][i];
			const MyVectorI& Dofs = curVertex->getCoupleDof(nCoupleDomainId);
			vecDofs.push_back(Dofs(0));
			vecDofs.push_back(Dofs(1));
			vecDofs.push_back(Dofs(2));
		}
	}

	void Cell::get_dof_indices(unsigned gaussIdx,std::vector<int> &vecDofs)
	{
		vecDofs.clear();
		const unsigned nInflPts = InflPts_8_27[gaussIdx].size();
		for (unsigned i = 0 ; i < nInflPts;++i)
		{
			VertexPtr  curVertex = InflPts_8_27[gaussIdx][i];
			vecDofs.push_back(curVertex->getDof(0));
			vecDofs.push_back(curVertex->getDof(1));
			vecDofs.push_back(curVertex->getDof(2));
		}
	}

	int Cell::s_nFEM_Cell_Count = 0;
	int Cell::s_nEFG_Cell_Count = 0;
	int Cell::s_nCOUPLE_Cell_Count = 0;

	void Cell::initialize()
	{
		if (FEM == m_CellType)
		{
			printf("Cell FEM\n");
			m_Polynomials = Polynomial::generate_complete_basis(1);
			computeGaussPoint();
			computeShapeFunction();
			computeJxW();
			computeShapeGrad();
			makeMaterialMatrix(m_nDomainId);
			makeStiffnessMatrix();
			makeMassMatrix();
			makeLocalRhs();
			s_nFEM_Cell_Count++;
		}
		else if (EFG == m_CellType)
		{
			printf("Cell EFG\n");
			computeGaussPoint();
			makeMaterialMatrix(m_nDomainId);
			computeShapeFunction();
			s_nEFG_Cell_Count++;
		}
		else if (COUPLE == m_CellType)
		{
			printf("Cell Couple\n");
			m_Polynomials = Polynomial::generate_complete_basis(1);
			computeGaussPoint();
			makeMaterialMatrix(m_nDomainId);
			computeShapeFunctionCouple();
			//MyPause;
			s_nCOUPLE_Cell_Count++;
		}
	}

	void Cell::initialize_Couple()
	{
		m_CellType = FEM;
		//printf("Cell FEM\n");
		m_Polynomials = Polynomial::generate_complete_basis(1);
		computeGaussPoint();
		computeShapeFunction();
		computeJxW();
		computeShapeGrad();
		makeMaterialMatrix(m_nDomainId);

		makeLocalStiffnessMatrix_FEM();
		makeLocalMassMatrix_FEM();
		makeLocalRhs_FEM();
	}

	void Cell::initialize_Couple_Joint()
	{
		m_CellType = EFG;
		computeGaussPoint();
		makeMaterialMatrix(m_nDomainId);
		computeShapeFunction_Couple_EFG();
	}

	void Cell::assembleSystemMatrix_Joint()
	{
		
	}

	void Cell::clear()
	{
		for (unsigned i=0;i<Geometry::gauss_Sample_Point;++i)
		{
			shapeFunctionInEFG_8_27[i].resize(0);
			shapeDerivativeValue_8_27_3[i].resize(0,0);
			//StiffnessMatrix_81_81[i].resize(0,0);
			MassMatrix_81_81[i].resize(0,0);
			RightHandValue_81_1[i].resize(0);

			for (unsigned j=0;j<Geometry::vertexs_per_cell;++j)
			{
				//shapeDerivativeValue_8_8_3[i][j].resize(0);
				//shapeDerivativeValue_mapping_8_8_3[i][j].resize(0);
				shapeSecondDerivativeValue_8_8_3_3[i][j].resize(0,0);
			}
		}

		shapeFunctionValue_8_8.resize(0,0);
		StrainMatrix_6_24.resize(0,0);
		//StiffnessMatrix_24_24.resize(0,0);
		MassMatrix_24_24.resize(0,0);
		//RightHandValue_24_1.resize(0,0);
		m_Polynomials.clear();
	}
	int Cell::isHasInitialized(const int nDomainId, const long radius, const std::vector< tuple_matrix >& mapp )
	{
		for (unsigned i=0;i<mapp.size();++i)
		{
			if (nDomainId == mapp.at(i).m_nDomainId && radius == mapp.at(i).m_Radius)
			{
				return i;
			}
		}
		return -1;
	}

	int Cell::isHasInitialized(const int nDomainId, const long radius, const std::vector< tuple_vector >& mapp )
	{
		for (unsigned i=0;i<mapp.size();++i)
		{
			if (nDomainId == mapp.at(i).m_nDomainId && radius == mapp.at(i).m_Radius)
			{
				return i;
			}
		}
		return -1;
	}

	int Cell::appendMatrix(std::vector< tuple_matrix >& mapp,const int nDomainId, const long radius,const MyDenseMatrix& matrix)
	{
		int id = mapp.size();
		tuple_matrix t;
		t.m_nDomainId = nDomainId;
		t.m_Radius = radius;
		t.matrix = matrix;
		mapp.push_back(t);
		return id;
	}

	int Cell::appendVector(std::vector< tuple_vector >& mapp,const int nDomainId, const long radius,const MyVector& Vector)
	{
		int id = mapp.size();
		tuple_vector t;
		t.m_nDomainId = nDomainId;
		t.m_Radius = radius;
		t.vec = Vector;
		mapp.push_back(t);
		return id;
	}

	void Cell::appendFEMShapeValue(Cell * pThis)
	{
		/*MyMatrix shapeFunctionValue_8_8;
		MyDenseVector shapeDerivativeValue_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];*/
		FEMShapeValue tmpNode;
		for (int v=0;v<Geometry::vertexs_per_cell;++v)
		{
			for (int i=0;i<Geometry::vertexs_per_cell;++i)
			{
				tmpNode.shapeFunctionValue_8_8[v][i] = (pThis->shapeFunctionValue_8_8).coeff(v,i);
				tmpNode.shapeDerivativeValue_8_8_3[v][i][0] = (pThis->shapeDerivativeValue_8_8_3)[v][i][0];
				tmpNode.shapeDerivativeValue_8_8_3[v][i][1] = (pThis->shapeDerivativeValue_8_8_3)[v][i][1];
				tmpNode.shapeDerivativeValue_8_8_3[v][i][2] = (pThis->shapeDerivativeValue_8_8_3)[v][i][2];
			}
		}
		vec_FEM_ShapeValue.push_back(tmpNode);		
	}

	void Cell::assembleSystemMatrix()
	{
		std::vector<int> vecDofs;

		get_dof_indices_Local_FEM(m_nDomainId,vecDofs);
		/*printf("vecDofs.size %d\n",vecDofs.size());
		MyPause;*/
		
		compressLocalFEMStiffnessMatrix(m_nDomainId,vec_cell_stiffness_matrix[m_nStiffnessMatrixIdx].matrix,vecDofs);

		compressLocalFEMMassMatrix(m_nDomainId,vec_cell_mass_matrix[m_nMassMatrixIdx].matrix,vecDofs);

		compressLocalFEMRHS(m_nDomainId,vec_cell_rhs_matrix[m_nRhsIdx].vec,vecDofs);
	}

	void Cell::makeLocalStiffnessMatrix_FEM()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nStiffnessMatrixIdx = isHasInitialized(m_nDomainId,lRadius,vec_cell_stiffness_matrix);
		if (-1 < m_nStiffnessMatrixIdx)
		{
			//has initialized
			return ;
		}
		std::vector<int> vecDofs;
		StrainMatrix_6_24.resize(6,24);
		StrainMatrix_6_24.setZero();

		StiffnessMatrix_24_24.resize(24,24);
		StiffnessMatrix_24_24.setZero();


		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			StrainMatrix_6_24.setZero();
			for (unsigned I=0;I < Geometry::vertexs_per_cell;++I)
			{
				const unsigned col = I*3;

				StrainMatrix_6_24.coeffRef(0,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][0];
				StrainMatrix_6_24.coeffRef(1,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][1];
				StrainMatrix_6_24.coeffRef(2,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][2];

				StrainMatrix_6_24.coeffRef(3,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][2];
				StrainMatrix_6_24.coeffRef(3,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][1];

				StrainMatrix_6_24.coeffRef(4,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][2];
				StrainMatrix_6_24.coeffRef(4,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][0];

				StrainMatrix_6_24.coeffRef(5,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][1];
				StrainMatrix_6_24.coeffRef(5,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][0];
			}

			StiffnessMatrix_24_24 += StrainMatrix_6_24.transpose() * MaterialMatrix_6_5[m_nDomainId] * StrainMatrix_6_24*getJxW(q);
		}

		m_nStiffnessMatrixIdx = appendMatrix(vec_cell_stiffness_matrix,m_nDomainId,lRadius,StiffnessMatrix_24_24);
		appendFEMShapeValue(this);
	}


	void Cell::makeLocalMassMatrix_FEM()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nMassMatrixIdx = isHasInitialized(m_nDomainId,lRadius,vec_cell_mass_matrix);
		if (-1 < m_nMassMatrixIdx)
		{
			//has initialized
			return ;
		}

		std::vector<int> vecDofs;
		const MyFloat dbDensity = Material::Density*10;
		MassMatrix_24_24.resize(24,24);
		MassMatrix_24_24.setZero();

		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			MyMatrix tmpShapeMatrix(3,24);
			tmpShapeMatrix.setZero();
			for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
			{
				const unsigned col = k*dim;
				for (unsigned i=0;i<dim;++i)
				{
					tmpShapeMatrix.coeffRef(0,col+0) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(1,col+1) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(2,col+2) = shapeFunctionValue_8_8.coeff(q,k);
				}
			}

			MassMatrix_24_24 += dbDensity * tmpShapeMatrix.transpose() * tmpShapeMatrix * getJxW(q) ;
		}

		m_nMassMatrixIdx = appendMatrix(vec_cell_mass_matrix,m_nDomainId,lRadius,MassMatrix_24_24);
	}

	void Cell::makeLocalRhs_FEM()
	{

		long lRadius = ValueScaleFactor*m_radius;
		m_nRhsIdx = isHasInitialized(m_nDomainId,lRadius,vec_cell_rhs_matrix);
		if (-1 < m_nRhsIdx)
		{
			//has initialized
			return ;
		}

		std::vector<int> vecDofs;

		RightHandValue_24_1.resize(24,1);
		RightHandValue_24_1.setZero();

		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			MyMatrix tmpShapeMatrix(3,24);
			tmpShapeMatrix.setZero();
			for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
			{
				const unsigned col = k*dim;
				for (unsigned i=0;i<dim;++i)
				{
					tmpShapeMatrix.coeffRef(0,col+0) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(1,col+1) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(2,col+2) = shapeFunctionValue_8_8.coeff(q,k);
				}
			}

			RightHandValue_24_1 += tmpShapeMatrix.transpose() * externalForce * getJxW(q) ;
		}

		m_nRhsIdx = appendVector(vec_cell_rhs_matrix,m_nDomainId,lRadius,RightHandValue_24_1);
	}

	void Cell::compressLocalFEMStiffnessMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &vecDofs)
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_LocalStiffness[id]);
	}

	void Cell::compressLocalFEMMassMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &vecDofs)		
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_LocalMass[id]);
	}

	void Cell::compressLocalFEMRHS(MyInt id,const MyVector& rhs,std::vector<int> &vecDofs)
	{
		for (unsigned r=0;r<rhs.size();++r)
		{
			const MyFloat val = rhs[r];
			if (!numbers::isZero(val))
			{
				m_TripletNode_LocalRhs[id][vecDofs[r]].val += val;
			}
		}
	}

	void Cell::compressCoupleEFGStiffnessMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &vecDofs)
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_LocalStiffness_EFG[id]);
	}

	void Cell::compressCoupleEFGMassMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &vecDofs)
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_LocalMass_EFG[id]);
	}

	void Cell::compressCoupleEFGRHS(MyInt id,const MyVector& rhs,std::vector<int> &vecDofs)
	{
		for (unsigned r=0;r<rhs.size();++r)
		{
			const MyFloat val = rhs[r];
			if (!numbers::isZero(val))
			{
				m_TripletNode_LocalRhs_EFG[id][vecDofs[r]].val += val;
			}
		}
	}

	void Cell::makeStiffnessMatrix()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nStiffnessMatrixIdx = isHasInitialized(m_nDomainId,lRadius,vec_cell_stiffness_matrix);
		if (-1 < m_nStiffnessMatrixIdx)
		{
			//has initialized
			return ;
		}
		std::vector<int> vecDofs;
		StrainMatrix_6_24.resize(6,24);
		StrainMatrix_6_24.setZero();

		StiffnessMatrix_24_24.resize(24,24);
		StiffnessMatrix_24_24.setZero();

		
		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			StrainMatrix_6_24.setZero();
			for (unsigned I=0;I < Geometry::vertexs_per_cell;++I)
			{
				const unsigned col = I*3;

				StrainMatrix_6_24.coeffRef(0,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][0];
				StrainMatrix_6_24.coeffRef(1,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][1];
				StrainMatrix_6_24.coeffRef(2,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][2];

				StrainMatrix_6_24.coeffRef(3,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][2];
				StrainMatrix_6_24.coeffRef(3,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][1];

				StrainMatrix_6_24.coeffRef(4,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][2];
				StrainMatrix_6_24.coeffRef(4,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][0];

				StrainMatrix_6_24.coeffRef(5,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][1];
				StrainMatrix_6_24.coeffRef(5,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][0];
			}

			StiffnessMatrix_24_24 += StrainMatrix_6_24.transpose() * MaterialMatrix_6_5[m_nDomainId] * StrainMatrix_6_24*getJxW(q);
		}
		get_dof_indices(vecDofs);
		compressStiffnessMatrix(StiffnessMatrix_24_24,vecDofs);

		m_nStiffnessMatrixIdx = appendMatrix(vec_cell_stiffness_matrix,m_nDomainId,lRadius,StiffnessMatrix_24_24);
		appendFEMShapeValue(this);
		
	}

	void Cell::makeMaterialMatrix(const int nDomainId)
	{
		Q_ASSERT( (nDomainId >= 0) && (nDomainId < Cell::LocalDomainCount) );
		if (bMaterialMatrixInitial[nDomainId])
		{
			return ;
		}
		else
		{
			bMaterialMatrixInitial[nDomainId] = true;
		}

		MyMatrix& MaterialMatrix_6_6 = MaterialMatrix_6_5[nDomainId];
		MaterialMatrix_6_6.resize(MaterialMatrixSize,MaterialMatrixSize);
		MaterialMatrix_6_6.setZero();

		MyFloat E = Material::YoungModulus;
		MyFloat mu = Material::PossionRatio;

		/*if (2 == nDomainId)
		{
			E = Material::YoungModulus/2;
		}
		else if (4 == nDomainId)
		{
			E = Material::YoungModulus * 100;
			mu = 0.01f;
		}*/

		MyFloat G = E/(2*(1+mu));
		MyFloat lai = mu*E/((1+mu)*(1-2*mu));
		
		
		MaterialMatrix_6_6.coeffRef(0,0) = lai + 2*G;

		MaterialMatrix_6_6.coeffRef(1,0) = lai;
		MaterialMatrix_6_6.coeffRef(1,1) = lai + 2*G;

		MaterialMatrix_6_6.coeffRef(2,0) = lai;
		MaterialMatrix_6_6.coeffRef(2,1) = lai;
		MaterialMatrix_6_6.coeffRef(2,2) = lai + 2*G;

		MaterialMatrix_6_6.coeffRef(3,0) = 0.0;
		MaterialMatrix_6_6.coeffRef(3,1) = 0.0;
		MaterialMatrix_6_6.coeffRef(3,2) = 0,0;
		MaterialMatrix_6_6.coeffRef(3,3) = G;

		MaterialMatrix_6_6.coeffRef(4,0) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,1) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,2) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,3) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,4) = G;

		MaterialMatrix_6_6.coeffRef(5,0) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,1) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,2) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,3) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,4) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,5) = G;

		makeSymmetry(MaterialMatrix_6_6);

		//std::cout << MaterialMatrix_6_6 << std::endl;
	}

	void Cell::makeSymmetry(MyMatrix& objMatrix)
	{
		for (unsigned r=0;r<objMatrix.rows();++r)
		{
			for (unsigned c=0;c<r;++c)
			{
				objMatrix.coeffRef(c,r) = objMatrix.coeff(r,c);
			}
		}
	}

	void Cell::makeGaussPoint(double gauss[2],double w)
	{
		static unsigned int index[8][3] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};

		for (unsigned int idx = 0; idx <Geometry::vertexs_per_cell;++idx)
		{
			GaussVertex[idx] = MyPoint(gauss[index[idx][0]],gauss[index[idx][1]],gauss[index[idx][2]]);
			GaussVertexWeight[idx] = w;
		}
	}

	void Cell::computeGaussPoint()
	{
		static bool bFirst=true;
		if (!bFirst)
		{
			return ;
		}
		bFirst = false;
		//const T sqrt3_Reciprocal = 1.0/sqrt(3.0f);
		const  double long_double_eps = static_cast< double>(std::numeric_limits< double>::epsilon());
		const  double double_eps      = static_cast< double>(std::numeric_limits<double>::epsilon());
		volatile  double runtime_one = 1.0;
		const  double tolerance
			= (runtime_one + long_double_eps != runtime_one
			?
			(((double_eps / 100) > (long_double_eps * 5)) ? (double_eps / 100) : (long_double_eps * 5)) 
			/*std::max (double_eps / 100, long_double_eps * 5)*/
			:
			double_eps * 5
			);

		//printf("YANGCHENDEBUG QGauss<1>::QGauss long_double_eps=%f double_eps=%f tolerance=%f\n",long_double_eps,double_eps,tolerance);
		//std::cout << "long_double_eps=" << long_double_eps << " double_eps=" << double_eps << " tolerance=" << tolerance << std::endl;
		const unsigned int n_Quads = 2;
		double gauss[2];

		const unsigned int m = (n_Quads+1)/2;

		for (unsigned int i=1; i<=m; ++i)
		{
			double z = std::cos(numbers::PI * (i-.25)/(n_Quads+.5));


			double pp;
			double p1, p2, p3;

			// Newton iteration
			do
			{
				// compute L_n (z)
				p1 = 1.;
				p2 = 0.;
				for (unsigned int j=0;j<n_Quads;++j)
				{
					p3 = p2;
					p2 = p1;
					p1 = ((2.*j+1.)*z*p2-j*p3)/(j+1);
				}
				pp = n_Quads*(z*p1-p2)/(z*z-1);
				z = z-p1/pp;

				//std::cout << "pp=" << pp << " p1=" << p1 << " p2=" << p2 << " p3=" << p3 << " z=" << z << std::endl;
			}
			while (std::fabs(p1/pp) > tolerance);

			double x = .5*z;
			double w = 1./((1.-z*z)*pp*pp);
			gauss[0]=.5-x;
			gauss[1]=.5+x;
			makeGaussPoint(&gauss[0],w*w*w);
			//printf("YANGCHENDEBUG QGauss<1>::QGauss quadrature_points (%f,%f)  weight(%f,%f) z(%f)\n", gauss[0], gauss[1],w,w,z);
			//std::cout << "quadrature_points(" << gauss[0] << "," << gauss[1] << ") weight=" << w << " z=" << z << std::endl;

		}
	}

	void Cell::computeShapeFunction_Couple_EFG()
	{
		const MyInt nCoupleDomainId = getCoupleDomainId();
		Q_ASSERT(nCoupleDomainId != Invalid_Id);
		std::vector<int> vecDofs;
		const unsigned lint = sizeof(GaussVertex)/sizeof(GaussVertex[0]);
		for (int ls=0; ls<lint; ls++)
		{
			const MyPoint& curGaussPoint = GaussVertex[ls];
			MyPoint gaussPointInGlobalCoordinate;
			MyFloat JacobiValue;
			jacobi(curGaussPoint, gaussPointInGlobalCoordinate, JacobiValue);
			setGlobalGaussPoint(ls,gaussPointInGlobalCoordinate);
				
			
			ApproxAtPoint_Couple_EFG(ls,gaussPointInGlobalCoordinate);

			get_dof_indices_Couple_EFG(ls,vecDofs);

			const unsigned NPTS = gaussPointInfluncePointCount_8[ls];
				
			const MyMatrix& DSH = shapeDerivativeValue_8_27_3[ls];
			const MyVector& SHP = shapeFunctionInEFG_8_27[ls];
			const MyFloat JxW = JacobiValue*GaussVertexWeight[ls];
			setEFGJxW(JxW);
			

			StiffnessMatrix_81_81[ls].resize(3*NPTS,3*NPTS);
			MyMatrix& stiffnessMatrix = StiffnessMatrix_81_81[ls];
			stiffnessMatrix.setZero();
				

			MassMatrix_81_81[ls].resize(3*NPTS,3*NPTS);
			MyMatrix& massMatrix = MassMatrix_81_81[ls];
			massMatrix.setZero();
				

			RightHandValue_81_1[ls].resize(3*NPTS);
			MyVector& rhs = RightHandValue_81_1[ls];
			rhs.setZero();

			MyMatrix bb(6,3*NPTS), tmp(3*NPTS,6);
			MyMatrix N_(3,3*NPTS);
			N_.setZero();

				
			for (unsigned m=0; m<NPTS; m++)
			{
				unsigned n=3*m;

				//bb.coeffRef(0,n) = (*DSH).coeffRef(m,0);
				bb.coeffRef(0,n  ) = (DSH).coeffRef(m,0);
				bb.coeffRef(1,n  ) = 0.0;
				bb.coeffRef(2,n  ) = 0.0;
				bb.coeffRef(3,n  ) = (DSH).coeffRef(m,1);
				bb.coeffRef(4,n  ) = (DSH).coeffRef(m,2);
				bb.coeffRef(5,n  ) = 0.0;

				bb.coeffRef(0,n+1) = 0.0;
				bb.coeffRef(1,n+1) = (DSH).coeffRef(m,1);
				bb.coeffRef(2,n+1) = 0.0;
				bb.coeffRef(3,n+1) = (DSH).coeffRef(m,0);
				bb.coeffRef(4,n+1) = 0.0;
				bb.coeffRef(5,n+1) = (DSH).coeffRef(m,2);

				bb.coeffRef(0,n+2) = 0.0;
				bb.coeffRef(1,n+2) = 0.0;
				bb.coeffRef(2,n+2) = (DSH).coeffRef(m,2);
				bb.coeffRef(3,n+2) = 0.0;	
				bb.coeffRef(4,n+2) = (DSH).coeffRef(m,0);
				bb.coeffRef(5,n+2) = (DSH).coeffRef(m,1);

					

				N_.coeffRef(0,n+0) = SHP[m];
				N_.coeffRef(1,n+1) = SHP[m];
				N_.coeffRef(2,n+2) = SHP[m];					
			}
#if 0
			printf("shape function : \n");
			for (unsigned m=0; m<NPTS; m++)
			{
				printf("%f,%f,%f\n %f\n",(DSH).coeffRef(m,0),(DSH).coeffRef(m,1),(DSH).coeffRef(m,2),SHP[m]);
			}
			printf("\n");
			MyPause;
#endif

			blasOperator::Blas_Mat_Trans_Mat_Mult( bb, MaterialMatrix_6_5[m_nDomainId], tmp, 1.0,0.0);
			blasOperator::Blas_Mat_Mat_Mult( tmp, bb, stiffnessMatrix, 1.0, 0.0 );
			blasOperator::Blas_Scale(JxW, stiffnessMatrix);

				
			compressCoupleEFGStiffnessMatrix(nCoupleDomainId,stiffnessMatrix,vecDofs);

			massMatrix = Material::Density * 10 * N_.transpose() * N_ * JxW;
			compressCoupleEFGMassMatrix(nCoupleDomainId,massMatrix,vecDofs);
				
			rhs = N_.transpose() * externalForce * JxW;
			compressCoupleEFGRHS(nCoupleDomainId,rhs,vecDofs);
				
			//make Pj for every gauss point
			{
				Pj_EFG[ls].resize(3*NPTS);Pj_EFG[ls].setZero();
					
				for (unsigned i=0;i<NPTS;++i)
				{
					Pj_EFG[ls].block(i*3,0,3,1) = InflPts_8_27[ls][i]->getPos();
				}
					
			}
				
		}
	}

	void Cell::computeShapeFunction()
	{
		if (FEM == m_CellType)
		{
			const unsigned int n_points=Geometry::vertexs_per_cell;

			std::vector< MyFloat> values(Geometry::dofs_per_cell_8);
			std::vector< MyDenseVector > grads(Geometry::dofs_per_cell_8);
			std::vector< MyMatrix_3X3 > grad_grads(0);

			for (unsigned int k=0; k<n_points; ++k)
			{
				compute(GaussVertex[k], values, grads, grad_grads);

				for (unsigned int i=0; i< Geometry::dofs_per_cell_8; ++i)
				{
					shapeFunctionValue_8_8.coeffRef(k,i) = values[i];
					//printf("shape function %f\n",values[i]);
				}

				for (unsigned int i=0; i< Geometry::dofs_per_cell_8; ++i)
				{
					shapeDerivativeValue_8_8_3[k][i] = grads[i];
				}
			}

			Pj_FEM.resize(Geometry::dofs_per_cell);Pj_FEM.setZero();
			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				Pj_FEM.block(i*3,0,3,1) = m_elem_vertex[i]->getPos();
			}
			//MyPause;
		}
		else if (EFG == m_CellType)
		{
			std::vector<int> vecDofs;
			const unsigned lint = sizeof(GaussVertex)/sizeof(GaussVertex[0]);
			for (int ls=0; ls<lint; ls++)
			{
				const MyPoint& curGaussPoint = GaussVertex[ls];
				MyPoint gaussPointInGlobalCoordinate;
				MyFloat JacobiValue;
				jacobi(curGaussPoint, gaussPointInGlobalCoordinate, JacobiValue);
				setGlobalGaussPoint(ls,gaussPointInGlobalCoordinate);
				//ms_->GetEFGCell(i)->Mat()->DMatrix(ms_->Plane(), DD);
				//ApproxAtPoint(&x, 1, NPTS, nd, SHP, DSH, D2SH);
				
				ApproxAtPoint(ls,gaussPointInGlobalCoordinate,false);

				get_dof_indices(ls,vecDofs);

				const unsigned NPTS = gaussPointInfluncePointCount_8[ls];
				
				/*printf("NPTS %d\n",NPTS);
				MyPause;*/
				const MyMatrix& DSH = shapeDerivativeValue_8_27_3[ls];
				const MyVector& SHP = shapeFunctionInEFG_8_27[ls];
				const MyFloat JxW = JacobiValue*GaussVertexWeight[ls];
				setEFGJxW(JxW);
			

				StiffnessMatrix_81_81[ls].resize(3*NPTS,3*NPTS);
				MyMatrix& stiffnessMatrix = StiffnessMatrix_81_81[ls];
				stiffnessMatrix.setZero();
				

				MassMatrix_81_81[ls].resize(3*NPTS,3*NPTS);
				MyMatrix& massMatrix = MassMatrix_81_81[ls];
				massMatrix.setZero();
				

				RightHandValue_81_1[ls].resize(3*NPTS);
				MyVector& rhs = RightHandValue_81_1[ls];
				rhs.setZero();

				MyMatrix bb(6,3*NPTS), tmp(3*NPTS,6);
				MyMatrix N_(3,3*NPTS);
				N_.setZero();

				
				for (unsigned m=0; m<NPTS; m++)
				{
					unsigned n=3*m;

					//bb.coeffRef(0,n) = (*DSH).coeffRef(m,0);
					bb.coeffRef(0,n  ) = (DSH).coeffRef(m,0);
					bb.coeffRef(1,n  ) = 0.0;
					bb.coeffRef(2,n  ) = 0.0;
					bb.coeffRef(3,n  ) = (DSH).coeffRef(m,1);
					bb.coeffRef(4,n  ) = (DSH).coeffRef(m,2);
					bb.coeffRef(5,n  ) = 0.0;

					bb.coeffRef(0,n+1) = 0.0;
					bb.coeffRef(1,n+1) = (DSH).coeffRef(m,1);
					bb.coeffRef(2,n+1) = 0.0;
					bb.coeffRef(3,n+1) = (DSH).coeffRef(m,0);
					bb.coeffRef(4,n+1) = 0.0;
					bb.coeffRef(5,n+1) = (DSH).coeffRef(m,2);

					bb.coeffRef(0,n+2) = 0.0;
					bb.coeffRef(1,n+2) = 0.0;
					bb.coeffRef(2,n+2) = (DSH).coeffRef(m,2);
					bb.coeffRef(3,n+2) = 0.0;	
					bb.coeffRef(4,n+2) = (DSH).coeffRef(m,0);
					bb.coeffRef(5,n+2) = (DSH).coeffRef(m,1);

					

					N_.coeffRef(0,n+0) = SHP[m];
					N_.coeffRef(1,n+1) = SHP[m];
					N_.coeffRef(2,n+2) = SHP[m];					
				}
#if 0
				printf("shape function : \n");
				for (unsigned m=0; m<NPTS; m++)
				{
					printf("%f,%f,%f\n %f\n",(DSH).coeffRef(m,0),(DSH).coeffRef(m,1),(DSH).coeffRef(m,2),SHP[m]);
				}
				printf("\n");
				MyPause;
#endif

				blasOperator::Blas_Mat_Trans_Mat_Mult( bb, MaterialMatrix_6_5[m_nDomainId], tmp, 1.0,0.0);
				blasOperator::Blas_Mat_Mat_Mult( tmp, bb, stiffnessMatrix, 1.0, 0.0 );
				blasOperator::Blas_Scale(JxW, stiffnessMatrix);

				
				compressStiffnessMatrix(stiffnessMatrix,vecDofs);

				massMatrix = Material::Density * N_.transpose() * N_ * JxW;
				compressMassMatrix(massMatrix,vecDofs);
				
				rhs = N_.transpose() * externalForce * JxW;
				compressRHS(rhs,vecDofs);
				
				//make Pj for every gauss point
				{
					Pj_EFG[ls].resize(3*NPTS);Pj_EFG[ls].setZero();
					
					for (unsigned i=0;i<NPTS;++i)
					{
						Pj_EFG[ls].block(i*3,0,3,1) = InflPts_8_27[ls][i]->getPos();
					}
					
				}
				
			}
		}
	}

	void Cell::computeShapeFunctionCouple()
	{
		//make FEM Shape Function
		{
			const unsigned int n_points=Geometry::vertexs_per_cell;

			std::vector< MyFloat> values(Geometry::dofs_per_cell_8);
			std::vector< MyDenseVector > grads(Geometry::dofs_per_cell_8);
			std::vector< MyMatrix_3X3 > grad_grads(0);

			for (unsigned int k=0; k<n_points; ++k)
			{
				compute(GaussVertex[k], values, grads, grad_grads);

				for (unsigned int i=0; i< Geometry::dofs_per_cell_8; ++i)
				{
					shapeFunctionValue_8_8.coeffRef(k,i) = values[i];
				}

				for (unsigned int i=0; i< Geometry::dofs_per_cell_8; ++i)
				{
					shapeDerivativeValue_8_8_3[k][i] = grads[i];
				}
			}
			computeJxW();
			computeShapeGrad();
		}
		{
			//compute R(x) for couple
			m_vec_VertexIdInEFG_Boundary.clear();
			for (unsigned i=0;i< Geometry::vertexs_per_cell;++i)
			{
				if (2 == m_elem_vertex[i]->getFromDomainId())
				{
					m_vec_VertexIdInEFG_Boundary.push_back(std::make_pair(m_elem_vertex[i]->getId(),i));
				}
				else if (0 == m_elem_vertex[i]->getFromDomainId() ||4 == m_elem_vertex[i]->getFromDomainId()  )
				{
					m_vec_VertexIdInFEM_Boundary.push_back(std::make_pair(m_elem_vertex[i]->getId(),i));
				}
				else
				{
					printf("error 503\n");
					MyPause;
				}
			}
			/*if (4 != m_vec_VertexIdInEFG_Boundary.size())
			{
				printf("m_vec_VertexIdInEFG_Boundary.size() = %d\n",m_vec_VertexIdInEFG_Boundary.size());
			}
			Q_ASSERT(4 == m_vec_VertexIdInEFG_Boundary.size());*/

			m_vec_RInGaussPt_8.resize(Geometry::gauss_Sample_Point);
			m_vec_RDerivInGaussPt_8_3.resize(Geometry::gauss_Sample_Point);
			for (unsigned gaussPtIdx=0;gaussPtIdx < Geometry::gauss_Sample_Point;++gaussPtIdx)
			{
				m_vec_RInGaussPt_8[gaussPtIdx] = 0.0f;
				m_vec_RDerivInGaussPt_8_3[gaussPtIdx].setZero();
				for (unsigned supportPtOnEFGBoundaryIdx=0; supportPtOnEFGBoundaryIdx<m_vec_VertexIdInEFG_Boundary.size();++supportPtOnEFGBoundaryIdx)
				{
					//int nSptPtId = m_vec_VertexIdInEFG_Boundary[supportPtOnEFGBoundaryIdx].first;
					int nSptPtIdx = m_vec_VertexIdInEFG_Boundary[supportPtOnEFGBoundaryIdx].second;

					m_vec_RInGaussPt_8[gaussPtIdx] += shapeFunctionValue_8_8.coeff(gaussPtIdx,nSptPtIdx);

					for (unsigned dimI=0;dimI<dim;++dimI)
					{
						m_vec_RDerivInGaussPt_8_3[gaussPtIdx][dimI] += shapeDerivativeValue_8_8_3[gaussPtIdx][nSptPtIdx][dimI];
					}
				}
			}

			for (unsigned i=0;i<Geometry::gauss_Sample_Point;++i)
			{
				printf("R[%d]=%f\n",i,m_vec_RInGaussPt_8[i]);
			}
			//MyPause;
		}

		//make Couple Cell Shape & Derivate Function
		{
			const std::map<unsigned,bool>& ConstRefMap = m_mapSupportPtsIdMap;
			std::vector<int> vecDofs;

			const unsigned lint = sizeof(GaussVertex)/sizeof(GaussVertex[0]);
			for (int ls=0; ls<lint; ls++)
			{
				const MyFloat RofGaussPt = m_vec_RInGaussPt_8.at(ls);

				const MyPoint& curGaussPoint = GaussVertex[ls];
				MyPoint gaussPointInGlobalCoordinate;
				MyFloat JacobiValue;
				jacobi(curGaussPoint, gaussPointInGlobalCoordinate, JacobiValue);
				ApproxAtPoint(ls,gaussPointInGlobalCoordinate,true);
				setGlobalGaussPoint(ls,gaussPointInGlobalCoordinate);

				//printf("gaussPointInGlobalCoordinate(%f,%f,%f)\n",gaussPointInGlobalCoordinate[0],gaussPointInGlobalCoordinate[1],gaussPointInGlobalCoordinate[2]);

				const unsigned NPTS = gaussPointInfluncePointCount_8[ls];
				//const unsigned NPTS_Couple = gaussPointInfluncePointCount_8[ls] + m_vec_VertexIdInFEM_Boundary.size();
				const MyFloat JxW = JacobiValue*GaussVertexWeight[ls];
				setEFGJxW(JxW);
				const MyVector& SHP = shapeFunctionInEFG_8_27[ls];
				const MyMatrix& DSH = shapeDerivativeValue_8_27_3[ls];

				MyVector& SHP_Couple = shapeFunctionInCouple_8_31[ls];
				MyMatrix& DSH_Couple = shapeDerivativeValueInCouple_8_31_3[ls];

				/*printf("NPTS %d\n",NPTS);
				MyPause;*/

				SHP_Couple.resize(NPTS);SHP_Couple.setZero();
				DSH_Couple.resize(NPTS,dim);DSH_Couple.setZero();

				std::vector< VertexPtr >& vecEFGSupportPts = InflPts_8_27[ls];
				for (unsigned I=0;I<vecEFGSupportPts.size();++I)
				{
					VertexPtr XI = vecEFGSupportPts[I];
					if (ConstRefMap.find(XI->getId()) != ConstRefMap.end())
					{
						int vvv=-1;
						for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
						{
							if (m_elem_vertex[v]->getId() == XI->getId())
							{
								vvv = v;
								break;
							}
						}
						Q_ASSERT(vvv != -1 /*&& 2 == m_elem_vertex[vvv]->getFromDomainId()*/ );
						const MyFloat NI = shapeFunctionValue_8_8.coeff(ls,vvv);
						const MyFloat FaiI = SHP[I];
						SHP_Couple[I] = (1-RofGaussPt)* NI + RofGaussPt * FaiI;

						for (unsigned derivIdx=0; derivIdx<dim; ++derivIdx)
						{
							const MyFloat Rderiv = m_vec_RDerivInGaussPt_8_3[ls][derivIdx];
							const MyFloat NIderiv = shapeDerivativeValue_8_8_3[ls][vvv][derivIdx];
							const MyFloat FaiIderive = DSH.coeff(I,derivIdx);
							DSH_Couple.coeffRef(I,derivIdx) = Rderiv * NI + (1-RofGaussPt) * NIderiv + Rderiv * FaiI + RofGaussPt * FaiIderive;
						}
					}
					else
					{
						const MyFloat FaiI = SHP[I];
						SHP_Couple[I] = RofGaussPt * FaiI;
						
						for (unsigned derivIdx=0; derivIdx<dim; ++derivIdx)
						{
							const MyFloat Rderiv = m_vec_RDerivInGaussPt_8_3[ls][derivIdx];
							const MyFloat FaiIderive = DSH.coeff(I,derivIdx);
							DSH_Couple.coeffRef(I,derivIdx) = Rderiv * FaiI + RofGaussPt * FaiIderive;
						}
					}
				}//for current gauss point influnces support point

				get_dof_indices(ls,vecDofs);

				/*std::cout << SHP_Couple << std::endl;
				std::cout << "********************************************" << std::endl;*/
				

				StiffnessMatrix_81_81[ls].resize(3*NPTS,3*NPTS);
				MyMatrix& stiffnessMatrix = StiffnessMatrix_81_81[ls];
				stiffnessMatrix.setZero();


				MassMatrix_81_81[ls].resize(3*NPTS,3*NPTS);
				MyMatrix& massMatrix = MassMatrix_81_81[ls];
				massMatrix.setZero();


				RightHandValue_81_1[ls].resize(3*NPTS);
				MyVector& rhs = RightHandValue_81_1[ls];
				rhs.setZero();

				MyMatrix bb(6,3*NPTS), tmp(3*NPTS,6);
				MyMatrix N_(3,3*NPTS);
				N_.setZero();

				//printf("shape function couple\n");
				for (unsigned m=0; m<NPTS; m++)
				{
					unsigned n=3*m;

					//bb.coeffRef(0,n) = (*DSH).coeffRef(m,0);
					bb.coeffRef(0,n  ) = (DSH_Couple).coeff(m,0);
					bb.coeffRef(1,n  ) = 0.0;
					bb.coeffRef(2,n  ) = 0.0;
					bb.coeffRef(3,n  ) = (DSH_Couple).coeff(m,1);
					bb.coeffRef(4,n  ) = (DSH_Couple).coeff(m,2);
					bb.coeffRef(5,n  ) = 0.0;

					bb.coeffRef(0,n+1) = 0.0;
					bb.coeffRef(1,n+1) = (DSH_Couple).coeffRef(m,1);
					bb.coeffRef(2,n+1) = 0.0;
					bb.coeffRef(3,n+1) = (DSH_Couple).coeffRef(m,0);
					bb.coeffRef(4,n+1) = 0.0;
					bb.coeffRef(5,n+1) = (DSH_Couple).coeffRef(m,2);

					bb.coeffRef(0,n+2) = 0.0;
					bb.coeffRef(1,n+2) = 0.0;
					bb.coeffRef(2,n+2) = (DSH_Couple).coeffRef(m,2);
					bb.coeffRef(3,n+2) = 0.0;	
					bb.coeffRef(4,n+2) = (DSH_Couple).coeffRef(m,0);
					bb.coeffRef(5,n+2) = (DSH_Couple).coeffRef(m,1);


					N_.coeffRef(0,n+0) = SHP_Couple[m];
					N_.coeffRef(1,n+1) = SHP_Couple[m];
					N_.coeffRef(2,n+2) = SHP_Couple[m];

					//printf("DSH(%f,%f,%f) SHP(%f)\n",(DSH_Couple).coeffRef(m,0),(DSH_Couple).coeffRef(m,1),(DSH_Couple).coeffRef(m,2),SHP_Couple[m]);
				}
				//MyPause;

				blasOperator::Blas_Mat_Trans_Mat_Mult( bb, MaterialMatrix_6_5[m_nDomainId], tmp, 1.0,0.0);
				blasOperator::Blas_Mat_Mat_Mult( tmp, bb, stiffnessMatrix, 1.0, 0.0 );
				blasOperator::Blas_Scale(JxW, stiffnessMatrix);

				compressStiffnessMatrix(stiffnessMatrix,vecDofs);

				massMatrix = Material::Density * N_.transpose() * N_ * JxW;
				compressMassMatrix(massMatrix,vecDofs);

				rhs = N_.transpose() * externalForce * JxW;
				compressRHS(rhs,vecDofs);

				//make Pj for every gauss point
				{
					Pj_Couple[ls].resize(3*NPTS);Pj_Couple[ls].setZero();

					for (unsigned i=0;i<NPTS;++i)
					{
						Pj_Couple[ls].block(i*3,0,3,1) = InflPts_8_27[ls][i]->getPos();
					}

				}
			}//FOR every gauss point
			
		}
	}//function

	void Cell::computeJxW()
	{
		for (unsigned int point=0; point<Geometry::vertexs_per_cell; ++point )
		{
			for (unsigned int k=0; k< Geometry::shape_Function_Count_In_FEM; ++k)
			{
				const MyDenseVector  &data_derv = shapeDerivativeValue_8_8_3[point][k];
				const MyPoint &supp_pts = m_elem_vertex[k]->getPos();//vertex[k];


				/*std::cout << data_derv << std::endl;
				std::cout << supp_pts << std::endl;*/

				for (unsigned int i=0; i<dim; ++i)
				{
					for (unsigned int j=0; j<dim; ++j)
					{
						contravariant[point].coeffRef(i,j) += data_derv[j] * supp_pts[i];
					}
				}
			}
			
			JxW_values[point] = determinant(contravariant[point])*GaussVertexWeight[point];
		}
	}

	void Cell::computeShapeGrad()
	{
		const unsigned int n_points=Geometry::vertexs_per_cell;
		for (unsigned int k=0; k < n_points;++k)
		{
			covariant[k] = invert(contravariant[k]);
			//printf("curCellRef.covariant[%d] is %f\n",k,covariant[k]);
		}

		for (unsigned int i=0;i<Geometry::dofs_per_cell_8;++i)
		{
			MyDenseVector input[n_points];
			MyDenseVector output[n_points];
			MyDenseVector auxiliary;
			for (unsigned int p=0;p<n_points;++p)
			{
				input[p] = shapeDerivativeValue_8_8_3[p][i];
			}
			for (unsigned int p=0; p<n_points; ++p)
			{
				for (unsigned int d=0;d<dim;++d)
				{
					auxiliary[d] = input[p][d];
				}
				contract (output[p], auxiliary, covariant[p]);
				{
					//{
					//	printf("\n{%f,%f,%f}\n{%f,%f,%f}\n{%f,%f,%f}\n"
					//		,covariant[p].coeff(0,0)/*[0][0]*/,covariant[p].coeff(0,1)/*[0][1]*/,covariant[p].coeff(0,2)/*[0][2]*/
					//		,covariant[p].coeff(1,0)/*[1][0]*/,covariant[p].coeff(1,1)/*[1][1]*/,covariant[p].coeff(1,2)/*[1][2]*/
					//		,covariant[p].coeff(2,0)/*[2][0]*/,covariant[p].coeff(2,1)/*[2][1]*/,covariant[p].coeff(2,2)/*[2][2]*/);
					//	printf("{(%f,%f,%f)-3-12>(%f,%f,%f)}\n",auxiliary[0],auxiliary[1],auxiliary[2],output[p][0],output[p][1],output[p][2]);
					//}
				}
			}
#if 0
			printf("shapeDerivative\n");
			for (unsigned int p=0;p<n_points;++p)
			{
				shapeDerivativeValue_mapping_8_8_3[p][i] = output[p];
				printf("{%f,%f,%f}\n",output[p][0],output[p][1],output[p][2]);
			}
			printf("\n");
			MyPause;
#else
			for (unsigned int p=0;p<n_points;++p)
			{
				shapeDerivativeValue_mapping_8_8_3[p][i] = output[p];
			}
#endif
		}
	}

	void Cell::compute(const MyPoint &p, std::vector< MyFloat> &values, std::vector< MyDenseVector > &grads, std::vector< MyMatrix_3X3 > &grad_grads)
	{

		unsigned int n_values_and_derivatives = 0;
		n_values_and_derivatives = 2;

		//Table<2,Tensor<1,3> > v(dim, Polynomials.size()/*2*/);
		std::vector< std::vector< MyDenseVector > > v(dim);
		for (unsigned int i=0; i < dim; ++i)
		{
			v[i].resize( m_Polynomials.size() );
		}

		{
			std::vector< MyFloat > tmp (n_values_and_derivatives);
			for (unsigned int d=0; d<dim; ++d)
			{
				for (unsigned int i=0; i<m_Polynomials.size(); ++i)
				{
					m_Polynomials[i].value(p(d), tmp);
					for (unsigned int e=0; e<n_values_and_derivatives; ++e)
					{
						v[d][i]/*(d,i)*/[e] = tmp[e];
						//printf("YANGCHENDEBUG 3-12 v(%d,%d)[%d]=tmp[%d] {%f} {p(%d) %f}\n",d,i,e,e,tmp[e],d,p(d));
					}
				}
			}
		}

		for (unsigned int i=0; i<Geometry::n_tensor_pols; ++i)
		{
			unsigned int indices[dim];
			compute_index (i, indices);
			//printf("YANGCHENDEBUG 3-12 [%d]:{%d,%d,%d}\n",i,indices[0],indices[1],indices[2]);

			values[i] = 1;
			for (unsigned int x=0; x<dim; ++x)
			{
				values[i] *= v[x][indices[x]]/*(x,indices[x])*/[0];
			}

			//printf("YANGCHENDEBUG 3-12 value[%d]=%f {v(0,indices[0])[0]=%f}{v(1,indices[1])[1]=%f}{v(2,indices[2])[2]=%f}\n",i,values[i],
			//v[0][indices[0]][0],v[1][indices[1]][0],v[2][indices[2]][0]);

			for (unsigned int d=0; d<dim; ++d)
			{
				grads[i][d] = 1.;
				for (unsigned int x=0; x<dim; ++x)
					grads[i][d] *= v[x][indices[x]]/*(x,indices[x])*/[d==x];

				//printf("YANGCHENDEBUG 3-12  grads[%d][%d]=%f\n",i,d,grads[i][d]);
			}

		}
		//MyPause;
	}

	MyFloat Cell::determinant (const MyMatrix_3X3 &t)
	{
		return  t.coeff(0,0)*t.coeff(1,1)*t.coeff(2,2) + t.coeff(0,1)*t.coeff(1,2)*t.coeff(2,0) + t.coeff(0,2)*t.coeff(1,0)*t.coeff(2,1) -
			t.coeff(2,0)*t.coeff(1,1)*t.coeff(0,2) - t.coeff(1,0)*t.coeff(0,1)*t.coeff(2,2) - t.coeff(0,0)*t.coeff(2,1)*t.coeff(1,2);
	}

	void Cell::contract (MyDenseVector &dest, const MyDenseVector &src1, const MyMatrix_3X3 &src2)
	{
		dest.setZero();
		for (unsigned int i=0; i<dim; ++i)
			for (unsigned int j=0; j<dim; ++j)
				dest[i] += src1[j] * src2.coeff(j,i)/*[j][i]*/;
	}


	MyMatrix_3X3 Cell::invert (const MyMatrix_3X3 &t)
	{
		MyMatrix_3X3 return_tensor;
		const MyFloat t4 = t.coeff(0,0)*t.coeff(1,1),
			t6 = t.coeff(0,0)*t.coeff(1,2),
			t8 = t.coeff(0,1)*t.coeff(1,0),
			t00 = t.coeff(0,2)*t.coeff(1,0),
			t01 = t.coeff(0,1)*t.coeff(2,0),
			t04 = t.coeff(0,2)*t.coeff(2,0),
			t07 = 1.0/(t4*t.coeff(2,2)-t6*t.coeff(2,1)-t8*t.coeff(2,2)+
			t00*t.coeff(2,1)+t01*t.coeff(1,2)-t04*t.coeff(1,1));
		return_tensor.coeffRef(0,0) = (t.coeff(1,1)*t.coeff(2,2)-t.coeff(1,2)*t.coeff(2,1))*t07;
		return_tensor.coeffRef(0,1) = -(t.coeff(0,1)*t.coeff(2,2)-t.coeff(0,2)*t.coeff(2,1))*t07;
		return_tensor.coeffRef(0,2) = -(-t.coeff(0,1)*t.coeff(1,2)+t.coeff(0,2)*t.coeff(1,1))*t07;
		return_tensor.coeffRef(1,0) = -(t.coeff(1,0)*t.coeff(2,2)-t.coeff(1,2)*t.coeff(2,0))*t07;
		return_tensor.coeffRef(1,1) = (t.coeff(0,0)*t.coeff(2,2)-t04)*t07;
		return_tensor.coeffRef(1,2) = -(t6-t00)*t07;
		return_tensor.coeffRef(2,0) = -(-t.coeff(1,0)*t.coeff(2,1)+t.coeff(1,1)*t.coeff(2,0))*t07;
		return_tensor.coeffRef(2,1) = -(t.coeff(0,0)*t.coeff(2,1)-t01)*t07;
		return_tensor.coeffRef(2,2) = (t4-t8)*t07;

		return return_tensor;

	}

	void Cell::compute_index (const unsigned int i, unsigned int  (&indices)[3]) const
	{
		const unsigned int n_pols = m_Polynomials.size();
		static int index_map[Geometry::n_tensor_pols]={0,1,2,3,4,5,6,7};
		const unsigned int n=index_map[i];//index_map[0]=0 index_map[1]=1 index_map[2]=2 index_map[3]=3 index_map[4]=4 index_map[5]=5 index_map[6]=6 index_map[7]=7

		indices[0] = n % n_pols;
		indices[1] = (n/n_pols) % n_pols;
		indices[2] = n / (n_pols*n_pols);
	}

	void Cell::print(std::ostream& out)
	{

		out << "************* support point **********" << std::endl;
		for (unsigned int idx = 0;idx < Geometry::vertexs_per_cell;++idx)
		{
			out << "index[" << idx << "] : " << (m_elem_vertex[idx]->getPos()) << std::endl;
		}

		out << "************* gauss point **********" << std::endl;
		for (unsigned int idx = 0;idx < Geometry::vertexs_per_cell;++idx)
		{
			out << "index[" << idx << "] : " << GaussVertex[idx] << "  weight " << GaussVertexWeight[idx] << std::endl;
		}

		out << "************* shape function **********" << std::endl;
		out << shapeFunctionValue_8_8 << std::endl;

		out << "************* shape function first derivative**********" << std::endl;
		for (unsigned int r=0;r<Geometry::vertexs_per_cell;++r)
		{
			for (unsigned int c=0;c<Geometry::vertexs_per_cell;++c)
			{
				out << "index[" << r << "][" << c << "] : " << std::endl << shapeDerivativeValue_mapping_8_8_3[r][c] << std::endl;
			}
		}

		out << "************* contravariant **********" << std::endl;
		for (unsigned int idx = 0;idx < Geometry::vertexs_per_cell;++idx)
		{
			out << "index[" << idx << "] : " << contravariant[idx] << std::endl;
		}

		out << "************* JxW_values **********" << std::endl;
		for (unsigned int idx = 0;idx < Geometry::vertexs_per_cell;++idx)
		{
			out << "index[" << idx << "] : " << JxW_values[idx] << std::endl;
		}
	}

	void Cell::makeMassMatrix()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nMassMatrixIdx = isHasInitialized(m_nDomainId,lRadius,vec_cell_mass_matrix);
		if (-1 < m_nMassMatrixIdx)
		{
			//has initialized
			return ;
		}

		std::vector<int> vecDofs;
		const MyFloat dbDensity = Material::Density;
		MassMatrix_24_24.resize(24,24);
		MassMatrix_24_24.setZero();

		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			MyMatrix tmpShapeMatrix(3,24);
			tmpShapeMatrix.setZero();
			for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
			{
				const unsigned col = k*dim;
				for (unsigned i=0;i<dim;++i)
				{
					tmpShapeMatrix.coeffRef(0,col+0) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(1,col+1) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(2,col+2) = shapeFunctionValue_8_8.coeff(q,k);
				}
			}

			MassMatrix_24_24 += dbDensity * tmpShapeMatrix.transpose() * tmpShapeMatrix * getJxW(q) ;
		}
		get_dof_indices(vecDofs);
		compressMassMatrix(MassMatrix_24_24,vecDofs);

		m_nMassMatrixIdx = appendMatrix(vec_cell_mass_matrix,m_nDomainId,lRadius,MassMatrix_24_24);
	}

	void Cell::makeLocalRhs()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nRhsIdx = isHasInitialized(m_nDomainId,lRadius,vec_cell_rhs_matrix);
		if (-1 < m_nRhsIdx)
		{
			//has initialized
			return ;
		}

		std::vector<int> vecDofs;
		
		RightHandValue_24_1.resize(24,1);
		RightHandValue_24_1.setZero();

		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			MyMatrix tmpShapeMatrix(3,24);
			tmpShapeMatrix.setZero();
			for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
			{
				const unsigned col = k*dim;
				for (unsigned i=0;i<dim;++i)
				{
					tmpShapeMatrix.coeffRef(0,col+0) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(1,col+1) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(2,col+2) = shapeFunctionValue_8_8.coeff(q,k);
				}
			}

			RightHandValue_24_1 += tmpShapeMatrix.transpose() * externalForce * getJxW(q) ;
		}

		get_dof_indices(vecDofs);
		compressRHS(RightHandValue_24_1,vecDofs);

		m_nRhsIdx = appendVector(vec_cell_rhs_matrix,m_nDomainId,lRadius,RightHandValue_24_1);
	}

	void  Cell::jacobi(const MyPoint& gaussPoint,MyPoint& gaussPointInGlobalCoordinate, MyFloat &weight)
	{
		MyMatrix xs(3,4), shpDeri(4,8), xll(3,8);
		xs.setZero();
		shpDeri.setZero();
		xll.setZero();

		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			xll.coeffRef(0,q) = m_elem_vertex[q]->getPos().x();
			xll.coeffRef(1,q) = m_elem_vertex[q]->getPos().y();
			xll.coeffRef(2,q) = m_elem_vertex[q]->getPos().z();
		}

		const MyFloat x=gaussPoint[0];
		const MyFloat y=gaussPoint[1];
		const MyFloat z=gaussPoint[2];

		shpDeri.coeffRef(3,0) = (1.-x)*(1.-y)*(1.-z);
		shpDeri.coeffRef(3,1) = x*(1.-y)*(1.-z);
		shpDeri.coeffRef(3,2) = (1.-x)*y*(1.-z);
		shpDeri.coeffRef(3,3) = x*y*(1.-z);
		shpDeri.coeffRef(3,4) = (1.-x)*(1.-y)*z;
		shpDeri.coeffRef(3,5) = x*(1.-y)*z;
		shpDeri.coeffRef(3,6) = (1.-x)*y*z;
		shpDeri.coeffRef(3,7) = x*y*z;

		shpDeri.coeffRef(0,0) = (y-1.)*(1.-z);
		shpDeri.coeffRef(0,1) = (1.-y)*(1.-z);
		shpDeri.coeffRef(0,2) = -y*(1.-z);
		shpDeri.coeffRef(0,3) = y*(1.-z);
		shpDeri.coeffRef(0,4) = (y-1.)*z;
		shpDeri.coeffRef(0,5) = (1.-y)*z;
		shpDeri.coeffRef(0,6) = -y*z;
		shpDeri.coeffRef(0,7) = y*z;

		shpDeri.coeffRef(1,0) = (x-1.)*(1.-z);
		shpDeri.coeffRef(1,1) = -x*(1.-z);
		shpDeri.coeffRef(1,2) = (1.-x)*(1.-z);
		shpDeri.coeffRef(1,3) = x*(1.-z);
		shpDeri.coeffRef(1,4) = (x-1.)*z;
		shpDeri.coeffRef(1,5) = -x*z;
		shpDeri.coeffRef(1,6) = (1.-x)*z;
		shpDeri.coeffRef(1,7) = x*z;

		shpDeri.coeffRef(2,0) = (x-1)*(1.-y);
		shpDeri.coeffRef(2,1) = x*(y-1.);
		shpDeri.coeffRef(2,2) = (x-1.)*y;
		shpDeri.coeffRef(2,3) = -x*y;
		shpDeri.coeffRef(2,4) = (1.-x)*(1.-y);
		shpDeri.coeffRef(2,5) = x*(1.-y);
		shpDeri.coeffRef(2,6) = (1.-x)*y;
		shpDeri.coeffRef(2,7) = x*y;

		xs = xll * shpDeri.transpose();

		gaussPointInGlobalCoordinate[0] = xs.coeff(0,3);
		gaussPointInGlobalCoordinate[1] = xs.coeff(1,3);
		gaussPointInGlobalCoordinate[2] = xs.coeff(2,3);

		weight = xs.coeff(0,0)*(xs.coeff(1,1)*xs.coeff(2,2)-xs.coeff(1,2)*xs.coeff(2,1))
				- xs.coeff(0,1)*(xs.coeff(1,0)*xs.coeff(2,2)-xs.coeff(1,2)*xs.coeff(2,0))
				+ xs.coeff(0,2)*(xs.coeff(1,0)*xs.coeff(2,1)-xs.coeff(1,1)*xs.coeff(2,0));
	}

	void Cell::InfluentPoints_Couple_EFG(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate )
	{
		std::map< MyInt,bool > mapInflPts;
		const MyInt nCellSize = getCellSize();
		const MyInt nCoupleDomainId = getCoupleDomainId();
		Q_ASSERT(Invalid_Id != nCoupleDomainId);
		for (MyInt c=0;c<nCellSize;++c)
		{
			CellPtr curCellPtr = getCell(c);
			if (nCoupleDomainId == (curCellPtr->getCoupleDomainId()) )
			{
				for (MyInt v=0;v<Geometry::vertexs_per_cell;++v)
				{
					VertexPtr curVtxPtr = getVertex(v);
					const MyPoint& vertexPos = curVtxPtr->getPos();
					if( abs( gaussPointInGlobalCoordinate.x() - vertexPos.x()) < SupportSize && 
						abs( gaussPointInGlobalCoordinate.y() - vertexPos.y() ) < SupportSize && 
						abs( gaussPointInGlobalCoordinate.z() - vertexPos.z() ) < SupportSize)
					{
						mapInflPts[ (curVtxPtr->getId()) ] = true;
					}
				}
			}
		}
		std::map< MyInt,bool >::const_iterator ci = mapInflPts.begin();
		std::map< MyInt,bool >::const_iterator endc = mapInflPts.end();
		for (;ci != endc;++ci)
		{
			InflPts_8_27[gaussIdx].push_back(Vertex::getVertex(ci->first));
		}

		gaussPointInfluncePointCount_8[gaussIdx] = InflPts_8_27[gaussIdx].size();
	}

	void Cell::InfluentPoints(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate )
	{
		const unsigned Npts = Vertex::getVertexSize();
		
		InflPts_8_27[gaussIdx].clear();
		for( unsigned i = 0; i< Npts; i++ )
		{
			VertexPtr currentVertex = Vertex::getVertex(i);
			if (2 == currentVertex->getFromDomainId())
			{
				const MyPoint& vertexPos = currentVertex->getPos();
				//MyFloat wgt = materialDistance(gaussPointInGlobalCoordinate,vertexPos);
				if( abs( gaussPointInGlobalCoordinate.x() - vertexPos.x()) < SupportSize && 
					abs( gaussPointInGlobalCoordinate.y() - vertexPos.y() ) < SupportSize && 
					abs( gaussPointInGlobalCoordinate.z() - vertexPos.z() ) < SupportSize)
				{
					InflPts_8_27[gaussIdx].push_back(currentVertex);
				}				
			}
		}
		//resortInfluncePtsVec(InflPts_8_27[gaussIdx]);

		gaussPointInfluncePointCount_8[gaussIdx] = InflPts_8_27[gaussIdx].size();
	}

	void Cell::InfluentPoints4Couple(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate )
	{
		const unsigned Npts = Vertex::getVertexSize();

		InflPts_8_27[gaussIdx].clear();
		for( unsigned i = 0; i< Npts; i++ )
		{
			VertexPtr currentVertex = Vertex::getVertex(i);
			if (2 == currentVertex->getFromDomainId())
			{
				const MyPoint& vertexPos = currentVertex->getPos();
				//MyFloat wgt = materialDistance(gaussPointInGlobalCoordinate,vertexPos);
				if( abs( gaussPointInGlobalCoordinate.x() - vertexPos.x()) < SupportSize && 
					abs( gaussPointInGlobalCoordinate.y() - vertexPos.y() ) < SupportSize && 
					abs( gaussPointInGlobalCoordinate.z() - vertexPos.z() ) < SupportSize)
				{
					InflPts_8_27[gaussIdx].push_back(currentVertex);
				}				
			}
		}
		
		for (unsigned i =0;i<Geometry::vertexs_per_cell;++i)
		{
			if (0 == m_elem_vertex[i]->getFromDomainId()||4 == m_elem_vertex[i]->getFromDomainId())
			{
				InflPts_8_27[gaussIdx].push_back(m_elem_vertex[i]);
			}
		}
		//resortInfluncePtsVec(InflPts_8_27[gaussIdx]);

		gaussPointInfluncePointCount_8[gaussIdx] = InflPts_8_27[gaussIdx].size();
	}

	void Cell::resortInfluncePtsVec(std::vector< VertexPtr >& ref_InflunceVec)
	{
		const int nativeCount = ref_InflunceVec.size();
		std::vector< VertexPtr > tmpVec;
		std::vector< VertexPtr >::iterator ci = ref_InflunceVec.begin();
		for (; ci != ref_InflunceVec.end(); )
		{
			const int currentVertexId = (*ci)->getId();
			if ( m_mapSupportPtsIdMap.find(currentVertexId) != m_mapSupportPtsIdMap.end() )
			{
				//current vertex in the current cell
				tmpVec.push_back((*ci));
				ci = ref_InflunceVec.erase(ci);
			}
			else
			{
				++ci;
			}
		}
		copy(ref_InflunceVec.begin(),ref_InflunceVec.end(),back_inserter(tmpVec)) ;
		Q_ASSERT(nativeCount == ref_InflunceVec.size());
	}

	void Cell::basis(const MyPoint& pX, const MyPoint& pXI, MyVector& pP)
	{
		const MyFloat XI = pXI.x();
		const MyFloat YI = pXI.y();
		const MyFloat ZI = pXI.z();

		pP(0) = 1.0;
		pP(1) = XI;
		pP(2) = YI;
		pP(3) = ZI;

		if (EFG_BasisNb_ > 4)
		{
			pP(4) = XI * XI;
			pP(5) = YI * YI;
			pP(6) = ZI * ZI;
			pP(7) = XI * YI;
			pP(8) = XI * ZI;
			pP(9) = YI * ZI;
		}
	}

	void Cell::dbasis(const MyPoint& pX, const MyPoint& pXI, MyMatrix& DP_27_3)
	{
		DP_27_3.setZero();
		MyFloat X = pX.x();
		MyFloat Y = pX.y();
		MyFloat Z = pX.z();

		MyFloat XI = pXI.x();
		MyFloat YI = pXI.y();
		MyFloat ZI = pXI.z();

		DP_27_3.coeffRef(0,0) = 0.0;
		DP_27_3.coeffRef(0,1) = 0.0;
		DP_27_3.coeffRef(0,2) = 0.0;

		DP_27_3.coeffRef(1,0) = 1.0;
		DP_27_3.coeffRef(1,1) = 0.0;
		DP_27_3.coeffRef(1,2) = 0.0;

		DP_27_3.coeffRef(2,0) = 0.0;
		DP_27_3.coeffRef(2,1) = 1.0;
		DP_27_3.coeffRef(2,2) = 0.0;

		DP_27_3.coeffRef(3,0) = 0.0;
		DP_27_3.coeffRef(3,1) = 0.0;
		DP_27_3.coeffRef(3,2) = 1.0;

		if (EFG_BasisNb_ > 4)
		{
			DP_27_3.coeffRef(4,0) = 2 * XI;
			DP_27_3.coeffRef(4,1) = 0.0;
			DP_27_3.coeffRef(4,2) = 0.0;

			DP_27_3.coeffRef(5,0) = 0.0;
			DP_27_3.coeffRef(5,1) = 2 * YI;
			DP_27_3.coeffRef(5,2) = 0.0;

			DP_27_3.coeffRef(6,0) = 0.0;
			DP_27_3.coeffRef(6,1) = 0.0;
			DP_27_3.coeffRef(6,2) = 2 * ZI;	

			DP_27_3.coeffRef(7,0) = YI;
			DP_27_3.coeffRef(7,1) = XI;
			DP_27_3.coeffRef(7,2) = 0.0;	

			DP_27_3.coeffRef(8,0) = ZI;
			DP_27_3.coeffRef(8,1) = 0.0;
			DP_27_3.coeffRef(8,2) = XI;

			DP_27_3.coeffRef(9,0) = 0.0;
			DP_27_3.coeffRef(9,1) = ZI;
			DP_27_3.coeffRef(9,2) = YI;
		}
	}

	

	MyFloat Cell::WeightFun(const MyPoint& pCenterPt, const MyDenseVector& Radius, const MyPoint& pEvPoint, int DerOrder)
	{
		//value = SL234Fun(pCenterPt, Radius, pEvPoint, DerOrder);
		MyFloat difx = ( pEvPoint.x() - pCenterPt.x() );
		MyFloat dify = ( pEvPoint.y() - pCenterPt.y() );
		MyFloat difz = ( pEvPoint.z() - pCenterPt.z() );

		MyFloat drdx = numbers::sign(difx)/Radius[0];
		MyFloat drdy = numbers::sign(dify)/Radius[1];
		MyFloat drdz = numbers::sign(difz)/Radius[2];

		MyFloat rx = abs(difx)/Radius[0];
		MyFloat ry = abs(dify)/Radius[1];
		MyFloat rz = abs(difz)/Radius[2];

		MyFloat wx,wy,wz,dwx,dwy,dwz,ddwx,ddwy,ddwz;

		wx = 1 - 6*rx*rx + 8*rx*rx*rx - 3*rx*rx*rx*rx;
		wy = 1 - 6*ry*ry + 8*ry*ry*ry - 3*ry*ry*ry*ry;
		wz = 1 - 6*rz*rz + 8*rz*rz*rz - 3*rz*rz*rz*rz;

		if( DerOrder == 0 )
			return wx*wy*wz;

		dwx = ( -12*rx+24*rx*rx-12*rx*rx*rx ) * drdx;
		dwy = ( -12*ry+24*ry*ry-12*ry*ry*ry ) * drdy;
		dwz = ( -12*rz+24*rz*rz-12*rz*rz*rz ) * drdz;

		if( DerOrder == 1 )
			return wy*dwx*wz;

		if( DerOrder == 2 )
			return wx*dwy*wz;

		if( DerOrder == 3 )
			return wx*wy*dwz;

		ddwx = ( -12+48*rx-36*rx*rx )*drdx*drdx;
		ddwy = ( -12+48*ry-36*ry*ry )*drdy*drdy;
		ddwz = ( -12+48*rz-36*rz*rz )*drdz*drdz;

		if( DerOrder == 4 )
			return ddwx*wy*wz;
		if( DerOrder == 5 )
			return wx*ddwy*wz;
		if( DerOrder == 6 )
			return wx*wy*ddwz;
		if( DerOrder == 7 )
			return dwx*dwy*wz;
		if( DerOrder == 8 )
			return dwx*wy*dwz;
		if( DerOrder == 9 )
			return wx*dwy*dwz;
		else
			return 0.0;
	}

	void Cell::ApproxAtPoint_Couple_EFG(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate)
	{
		const MyPoint& pX = gaussPointInGlobalCoordinate;
		MyVector& N_ =  shapeFunctionInEFG_8_27[gaussIdx];
		MyMatrix& D1N_ = shapeDerivativeValue_8_27_3[gaussIdx];

		MyFloat WI;         // Value of weight function at node I
		MyFloat WxI, WyI, WzI;   // Value of derivatives of weight function at node I
		MyFloat WxxI, WxyI, WyyI, WzzI, WxzI, WyzI;

		InfluentPoints_Couple_EFG(gaussIdx,gaussPointInGlobalCoordinate);	

		const unsigned InflPtsNb_ = gaussPointInfluncePointCount_8[gaussIdx];
		s_map_InfluncePointSize[InflPtsNb_] = true;
		MyMatrix  A_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ax_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ay_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Az_(EFG_BasisNb_,EFG_BasisNb_);

		MyMatrix  Axx_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ayy_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Azz_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Axy_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Axz_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ayz_(EFG_BasisNb_,EFG_BasisNb_);

		MyMatrix  B_(EFG_BasisNb_,InflPtsNb_);   // 当前点的B矩阵
		MyMatrix  Bx_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对x的偏导数
		MyMatrix  By_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对y的偏导数
		MyMatrix  Bz_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对z的偏导数

		MyMatrix  Bxx_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xx的偏导数
		MyMatrix  Byy_(EFG_BasisNb_,InflPtsNb_); // B矩阵对yy的偏导数
		MyMatrix  Bzz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对zz的偏导数
		MyMatrix  Bxy_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数
		MyMatrix  Bxz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数
		MyMatrix  Byz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数

		MyVector			P_(EFG_BasisNb_);
		MyMatrix            D1P_(EFG_BasisNb_,dim);

		N_.resize(InflPtsNb_);
		D1N_.resize(InflPtsNb_,dim);

		MyDenseVector Radius_;
		Radius_[0] = SupportSize;Radius_[1] = SupportSize;Radius_[2] = SupportSize;

		shapeFunctionInEFG_8_27[gaussIdx].resize(InflPtsNb_);
		shapeDerivativeValue_8_27_3[gaussIdx].resize(InflPtsNb_,dim);

		{
			A_.setZero();
			Ax_.setZero();
			Ay_.setZero();
			Az_.setZero();
			Axx_.setZero();
			Ayy_.setZero();
			Azz_.setZero();
			Axy_.setZero();
			Axz_.setZero();
			Ayz_.setZero();

			B_.setZero();
			Bx_.setZero();
			By_.setZero();
			Bz_.setZero();
			Bxx_.setZero();
			Byy_.setZero();
			Bzz_.setZero();
			Bxy_.setZero();
			Bxz_.setZero();
			Byz_.setZero();
			P_.setZero();
			N_.setZero();
			D1P_.setZero();
		}
		
		std::vector< VertexPtr >& refCurGaussPointInfls = InflPts_8_27[gaussIdx];
		Q_ASSERT(refCurGaussPointInfls.size() == InflPtsNb_);
		for (unsigned i=0; i<InflPtsNb_; i++)
		{
			VertexPtr curInflsPoint = refCurGaussPointInfls[i];
			MyPoint pts_j = curInflsPoint->getPos();

			//     A  = Sum of WI(X)*P(XI)*Transpose[P(XI)], I=1,N

			basis(pX,curInflsPoint->getPos(), P_);
			
			WI = WeightFun(pts_j, Radius_, pX, 0);
			

			//  (*A_) = (*A_) + WI * (*P_) * Transpose [ (*P_) ]
			MyFloat cof = ( i == 0 ? 0:1 );
			blasOperator::Blas_R1_Update(A_, P_, WI, cof);
			

			{
				WxI = WeightFun(pts_j, Radius_, pX, 1);
				WyI = WeightFun(pts_j, Radius_, pX, 2);
				WzI = WeightFun(pts_j, Radius_, pX, 3);

				//  (*Ax_) = (*Ax_) + WxI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ax_, P_, WxI, cof);

				//  (*Ay_) = (*Ay_) + WyI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ay_, P_, WyI, cof);
				blasOperator::Blas_R1_Update(Az_, P_, WzI, cof);
			}

			{
				WxxI = WeightFun(pts_j, Radius_, pX, 4);
				WyyI = WeightFun(pts_j, Radius_, pX, 5);
				WzzI = WeightFun(pts_j, Radius_, pX, 6);
				WxyI = WeightFun(pts_j, Radius_, pX, 7);
				WxzI = WeightFun(pts_j, Radius_, pX, 8);
				WyzI = WeightFun(pts_j, Radius_, pX, 9);
				//  (*Axx_) = (*Axx_) + WxxI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Axx_, P_, WxxI, cof);
				//  (*Ayy_) = (*Ayy_) + WyyI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ayy_, P_, WyyI, cof);
				blasOperator::Blas_R1_Update(Azz_, P_, WzzI, cof);
				//  (*Axy_) = (*Axy_) + WxyI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Axy_, P_, WxyI, cof);
				blasOperator::Blas_R1_Update(Axz_, P_, WxzI, cof);
				blasOperator::Blas_R1_Update(Ayz_, P_, WyzI, cof);
			}

			for ( unsigned k=0; k< EFG_BasisNb_; k++ )
			{
				//(*B_)(k, i) = (*P_)(k) * WI;
				B_.coeffRef(k,i) = P_(k) * WI;
			}
			

			{
				for(unsigned k = 0; k < EFG_BasisNb_; k++ )
				{
					/*(*Bx_)(k, i) = (*P_)(k) * WxI;
					(*By_)(k, i) = (*P_)(k) * WyI;
					(*Bz_)(k, i) = (*P_)(k) * WzI;*/

					Bx_.coeffRef(k, i) = (P_)(k) * WxI;
					By_.coeffRef(k, i) = (P_)(k) * WyI;
					Bz_.coeffRef(k, i) = (P_)(k) * WzI;
				}
			}

			
			{	
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Bxx_).coeffRef(k, i) = (P_)(k) * WxxI;
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Byy_).coeffRef(k, i) = (P_)(k) * WyyI;
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Bzz_).coeffRef(k, i) = (P_)(k) * WzzI;
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Bxy_).coeffRef(k, i) = (P_)(k) * WxyI;
				for(int  k = 0; k< EFG_BasisNb_; k++ )
					(Bxz_).coeffRef(k, i) = (P_)(k) * WxzI;
				for(int  k = 0; k< EFG_BasisNb_; k++ )
					(Byz_).coeffRef(k, i) = (P_)(k) * WyzI;
			}
		}

		Eigen::LLT<MyMatrix> lltOfA(A_);//LaLUFactor(*A_);

		//  r = Inverse[A] * P

		basis(pX, pX, P_);
		P_ = lltOfA.solve(P_);//LaLUSolve(*A_, *P_);  // r is stored in P_

		//  Transpose[N] = Transpose[r] * B
		blasOperator::Blas_Mat_Trans_Vec_Mult(B_, P_, N_, 1.0, 0.0);

		dbasis(pX, pX, D1P_);

		//  r is saved in P, and r' in D1P_
		
		{
			MyMatrix tmpD1P1 = D1P_.block(0,0,EFG_BasisNb_,1);
			blasOperator::Blas_Mat_Vec_Mult(Ax_, P_, tmpD1P1,  -1.0, 1.0);  // P' - A'*r
			D1P_.block(0,0,EFG_BasisNb_,1) = tmpD1P1;

			MyMatrix tmpD1P2 = D1P_.block(0,1,EFG_BasisNb_,1);
			blasOperator::Blas_Mat_Vec_Mult(Ay_, P_, tmpD1P2,  -1.0, 1.0);  // P' - A'*r
			D1P_.block(0,1,EFG_BasisNb_,1) = tmpD1P2;

			tmpD1P2 = D1P_.block(0,2,EFG_BasisNb_,1);
			blasOperator::Blas_Mat_Vec_Mult(Az_, P_, tmpD1P2,  -1.0, 1.0);  // P' - A'*r
			D1P_.block(0,2,EFG_BasisNb_,1) = tmpD1P2;
		}

		D1P_.block(0,0,EFG_BasisNb_,1) = lltOfA.solve(D1P_.block(0,0,EFG_BasisNb_,1));
		D1P_.block(0,1,EFG_BasisNb_,1) = lltOfA.solve(D1P_.block(0,1,EFG_BasisNb_,1));
		D1P_.block(0,2,EFG_BasisNb_,1) = lltOfA.solve(D1P_.block(0,2,EFG_BasisNb_,1));
		//LaLUSolve(*A_, *D1P_);   // solve for r'

		//  Transpose[N'] = Transpose[r'] * B + Transpose[r] * B'

		blasOperator::Blas_Mat_Trans_Mat_Mult(B_, D1P_, D1N_, 1.0, 0.0);  // N' = Transpose[B] * r'

		
		{   
			// N' = N' + Transpose[B'] * r
			MyMatrix tmpD1N1 = D1N_.block(0,0,InflPtsNb_,1);
			blasOperator::Blas_Mat_Trans_Vec_Mult(Bx_, P_, tmpD1N1, 1.0, 1.0);
			D1N_.block(0,0,InflPtsNb_,1) = tmpD1N1;

			MyMatrix tmpD1N2 = D1N_.block(0,1,InflPtsNb_,1);
			blasOperator::Blas_Mat_Trans_Vec_Mult(By_, P_, tmpD1N2, 1.0, 1.0);
			D1N_.block(0,1,InflPtsNb_,1) = tmpD1N2;

			tmpD1N2 = D1N_.block(0,2,InflPtsNb_,1);
			blasOperator::Blas_Mat_Trans_Vec_Mult(Bz_, P_, tmpD1N2, 1.0, 1.0);
			D1N_.block(0,2,InflPtsNb_,1) = tmpD1N2;
		}

		return ;
		
	}

	void Cell::ApproxAtPoint(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate,bool isCouple)
	{
		const MyPoint& pX = gaussPointInGlobalCoordinate;
		MyVector& N_ =  shapeFunctionInEFG_8_27[gaussIdx];
		MyMatrix& D1N_ = shapeDerivativeValue_8_27_3[gaussIdx];

		MyFloat WI;         // Value of weight function at node I
		MyFloat WxI, WyI, WzI;   // Value of derivatives of weight function at node I
		MyFloat WxxI, WxyI, WyyI, WzzI, WxzI, WyzI;

		if (isCouple)
		{
			InfluentPoints4Couple(gaussIdx,gaussPointInGlobalCoordinate);
		}
		else
		{
			InfluentPoints(gaussIdx,gaussPointInGlobalCoordinate);
		}		

		const unsigned InflPtsNb_ = gaussPointInfluncePointCount_8[gaussIdx];
		s_map_InfluncePointSize[InflPtsNb_] = true;
		MyMatrix  A_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ax_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ay_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Az_(EFG_BasisNb_,EFG_BasisNb_);

		MyMatrix  Axx_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ayy_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Azz_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Axy_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Axz_(EFG_BasisNb_,EFG_BasisNb_);
		MyMatrix  Ayz_(EFG_BasisNb_,EFG_BasisNb_);

		MyMatrix  B_(EFG_BasisNb_,InflPtsNb_);   // 当前点的B矩阵
		MyMatrix  Bx_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对x的偏导数
		MyMatrix  By_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对y的偏导数
		MyMatrix  Bz_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对z的偏导数

		MyMatrix  Bxx_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xx的偏导数
		MyMatrix  Byy_(EFG_BasisNb_,InflPtsNb_); // B矩阵对yy的偏导数
		MyMatrix  Bzz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对zz的偏导数
		MyMatrix  Bxy_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数
		MyMatrix  Bxz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数
		MyMatrix  Byz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数

		MyVector			P_(EFG_BasisNb_);
		MyMatrix            D1P_(EFG_BasisNb_,dim);

		N_.resize(InflPtsNb_);
		D1N_.resize(InflPtsNb_,dim);

		MyDenseVector Radius_;
		Radius_[0] = SupportSize;Radius_[1] = SupportSize;Radius_[2] = SupportSize;

		shapeFunctionInEFG_8_27[gaussIdx].resize(InflPtsNb_);
		shapeDerivativeValue_8_27_3[gaussIdx].resize(InflPtsNb_,dim);

		{
			A_.setZero();
			Ax_.setZero();
			Ay_.setZero();
			Az_.setZero();
			Axx_.setZero();
			Ayy_.setZero();
			Azz_.setZero();
			Axy_.setZero();
			Axz_.setZero();
			Ayz_.setZero();

			B_.setZero();
			Bx_.setZero();
			By_.setZero();
			Bz_.setZero();
			Bxx_.setZero();
			Byy_.setZero();
			Bzz_.setZero();
			Bxy_.setZero();
			Bxz_.setZero();
			Byz_.setZero();
			P_.setZero();
			N_.setZero();
			D1P_.setZero();
		}
		
		std::vector< VertexPtr >& refCurGaussPointInfls = InflPts_8_27[gaussIdx];
		Q_ASSERT(refCurGaussPointInfls.size() == InflPtsNb_);
		for (unsigned i=0; i<InflPtsNb_; i++)
		{
			VertexPtr curInflsPoint = refCurGaussPointInfls[i];
			MyPoint pts_j = curInflsPoint->getPos();

			//     A  = Sum of WI(X)*P(XI)*Transpose[P(XI)], I=1,N

			basis(pX,curInflsPoint->getPos(), P_);
			
			WI = WeightFun(pts_j, Radius_, pX, 0);
			

			//  (*A_) = (*A_) + WI * (*P_) * Transpose [ (*P_) ]
			MyFloat cof = ( i == 0 ? 0:1 );
			blasOperator::Blas_R1_Update(A_, P_, WI, cof);
			

			{
				WxI = WeightFun(pts_j, Radius_, pX, 1);
				WyI = WeightFun(pts_j, Radius_, pX, 2);
				WzI = WeightFun(pts_j, Radius_, pX, 3);

				//  (*Ax_) = (*Ax_) + WxI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ax_, P_, WxI, cof);

				//  (*Ay_) = (*Ay_) + WyI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ay_, P_, WyI, cof);
				blasOperator::Blas_R1_Update(Az_, P_, WzI, cof);
			}

			{
				WxxI = WeightFun(pts_j, Radius_, pX, 4);
				WyyI = WeightFun(pts_j, Radius_, pX, 5);
				WzzI = WeightFun(pts_j, Radius_, pX, 6);
				WxyI = WeightFun(pts_j, Radius_, pX, 7);
				WxzI = WeightFun(pts_j, Radius_, pX, 8);
				WyzI = WeightFun(pts_j, Radius_, pX, 9);
				//  (*Axx_) = (*Axx_) + WxxI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Axx_, P_, WxxI, cof);
				//  (*Ayy_) = (*Ayy_) + WyyI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ayy_, P_, WyyI, cof);
				blasOperator::Blas_R1_Update(Azz_, P_, WzzI, cof);
				//  (*Axy_) = (*Axy_) + WxyI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Axy_, P_, WxyI, cof);
				blasOperator::Blas_R1_Update(Axz_, P_, WxzI, cof);
				blasOperator::Blas_R1_Update(Ayz_, P_, WyzI, cof);
			}

			for ( unsigned k=0; k< EFG_BasisNb_; k++ )
			{
				//(*B_)(k, i) = (*P_)(k) * WI;
				B_.coeffRef(k,i) = P_(k) * WI;
			}
			

			{
				for(unsigned k = 0; k < EFG_BasisNb_; k++ )
				{
					/*(*Bx_)(k, i) = (*P_)(k) * WxI;
					(*By_)(k, i) = (*P_)(k) * WyI;
					(*Bz_)(k, i) = (*P_)(k) * WzI;*/

					Bx_.coeffRef(k, i) = (P_)(k) * WxI;
					By_.coeffRef(k, i) = (P_)(k) * WyI;
					Bz_.coeffRef(k, i) = (P_)(k) * WzI;
				}
			}

			
			{	
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Bxx_).coeffRef(k, i) = (P_)(k) * WxxI;
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Byy_).coeffRef(k, i) = (P_)(k) * WyyI;
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Bzz_).coeffRef(k, i) = (P_)(k) * WzzI;
				for( int k = 0; k< EFG_BasisNb_; k++ )
					(Bxy_).coeffRef(k, i) = (P_)(k) * WxyI;
				for(int  k = 0; k< EFG_BasisNb_; k++ )
					(Bxz_).coeffRef(k, i) = (P_)(k) * WxzI;
				for(int  k = 0; k< EFG_BasisNb_; k++ )
					(Byz_).coeffRef(k, i) = (P_)(k) * WyzI;
			}
		}

		Eigen::LLT<MyMatrix> lltOfA(A_);//LaLUFactor(*A_);

		//  r = Inverse[A] * P

		basis(pX, pX, P_);
		P_ = lltOfA.solve(P_);//LaLUSolve(*A_, *P_);  // r is stored in P_

		//  Transpose[N] = Transpose[r] * B
		blasOperator::Blas_Mat_Trans_Vec_Mult(B_, P_, N_, 1.0, 0.0);

		dbasis(pX, pX, D1P_);

		//  r is saved in P, and r' in D1P_
		
		{
			MyMatrix tmpD1P1 = D1P_.block(0,0,EFG_BasisNb_,1);
			blasOperator::Blas_Mat_Vec_Mult(Ax_, P_, tmpD1P1,  -1.0, 1.0);  // P' - A'*r
			D1P_.block(0,0,EFG_BasisNb_,1) = tmpD1P1;

			MyMatrix tmpD1P2 = D1P_.block(0,1,EFG_BasisNb_,1);
			blasOperator::Blas_Mat_Vec_Mult(Ay_, P_, tmpD1P2,  -1.0, 1.0);  // P' - A'*r
			D1P_.block(0,1,EFG_BasisNb_,1) = tmpD1P2;

			tmpD1P2 = D1P_.block(0,2,EFG_BasisNb_,1);
			blasOperator::Blas_Mat_Vec_Mult(Az_, P_, tmpD1P2,  -1.0, 1.0);  // P' - A'*r
			D1P_.block(0,2,EFG_BasisNb_,1) = tmpD1P2;
		}

		D1P_.block(0,0,EFG_BasisNb_,1) = lltOfA.solve(D1P_.block(0,0,EFG_BasisNb_,1));
		D1P_.block(0,1,EFG_BasisNb_,1) = lltOfA.solve(D1P_.block(0,1,EFG_BasisNb_,1));
		D1P_.block(0,2,EFG_BasisNb_,1) = lltOfA.solve(D1P_.block(0,2,EFG_BasisNb_,1));
		//LaLUSolve(*A_, *D1P_);   // solve for r'

		//  Transpose[N'] = Transpose[r'] * B + Transpose[r] * B'

		blasOperator::Blas_Mat_Trans_Mat_Mult(B_, D1P_, D1N_, 1.0, 0.0);  // N' = Transpose[B] * r'

		
		{   
			// N' = N' + Transpose[B'] * r
			MyMatrix tmpD1N1 = D1N_.block(0,0,InflPtsNb_,1);
			blasOperator::Blas_Mat_Trans_Vec_Mult(Bx_, P_, tmpD1N1, 1.0, 1.0);
			D1N_.block(0,0,InflPtsNb_,1) = tmpD1N1;

			MyMatrix tmpD1N2 = D1N_.block(0,1,InflPtsNb_,1);
			blasOperator::Blas_Mat_Trans_Vec_Mult(By_, P_, tmpD1N2, 1.0, 1.0);
			D1N_.block(0,1,InflPtsNb_,1) = tmpD1N2;

			tmpD1N2 = D1N_.block(0,2,InflPtsNb_,1);
			blasOperator::Blas_Mat_Trans_Vec_Mult(Bz_, P_, tmpD1N2, 1.0, 1.0);
			D1N_.block(0,2,InflPtsNb_,1) = tmpD1N2;
		}

		return ;
	}

	void Cell::compressMassMatrix(const MyMatrix& objMatrix,std::vector<int> & vecDofs)
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_Mass);
	}

	void Cell::compressStiffnessMatrix(const MyMatrix& objMatrix,std::vector<int> & vecDofs)
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_Stiffness);
	}

	void Cell::compressMatrix(const MyMatrix& objMatrix,std::vector<int> & vecDofs,std::map<long,std::map<long,TripletNode > >& TripletNodeMap)
	{
		for (unsigned r=0;r<objMatrix.rows();++r)
		{
			for (unsigned c=0;c<objMatrix.cols();++c)
			{
				const MyFloat val = objMatrix.coeff(r,c);
				if (!numbers::isZero(val))
				{
					TripletNodeMap[vecDofs[r]][vecDofs[c]].val += val;
				}
			}
		}
	}

	void Cell::compressRHS(const MyVector& rhs,std::vector<int> &vecDofs)
	{
		for (unsigned r=0;r<rhs.size();++r)
		{
			const MyFloat val = rhs[r];
			if (!numbers::isZero(val))
			{
				m_TripletNode_Rhs[vecDofs[r]].val += val;
			}
		}
	}

	MyFloat Cell::materialDistance(const MyPoint& gaussPoint, const MyPoint& samplePoint)
	{
		static MyPoint ret;
		if ( CheckLineTri(gaussPoint,samplePoint,MyPoint(11.8*CellRaidus,4*CellRaidus,200*CellRaidus),
												 MyPoint(11.8*CellRaidus,4*CellRaidus,-200*CellRaidus),
												 MyPoint(11.8*CellRaidus,200*CellRaidus,0),ret) )
		{
			return 10000000.f;
		}
		else
		{
			return 1.f;
		}
	}

	bool Cell::CheckLineTri( const MyPoint &L1, const MyPoint &L2, const MyPoint &PV1, const MyPoint &PV2, const MyPoint &PV3, MyPoint &HitP )
	{
		MyPoint VIntersect;

		// Find Triangle Normal, would be quicker to have these computed already
		MyPoint VNorm;
		VNorm = ( PV2 - PV1 ).cross( PV3 - PV1 );
		//printf("tmpVec3(%f,%f,%f)\n",VNorm.x,VNorm.y,VNorm.z);
		VNorm.normalize();
		//printf("VNorm(%f,%f,%f)\n",VNorm.x,VNorm.y,VNorm.z);

		// Find distance from L1 and L2 to the plane defined by the triangle
		float fDst1 = (L1-PV1).dot( VNorm );
		//printf("fDst1 is %f \n",fDst1);
		float fDst2 = (L2-PV1).dot( VNorm );
		//printf("fDst2 is %f \n",fDst2);

		if ( (fDst1 * fDst2) >= 0.0f) return false;  // line doesn't cross the triangle.
		if ( fDst1 == fDst2) {return false;} // line and plane are parallel

		// Find point on the line that intersects with the plane
		VIntersect = L1 + (L2-L1) * ( -fDst1/(fDst2-fDst1) );
		//printf("VIntersect(%f,%f,%f)\n",VIntersect.x,VIntersect.y,VIntersect.z);

		// Find if the interesection point lies inside the triangle by testing it against all edges
		MyPoint VTest;
		VTest = VNorm.cross( PV2-PV1 );
		//printf("VTest(%f,%f,%f)\n",VTest.x,VTest.y,VTest.z);
		if ( VTest.dot( VIntersect-PV1 ) < 0.0f ) return false;
		VTest = VNorm.cross( PV3-PV2 );
		//printf("VTest(%f,%f,%f)\n",VTest.x,VTest.y,VTest.z);
		if ( VTest.dot( VIntersect-PV2 ) < 0.0f ) return false;
		VTest = VNorm.cross( PV1-PV3 );
		//printf("VTest(%f,%f,%f)\n",VTest.x,VTest.y,VTest.z);
		if ( VTest.dot( VIntersect-PV1 ) < 0.0f ) return false;

		HitP = VIntersect;

		return true;
	}
	void Cell::computeCellType_Steak()
	{
		int nFEMVertexCount = 0;
		int nEFGVertexCount = 0;
		for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
		{
			//const MyFloat curX = m_elem_vertex[i]->getPos().y();//Take care of rotation 90
			//const MyFloat curY = m_elem_vertex[i]->getPos().x();
			const MyFloat curX = ((m_elem_vertex[i]->getPos().y()) ) ;//Take care of rotation 90
			const MyFloat curY = ((m_elem_vertex[i]->getPos().x()) ) ;

#if 1  //for debug

			if ( (curY < 0.85f && curY > 0.7f && curX < 0.9f && curX > 0.1f) || (curY < 0.85f && curY > 0.06f && curX>0.45f && curX < 0.55f) )
			{
				nFEMVertexCount++;
				//m_elem_vertex[i]->setFromDomainId(m_nDomainId);
			}
			else if (curX < 0.35f && curY < 0.6f)
			{
				nFEMVertexCount++;
				//m_elem_vertex[i]->setFromDomainId(m_nDomainId);
			}
			else if (curX > 0.65f && curY < 0.6f)
			{
				nFEMVertexCount++;
				//m_elem_vertex[i]->setFromDomainId(m_nDomainId);
			}
			else if (curY > 0.92f)
			{
				nFEMVertexCount++;
				//m_elem_vertex[i]->setFromDomainId(m_nDomainId);
			}
			else
#endif
			{
				nEFGVertexCount++;
				//m_elem_vertex[i]->setFromDomainId(m_nDomainId);
			}
			/*nFEMVertexCount++;
			m_nDomainId = 0;
			m_elem_vertex[i]->setFromDomainId(m_nDomainId);*/
		}


		if (Geometry::vertexs_per_cell == nFEMVertexCount)
		{
			m_CellType = FEM;
			s_nFEM_Cell_Count++;			
		}
		else if (Geometry::vertexs_per_cell == nEFGVertexCount)
		{
			m_CellType = EFG;
			s_nEFG_Cell_Count++;
		}
		else
		{
			m_CellType = COUPLE;
			s_nCOUPLE_Cell_Count++;
		}

		const MyFloat curX = getCenterPoint().y();
		const MyFloat curY = getCenterPoint().x();
		if ( (curY < 0.85f && curY > 0.7f && curX < 0.9f && curX > 0.1f) || (curY < 0.85f && curY > 0.06f && curX>0.45f && curX < 0.55f) )
		{
			m_nDomainId = 0;
			m_nCoupleId = Invalid_Id;
		}
		else if (curX < 0.35f && curY < 0.6f)
		{
			m_nDomainId = 1;
			m_nCoupleId = Invalid_Id;
		}
		else if (curX > 0.65f && curY < 0.6f)
		{
			m_nDomainId = 2;
			m_nCoupleId = Invalid_Id;
		}
		else if (curY > 0.92f)
		{
			m_nDomainId = 3;
			m_nCoupleId = Invalid_Id;
		}
		else
		{
			m_nDomainId = CoupleDomainId;
			
			if (curY > 0.75f)
			{
				m_nCoupleId = 0;
			}
			else
			{
				if (curX < 0.5f)
				{
					m_nCoupleId = 1;
				}
				else
				{
					m_nCoupleId = 2;
				}
			}
		}

		for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
		{
			VertexPtr curVtxPtr = getVertex(i);
			if (m_nDomainId != Invalid_Id && m_nDomainId != CoupleDomainId)
			{
				curVtxPtr->setTmpLocalDomainId(m_nDomainId);
			}
			if (m_nDomainId == CoupleDomainId)
			{
				curVtxPtr->setTmpCoupleDomainId(m_nCoupleId);
			}
		}
	}

	void Cell::computeCellType_Beam()
	{
		const MyFloat curX = getCenterPoint().x();
		const MyFloat curY = getCenterPoint().y();

		MyInt nTmp = curX / (2*CellRaidus);

		if (nTmp < 6)
		{
			m_nDomainId = 0;
			m_nCoupleId = Invalid_Id;
		}
		else if (nTmp > 13)
		{
			m_nDomainId = 1;
			m_nCoupleId = Invalid_Id;
		}
		else
		{
			m_nDomainId = CoupleDomainId;
			m_nCoupleId = 0;
			/*if (nTmp < 10)
			{
				m_nCoupleId = 0;
			}
			else
			{
				m_nCoupleId = 1;
			}*/
		}

		for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
		{
			VertexPtr curVtxPtr = getVertex(i);
			if (m_nDomainId != Invalid_Id && m_nDomainId != CoupleDomainId)
			{
				curVtxPtr->setTmpLocalDomainId(m_nDomainId);
			}
			if (m_nDomainId == CoupleDomainId)
			{
				curVtxPtr->setTmpCoupleDomainId(m_nCoupleId);
			}
		}
	}

	void Cell::computeCellType(const std::vector< Plane >& vecPlane)
	{
		Q_ASSERT(2 == vecPlane.size());
		static unsigned linePairs[12][2] = {{0,1},{4,5},{6,7},{2,3},
											{0,2},{1,3},{5,7},{4,6},
											{0,4},{1,5},{3,7},{2,6}};

		

		std::vector< MyFloat > vecFlags(vecPlane.size());
		std::vector< std::pair<MyPoint,MyPoint> > vecLines;
		for (unsigned i=0;i<sizeof(linePairs)/sizeof(linePairs[0]);++i)
		{
			vecLines.push_back(std::make_pair(m_elem_vertex[linePairs[i][0]]->getPos(),
											  m_elem_vertex[linePairs[i][1]]->getPos()) );
		}

		bool bIntersect=false;
		int nIntersectPlaneId=-1;
		for (unsigned i=0;i<vecPlane.size();++i)
		{
			if (vecPlane[i].checkIntersect(vecLines))
			{
				bIntersect = true;
				nIntersectPlaneId = i;
				break;
			}
		}

		if (bIntersect)
		{	
			m_CellType = COUPLE;
		}
		else
		{
			if ( 0 < vecPlane[0].checkPointPlane(getCenterPoint()) )
			{
				if (0 < vecPlane[1].checkPointPlane(getCenterPoint()))
				{
					m_CellType = FEM;
					m_nDomainId = 4;
				}
				else
				{
					m_CellType = EFG;
					m_nDomainId = 2;
				}
				
			}
			else
			{
				m_CellType = FEM;
				m_nDomainId = 0;
			}
		}

		if (FEM == m_CellType)
		{
			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				m_elem_vertex[i]->setFromDomainId(0);
			}
		}
		else if (COUPLE == m_CellType)
		{
			
			if (0 == nIntersectPlaneId)
			{
				m_nDomainId = 1;
			}
			else
			{
				m_nDomainId = 3;
			}
		}
		else if (EFG == m_CellType)
		{			
			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				m_elem_vertex[i]->setFromDomainId(2);
			}
		}
	}

	void Cell::computeRotationMatrix(MyVector & global_incremental_displacement)
	{
		incremental_displacement.setZero();
		for (unsigned i=0,localDof=0;i<Geometry::vertexs_per_cell;++i,localDof+=3)
		{
			MyVectorI& curDofs = m_elem_vertex[i]->getDofs();
			incremental_displacement[localDof+0] = global_incremental_displacement[curDofs[0]];
			incremental_displacement[localDof+1] = global_incremental_displacement[curDofs[1]];
			incremental_displacement[localDof+2] = global_incremental_displacement[curDofs[2]];
		}

		static MyDenseVector Signs[Geometry::vertexs_per_cell] = {
																	MyDenseVector(-1,-1,-1), 
																	MyDenseVector(1,-1,-1),
																	MyDenseVector(-1,1,-1),
																	MyDenseVector(1,1,-1),
																	MyDenseVector(-1,-1,1),
																	MyDenseVector(1,-1,1),
																	MyDenseVector(-1,1,1),
																	MyDenseVector(1,1,1)};
			MyMatrix_3X3 tmp;
			tmp.setZero();
			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				tmp += incremental_displacement.block(i*3,0,3,1) * Signs[i].transpose();
			}
			RotationMatrix = Matrix<MyFloat, 3, 3>::Identity();
			RotationMatrix += 0.25f * (1.f / (m_radius * 2)) * tmp;

			for (unsigned i=1;i<5;++i)
			{
				RotationMatrix = 0.5f*(RotationMatrix + RotationMatrix.inverse().transpose());
			}

			
			/*RotationMatrix_24_24.setZero();

			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				RotationMatrix_24_24.block(i*3,i*3,3,3) = RotationMatrix;
			}

			StiffnessMatrix_24_24_Corotation = RotationMatrix_24_24 * StiffnessMatrix_24_24 * RotationMatrix_24_24.transpose();
			RightHandValue_24_1_Corotation = RotationMatrix_24_24 * StiffnessMatrix_24_24 * (RotationMatrix_24_24.transpose()*Pj - Pj);*/
	}

	void Cell::assembleRotationMatrix()
	{
		std::vector<int> vecDofs;
		if (FEM == m_CellType)
		{
			Cell_Corotation_Matrix.resize(Geometry::dofs_per_cell,Geometry::dofs_per_cell);
			Cell_Corotation_Matrix.setZero();

			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				Cell_Corotation_Matrix.block(i*3,i*3,3,3) = RotationMatrix;
			}

			get_dof_indices(vecDofs);
			compressStiffnessMatrix(Cell_Corotation_Matrix*StiffnessMatrix_24_24*Cell_Corotation_Matrix.transpose(),vecDofs);

			compressRHS(Cell_Corotation_Matrix * StiffnessMatrix_24_24 * (Cell_Corotation_Matrix.transpose()*Pj_FEM - Pj_FEM),vecDofs);
		}
		else if (COUPLE == m_CellType)
		{
			for (unsigned ls=0;ls < Geometry::gauss_Sample_Point;++ls)
			{
				const unsigned NPTS = InflPts_8_27[ls].size();
				const MyMatrix& stiffnessMatrix = StiffnessMatrix_81_81[ls];

				Cell_Corotation_Matrix.resize(NPTS*3,NPTS*3);
				Cell_Corotation_Matrix.setZero();

				for (unsigned i=0;i<NPTS;++i)
				{
					Cell_Corotation_Matrix.block(i*3,i*3,3,3) = RotationMatrix;
				}

				get_dof_indices(ls,vecDofs);
				compressStiffnessMatrix(/*stiffnessMatrix*/Cell_Corotation_Matrix*stiffnessMatrix*Cell_Corotation_Matrix.transpose(),vecDofs);
				compressRHS(Cell_Corotation_Matrix * stiffnessMatrix * (Cell_Corotation_Matrix.transpose()*Pj_EFG[ls] - Pj_EFG[ls]),vecDofs);
			}
		}
		else if (EFG == m_CellType)
		{
			for (unsigned ls=0;ls < Geometry::gauss_Sample_Point;++ls)
			{
				const unsigned NPTS = InflPts_8_27[ls].size();
				const MyMatrix& stiffnessMatrix = StiffnessMatrix_81_81[ls];

				Cell_Corotation_Matrix.resize(NPTS*3,NPTS*3);
				Cell_Corotation_Matrix.setZero();

				for (unsigned i=0;i<NPTS;++i)
				{
					Cell_Corotation_Matrix.block(i*3,i*3,3,3) = RotationMatrix;
				}

				get_dof_indices(ls,vecDofs);
				compressStiffnessMatrix(/*stiffnessMatrix*/Cell_Corotation_Matrix*stiffnessMatrix*Cell_Corotation_Matrix.transpose(),vecDofs);
				compressRHS(Cell_Corotation_Matrix * stiffnessMatrix * (Cell_Corotation_Matrix.transpose()*Pj_EFG[ls] - Pj_EFG[ls]),vecDofs);

				Cell_Corotation_Matrix.resize(0,0);

			}
		}
	}

}