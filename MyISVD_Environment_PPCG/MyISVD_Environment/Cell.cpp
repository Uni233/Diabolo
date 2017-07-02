#include "Cell.h"
#include <iostream>
#include <algorithm>
#include <iterator>
#include <fstream>
namespace YC
{
	/************************************************************************/
	/* static reference                                                     */
	/************************************************************************/
	std::vector< CellPtr > Cell::s_Cell_Cache;
	int Cell::s_nFEM_Cell_Count = MyZero;
	int Cell::s_nEFG_Cell_Count = MyZero;
	int Cell::s_nCOUPLE_Cell_Count = MyZero;
	Cell::Assemble_State Cell::m_global_state;
	MyPoint Cell::GaussVertex[Geometry::vertexs_per_cell];
	MyFloat   Cell::GaussVertexWeight[Geometry::vertexs_per_cell];
	MaterialMatrix Cell::s_MaterialMatrix;
	MyDenseVector Cell::externalForce(0.f,Material::GravityFactor * Cell::scaleExternalForce * Material::Density,0.f);
	MyFloat Cell::scaleExternalForce(50.f);
	std::vector< Cell::tuple_matrix > Cell::vec_cell_stiffness_matrix;
	std::vector< Cell::tuple_matrix > Cell::vec_cell_mass_matrix;
	std::vector< Cell::tuple_vector > Cell::vec_cell_rhs_matrix;

	/************************************************************************/
	/* function definition                                                  */
	/************************************************************************/
	Cell::Cell(MyPoint center, MyFloat radius, VertexPtr vertexes[])
		:m_center(center),
		 m_radius(radius),
		 m_CellType(INVALIDTYPE),
		 m_nID(Invalid_Id),
		 m_nRhsIdx(Invalid_Id),
		 m_nMassMatrixIdx(Invalid_Id),
		 m_nStiffnessMatrixIdx(Invalid_Id)
	{
		m_CellType = FEM;
		for (unsigned i=0; i < Geometry::vertexs_per_cell; ++i)
		{
			m_elem_vertex[i] = vertexes[i];
		}

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

		
#if USE_MultidomainIndependent
		m_nDomainId = Invalid_Id;
#endif
	}


	Cell::~Cell(void)
	{
	}

#if USE_MultidomainIndependent
	std::map<long, std::map<long, Cell::TripletNode > >  Cell::s_TripletNode_LocalMass_DMI[LocalDomainCount];
	std::map<long, std::map<long, Cell::TripletNode > >  Cell::s_TripletNode_LocalStiffness_DMI[LocalDomainCount];
	std::map<long, Cell::TripletNode >				    Cell::s_TripletNode_LocalRhs_DMI[LocalDomainCount];
	/*std::map<long, std::map<long, Cell::TripletNode > > Cell::s_TripletNode_ModalWarp_DIM[LocalDomainCount];*/
	MyMatrix Cell::MaterialMatrix_6_5[LocalDomainCount];
	MyBOOL   Cell::bMaterialMatrixInitial[LocalDomainCount];

	void Cell::TestModalWrapMatrix_DMI(const MyVector& globlaDisp, MyVector& w, MyVector& translation, MyVector& incremental_displacement)
	{
		incremental_displacement.resize(Geometry::dofs_per_cell);
		//MyVector incremental_displacement(Geometry::dofs_per_cell);
		for (unsigned i = 0; i < Geometry::vertexs_per_cell; ++i)
		{
			MyVectorI dofs = m_elem_vertex[i]->getDofs_DMI_Global();

			incremental_displacement[3 * i + 0] = globlaDisp[dofs[0]];
			incremental_displacement[3 * i + 1] = globlaDisp[dofs[1]];
			incremental_displacement[3 * i + 2] = globlaDisp[dofs[2]];

			//std::cout << "# " << i << " : "<< dofs.transpose() << std::endl;
		}


		w = m_W_24_24 * incremental_displacement;

		MyVector nativePos(Geometry::dofs_per_cell);
		for (unsigned i = 0; i < Geometry::vertexs_per_cell; ++i)
		{
			//MyVectorI dofs = m_elem_vertex[i]->getGlobalDofs();
			MyDenseVector pos = m_elem_vertex[i]->getPos();
			nativePos[3 * i + 0] = pos[0];
			nativePos[3 * i + 1] = pos[1];
			nativePos[3 * i + 2] = pos[2];
		}
		MyMatrix shp(Geometry::dofs_per_cell, Geometry::dofs_per_cell); shp.setZero();
		for (unsigned q = 0; q < Geometry::vertexs_per_cell; ++q)
		{
			MyMatrix tmpShapeMatrix(3, 24);
			tmpShapeMatrix.setZero();
			for (unsigned k = 0; k < Geometry::shape_Function_Count_In_FEM; ++k)
			{
				const unsigned col = k*MyDIM;
				for (unsigned i = 0; i < MyDIM; ++i)
				{
					tmpShapeMatrix.coeffRef(0, col + 0) = shapeFunctionValue_8_8.coeff(q, k);
					tmpShapeMatrix.coeffRef(1, col + 1) = shapeFunctionValue_8_8.coeff(q, k);
					tmpShapeMatrix.coeffRef(2, col + 2) = shapeFunctionValue_8_8.coeff(q, k);
				}
			}

			shp.block(3 * q, 0, 3, 24) = tmpShapeMatrix;
		}
		
		translation = shp * incremental_displacement + nativePos;
	}

	void Cell::computeCellType_DMI(const int did)
	{
		m_CellType = FEM;
		m_nDomainId = did;
		s_nFEM_Cell_Count++;
		return;
	}

	CellPtr Cell::makeCell_DMI(MyPoint point, MyFloat radius, const int did)
	{
		std::vector< CellPtr >::reverse_iterator itr = std::find_if(s_Cell_Cache.rbegin(), s_Cell_Cache.rend(), CellCompare(point[0], point[1], point[2]));
		if (s_Cell_Cache.rend() == itr)
		{
			//no find
			VertexPtr vertexes[Geometry::vertexs_per_cell];
			Vertex::makeCellVertex_DMI(point, radius, vertexes, did);

			s_Cell_Cache.push_back(CellPtr(new Cell(point, radius, vertexes)));
			s_Cell_Cache[s_Cell_Cache.size() - 1]->setId(s_Cell_Cache.size() - 1);
#if USE_MODAL_WARP
			for (int v = 0; v < 8; ++v)
			{
				vertexes[v]->m_vec_ShareCell.push_back(s_Cell_Cache[s_Cell_Cache.size() - 1]);
			}
#endif//USE_MODAL_WARP

			return s_Cell_Cache[s_Cell_Cache.size() - 1];
		}
		else
		{
			//find it
			Q_ASSERT(false);
			return (*itr);
		}
	}

	void Cell::makeMaterialMatrix(const int nDomainId)
	{
		Q_ASSERT((nDomainId >= 0) && (nDomainId < LocalDomainCount));
		if (bMaterialMatrixInitial[nDomainId].getBool())
		{
			return;
		}
		else
		{
			bMaterialMatrixInitial[nDomainId].setBool(true);
		}

		MyMatrix& MaterialMatrix_6_6 = MaterialMatrix_6_5[nDomainId];
		MaterialMatrix_6_6.resize(MaterialMatrixSize, MaterialMatrixSize);
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

		MyFloat G = E / (2 * (1 + mu));
		MyFloat lai = mu*E / ((1 + mu)*(1 - 2 * mu));


		MaterialMatrix_6_6.coeffRef(0, 0) = lai + 2 * G;

		MaterialMatrix_6_6.coeffRef(1, 0) = lai;
		MaterialMatrix_6_6.coeffRef(1, 1) = lai + 2 * G;

		MaterialMatrix_6_6.coeffRef(2, 0) = lai;
		MaterialMatrix_6_6.coeffRef(2, 1) = lai;
		MaterialMatrix_6_6.coeffRef(2, 2) = lai + 2 * G;

		MaterialMatrix_6_6.coeffRef(3, 0) = 0.0;
		MaterialMatrix_6_6.coeffRef(3, 1) = 0.0;
		MaterialMatrix_6_6.coeffRef(3, 2) = 0, 0;
		MaterialMatrix_6_6.coeffRef(3, 3) = G;

		MaterialMatrix_6_6.coeffRef(4, 0) = 0.0;
		MaterialMatrix_6_6.coeffRef(4, 1) = 0.0;
		MaterialMatrix_6_6.coeffRef(4, 2) = 0.0;
		MaterialMatrix_6_6.coeffRef(4, 3) = 0.0;
		MaterialMatrix_6_6.coeffRef(4, 4) = G;

		MaterialMatrix_6_6.coeffRef(5, 0) = 0.0;
		MaterialMatrix_6_6.coeffRef(5, 1) = 0.0;
		MaterialMatrix_6_6.coeffRef(5, 2) = 0.0;
		MaterialMatrix_6_6.coeffRef(5, 3) = 0.0;
		MaterialMatrix_6_6.coeffRef(5, 4) = 0.0;
		MaterialMatrix_6_6.coeffRef(5, 5) = G;

		makeSymmetry(MaterialMatrix_6_6);

		//std::cout << MaterialMatrix_6_6 << std::endl;
	}

	void Cell::makeSymmetry(MyMatrix& objMatrix)
	{
		for (unsigned r = 0; r<objMatrix.rows(); ++r)
		{
			for (unsigned c = 0; c<r; ++c)
			{
				objMatrix.coeffRef(c, r) = objMatrix.coeff(r, c);
			}
		}
	}

	void Cell::initialize_DMI()
	{
		m_CellType = FEM;
		m_Polynomials = Polynomial::generate_complete_basis(1);
		computeGaussPoint();
		computeShapeFunction();
		computeJxW();
		computeShapeGrad();

		makeMaterialMatrix(m_nDomainId);
		makeStiffnessMatrix();
		makeMassMatrix_Lumping();
		makeLocalRhs();


#if USE_MODAL_WARP
		createW_24_24();
#endif
	}

	void Cell::get_dof_indices_Local_DMI(std::vector<int> &vecDofs)
	{
		vecDofs.clear();
		if (FEM == m_CellType)
		{
			for (int i = 0; i < Geometry::vertexs_per_cell; ++i)
			{
				MyVectorI& ref_Dof = m_elem_vertex[i]->getDofs_DMI();
				vecDofs.push_back(ref_Dof[0]);
				vecDofs.push_back(ref_Dof[1]);
				vecDofs.push_back(ref_Dof[2]);
			}
		}
	}

	void Cell::assembleSystemMatrix_DMI()
	{
		std::vector<int> vecDofs;

		get_dof_indices_Local_DMI(vecDofs);


		compressLocalFEMStiffnessMatrix_DMI(vec_cell_stiffness_matrix[m_nStiffnessMatrixIdx].matrix, vecDofs);

		compressLocalFEMMassMatrix_DMI(vec_cell_mass_matrix[m_nMassMatrixIdx].matrix, vecDofs);

		compressLocalFEMRHS_DMI(vec_cell_rhs_matrix[m_nRhsIdx].vec, vecDofs);
	}

	void Cell::compressLocalFEMStiffnessMatrix_DMI(const MyMatrix& objMatrix, std::vector<int> &vecDofs)
	{
		compressMatrix(objMatrix, vecDofs, Cell::s_TripletNode_LocalStiffness_DMI[m_nDomainId]);
	}

	void Cell::compressLocalFEMMassMatrix_DMI(const MyMatrix& objMatrix, std::vector<int> &vecDofs)
	{
		compressMatrix(objMatrix, vecDofs, Cell::s_TripletNode_LocalMass_DMI[m_nDomainId]);
	}

	void Cell::compressLocalFEMRHS_DMI(const MyVector& rhs, std::vector<int> &vecDofs)
	{
		for (unsigned r = 0; r < rhs.size(); ++r)
		{
			const MyFloat val = rhs[r];
			if (!numbers::isZero(val))
			{
				Cell::s_TripletNode_LocalRhs_DMI[m_nDomainId][vecDofs[r]].val += val;
			}
		}
	}
#endif//USE_MultidomainIndependent

	void Cell::computeCellType_Global()
	{
		m_CellType = FEM;
		s_nFEM_Cell_Count++;
		return ;
	}

	CellPtr Cell::makeCell(MyPoint point, MyFloat radius)
	{
		std::vector< CellPtr >::reverse_iterator itr = std::find_if(s_Cell_Cache.rbegin(),s_Cell_Cache.rend(),CellCompare(point[0],point[1],point[2]));
		if ( s_Cell_Cache.rend() == itr )
		{
			//no find
			VertexPtr vertexes[Geometry::vertexs_per_cell];
			Vertex::makeCellVertex(point,radius,vertexes);

			s_Cell_Cache.push_back( CellPtr(new Cell(point,radius,vertexes)) );
			s_Cell_Cache[s_Cell_Cache.size()-1]->setId(s_Cell_Cache.size()-1);

			for (int v=0;v<8;++v)
			{
				vertexes[v]->m_vec_ShareCell.push_back(s_Cell_Cache[s_Cell_Cache.size()-1]);
			}
			return s_Cell_Cache[s_Cell_Cache.size()-1];
		}
		else
		{
			//find it
			//Q_ASSERT(false);
			//printf("{%f,%f,%f,%f}\n",point[0],point[1],point[2],radius);
			return (*itr);
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
			double z = std::cos(numbers::MyPI * (i-.25)/(n_Quads+.5));


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

			}
			while (std::fabs(p1/pp) > tolerance);

			double x = .5*z;
			double w = 1./((1.-z*z)*pp*pp);
			gauss[0]=.5-x;
			gauss[1]=.5+x;
			makeGaussPoint(&gauss[0],w*w*w);

		}
	}

	void Cell::makeGaussPoint(double gauss[2],double w)
	{
		

		for (unsigned int idx = 0; idx <Geometry::vertexs_per_cell;++idx)
		{
			GaussVertex[idx] = MyPoint(gauss[Geometry::index[idx][0]],gauss[Geometry::index[idx][1]],gauss[Geometry::index[idx][2]]);
			GaussVertexWeight[idx] = w;
			//printf("%d : %f,%f,%f  w:%f\n",idx,gauss[index[idx][0]],gauss[index[idx][1]],gauss[index[idx][2]],w);
		}
		//MyPause;
	}

	void Cell::compute(const MyPoint &p, std::vector< MyFloat> &values, std::vector< MyDenseVector > &grads, std::vector< MyMatrix_3X3 > &grad_grads)
	{
		unsigned int n_values_and_derivatives = 0;
		n_values_and_derivatives = 2;

		//Table<2,Tensor<1,3> > v(dim, Polynomials.size()/*2*/);
		std::vector< std::vector< MyDenseVector > > v(MyDIM);
		for (unsigned int i=0; i < MyDIM; ++i)
		{
			v[i].resize( m_Polynomials.size() );
		}

		{
			std::vector< MyFloat > tmp (n_values_and_derivatives);
			for (unsigned int d=0; d<MyDIM; ++d)
			{
				for (unsigned int i=0; i<m_Polynomials.size(); ++i)
				{
					m_Polynomials[i].value(p(d), tmp);
					for (unsigned int e=0; e<n_values_and_derivatives; ++e)
					{
						v[d][i]/*(d,i)*/[e] = tmp[e];
					}
				}
			}
		}

		for (unsigned int i=0; i<Geometry::n_tensor_pols; ++i)
		{
			unsigned int indices[MyDIM];
			compute_index (i, indices);

			values[i] = 1;
			for (unsigned int x=0; x<MyDIM; ++x)
			{
				values[i] *= v[x][indices[x]]/*(x,indices[x])*/[0];
			}


			for (unsigned int d=0; d<MyDIM; ++d)
			{
				grads[i][d] = 1.;
				for (unsigned int x=0; x<MyDIM; ++x)
					grads[i][d] *= v[x][indices[x]]/*(x,indices[x])*/[d==x];


			}

		}
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

	MyFloat Cell::determinant (const MyMatrix_3X3 &t)
	{
		//t.determinant();
		return  t.coeff(0,0)*t.coeff(1,1)*t.coeff(2,2) + t.coeff(0,1)*t.coeff(1,2)*t.coeff(2,0) + t.coeff(0,2)*t.coeff(1,0)*t.coeff(2,1) -
			t.coeff(2,0)*t.coeff(1,1)*t.coeff(0,2) - t.coeff(1,0)*t.coeff(0,1)*t.coeff(2,2) - t.coeff(0,0)*t.coeff(2,1)*t.coeff(1,2);
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

	void Cell::contract (MyDenseVector &dest, const MyDenseVector &src1, const MyMatrix_3X3 &src2)
	{
		dest.setZero();
		for (unsigned int i=0; i<MyDIM; ++i)
			for (unsigned int j=0; j<MyDIM; ++j)
				dest[i] += src1[j] * src2.coeff(j,i)/*[j][i]*/;
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
				//printf("value %d :",k);
				for (unsigned int i=0; i< Geometry::dofs_per_cell_8; ++i)
				{
					shapeFunctionValue_8_8.coeffRef(k,i) = values[i];
					//printf("%f,",values[i]);
					//printf("shape function %f\n",values[i]);
				}
				//printf("\n");MyPause;
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
		else
		{
			MyError("UnSupport Cell Type! (FEM)");
		}
	}

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

				for (unsigned int i=0; i<MyDIM; ++i)
				{
					for (unsigned int j=0; j<MyDIM; ++j)
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
				for (unsigned int d=0;d<MyDIM;++d)
				{
					auxiliary[d] = input[p][d];
				}
				contract (output[p], auxiliary, covariant[p]);

			}
#if 0
#else
			for (unsigned int p=0;p<n_points;++p)
			{
				shapeDerivativeValue_mapping_8_8_3[p][i] = output[p];
			}
#endif
		}
	}

	void Cell::makeStiffnessMatrix()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nStiffnessMatrixIdx = isHasInitialized(lRadius,vec_cell_stiffness_matrix);
		if (-1 < m_nStiffnessMatrixIdx)
		{
			//has initialized
			m_nFEMShapeIdx = m_nStiffnessMatrixIdx;
			return ;
		}
		static std::vector<int> vecDofs;
		static MyMatrix StiffnessMatrix_24_24;
		StrainMatrix_6_24.resize(6,24);
		StrainMatrix_6_24.setZero();

		StiffnessMatrix_24_24.resize(24,24);
		StiffnessMatrix_24_24.setZero();

		MyMatrix refMaterialMatrix =  MaterialMatrix::getMaterialMatrix(Material::YoungModulus,Material::PossionRatio);

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

				/*StrainMatrix_6_24.coeffRef(4,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][2];
				StrainMatrix_6_24.coeffRef(4,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][1];

				StrainMatrix_6_24.coeffRef(5,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][2];
				StrainMatrix_6_24.coeffRef(5,col+2) = shapeDerivativeValue_mapping_8_8_3[q][I][0];

				StrainMatrix_6_24.coeffRef(3,col+0) = shapeDerivativeValue_mapping_8_8_3[q][I][1];
				StrainMatrix_6_24.coeffRef(3,col+1) = shapeDerivativeValue_mapping_8_8_3[q][I][0];*/
			}
			StiffnessMatrix_24_24 += StrainMatrix_6_24.transpose() * refMaterialMatrix * StrainMatrix_6_24*getJxW(q);
		}

		//std::cout << StiffnessMatrix_24_24 << std::endl;MyPause;
		m_nStiffnessMatrixIdx = appendMatrix(vec_cell_stiffness_matrix,lRadius,StiffnessMatrix_24_24);
		printf("m_nStiffnessMatrixIdx=%d\n",m_nStiffnessMatrixIdx);
	}

	void Cell::makeMassMatrix_Lumping()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nMassMatrixIdx = isHasInitialized(lRadius,vec_cell_mass_matrix);
		if (-1 < m_nMassMatrixIdx)
		{
			//has initialized
			return ;
		}

		static std::vector<int> vecDofs;
		static MyMatrix MassMatrix_24_24;
		const MyFloat dbDensity = Material::Density*10;
		MassMatrix_24_24.resize(24,24);
		MassMatrix_24_24.setZero();
		MyMatrix lumpingMatrix;
		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			MyMatrix tmpShapeMatrix(3,24);
			tmpShapeMatrix.setZero();
			for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
			{
				const unsigned col = k*MyDIM;
				for (unsigned i=0;i<MyDIM;++i)
				{
					tmpShapeMatrix.coeffRef(0,col+0) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(1,col+1) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(2,col+2) = shapeFunctionValue_8_8.coeff(q,k);
				}
			}
#if 1
			lumpingMatrix = dbDensity * tmpShapeMatrix.transpose() * tmpShapeMatrix * getJxW(q) ;
			MyFloat lumpingValue = 0.f;
			for (int r=0;r<24;++r)
			{
				for (int c=0;c<24;++c)
				{
					lumpingValue += lumpingMatrix.coeff(r,c);
				}
			}
			MassMatrix_24_24.coeffRef(q*3+0,q*3+0) = lumpingValue/3.f;
			MassMatrix_24_24.coeffRef(q*3+1,q*3+1) = lumpingValue/3.f;
			MassMatrix_24_24.coeffRef(q*3+2,q*3+2) = lumpingValue/3.f;
#else
			MassMatrix_24_24 += dbDensity * tmpShapeMatrix.transpose() * tmpShapeMatrix * getJxW(q) ;
#endif
			
			
		}
		//std::cout << MassMatrix_24_24 << std::endl;MyPause;
		m_nMassMatrixIdx = appendMatrix(vec_cell_mass_matrix,lRadius,MassMatrix_24_24);
	}

	void Cell::makeLocalRhs()
	{
		long lRadius = ValueScaleFactor*m_radius;
		m_nRhsIdx = isHasInitialized(lRadius,vec_cell_rhs_matrix);
		if (-1 < m_nRhsIdx)
		{
			//has initialized
			return ;
		}

		static std::vector<int> vecDofs;
		static MyVector RightHandValue_24_1;
		RightHandValue_24_1.resize(24,1);
		RightHandValue_24_1.setZero();

		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			MyMatrix tmpShapeMatrix(3,24);
			tmpShapeMatrix.setZero();
			for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
			{
				const unsigned col = k*MyDIM;
				for (unsigned i=0;i<MyDIM;++i)
				{
					tmpShapeMatrix.coeffRef(0,col+0) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(1,col+1) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(2,col+2) = shapeFunctionValue_8_8.coeff(q,k);
				}
			}

			RightHandValue_24_1 += tmpShapeMatrix.transpose() * externalForce * getJxW(q);
		}

		m_nRhsIdx = appendVector(vec_cell_rhs_matrix,lRadius,RightHandValue_24_1);
	}

	int Cell::isHasInitialized( const long radius, const std::vector< tuple_matrix >& mapp )
	{
		for (unsigned i=0;i<mapp.size();++i)
		{
			if (radius == mapp.at(i).m_Radius)
			{
				return i;
			}
		}
		return -1;
	}

	int Cell::isHasInitialized( const long radius, const std::vector< tuple_vector >& mapp )
	{
		for (unsigned i=0;i<mapp.size();++i)
		{
			if (radius == mapp.at(i).m_Radius)
			{
				return i;
			}
		}
		return -1;
	}

	int Cell::appendMatrix(std::vector< tuple_matrix >& mapp, const long radius,const MyDenseMatrix& matrix)
	{
		tuple_matrix t;
		t.m_Radius = radius;
		t.matrix = matrix;
		mapp.push_back(t);
		return mapp.size() -1;
	}

	int Cell::appendVector(std::vector< tuple_vector >& mapp, const long radius,const MyVector& Vector)
	{
		tuple_vector t;
		t.m_Radius = radius;
		t.vec = Vector;
		mapp.push_back(t);
		return mapp.size()-1;
	}

	void Cell::initialize_Global()
	{
		m_Polynomials = Polynomial::generate_complete_basis(1);
		computeGaussPoint();
		computeShapeFunction();
		computeJxW();
		computeShapeGrad();
		//print(std::cout);
		makeStiffnessMatrix();
		makeMassMatrix_Lumping();
		makeLocalRhs();

#if USE_MODAL_WARP
		createW_24_24();
#endif
	}

	void Cell::get_dof_indices_Global(std::vector<int> &vecDofs)
	{
		vecDofs.clear();
		Q_ASSERT(FEM == m_CellType);
		if (FEM == m_CellType)
		{
			for (int i=0;i<Geometry::vertexs_per_cell;++i)
			{
				MyVectorI& ref_Dof = m_elem_vertex[i]->getGlobalDofs();
				vecDofs.push_back(ref_Dof[0]);
				vecDofs.push_back(ref_Dof[1]);
				vecDofs.push_back(ref_Dof[2]);
			}
		}
		else
		{
			MyError("UnSupport Cell Type! (FEM)");
		}
	}

	void Cell::assembleSystemMatrix_Global()
	{
		static std::vector<int> vecDofs;

		get_dof_indices_Global(vecDofs);

		compressStiffnessMatrix_Global(vec_cell_stiffness_matrix[m_nStiffnessMatrixIdx].matrix,vecDofs);
		compressMassMatrix_Global(vec_cell_mass_matrix[m_nMassMatrixIdx].matrix,vecDofs);
		compressRHS_Global(vec_cell_rhs_matrix[m_nRhsIdx].vec,vecDofs);

#if 0
		static std::vector<float> vecDofsWeights;
		get_dof_indices_ShareCellCountWeight_Global(vecDofsWeights);
		compress_W_Matrix_Global(m_W_24_24,vecDofs,vecDofsWeights);
#endif
	}

	void Cell::compressMassMatrix_Global(const MyMatrix& objMatrix,std::vector<int> & vecDofs)
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_global_state.m_TripletNode_Mass);
	}

	void Cell::compressStiffnessMatrix_Global(const MyMatrix& objMatrix,std::vector<int> & vecDofs)
	{
		compressMatrix(objMatrix,vecDofs,Cell::m_global_state.m_TripletNode_Stiffness);
	}

	void Cell::compressMatrix(const MyMatrix& objMatrix,std::vector<int> &vecDofs,std::map<long,std::map<long,TripletNode > >& TripletNodeMap)
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

	void Cell::compressRHS_Global(const MyVector& rhs,std::vector<int> &vecDofs)
	{
		for (unsigned r=0;r<rhs.size();++r)
		{
			const MyFloat val = rhs[r];
			if (!numbers::isZero(val))
			{
				Cell::m_global_state.m_TripletNode_Rhs[vecDofs[r]].val += val;
			}
		}
	}

#if USE_MODAL_WARP

	MyMatrix Cell::m_W_24_24;

	void Cell::createW_24_24()
	{
		static bool bFirst=true;
		if (bFirst)
		{
			bFirst = false;
			m_W_24_24.resize(Geometry::dofs_per_cell,Geometry::dofs_per_cell);
			m_W_24_24.setZero();

			const float half_for_curl = 0.5f;
			for (int q=0;q<Geometry::vertexs_per_cell;++q)
			{
				const int row_base = q*3;
				for (unsigned I=0;I < Geometry::vertexs_per_cell;++I)
				{
					//shapeDerivativeValue_mapping_8_8_3
					//shapeDerivativeValue_8_8_3
					const unsigned col_base = I*3;
#if 0
					m_W_24_24.coeffRef(row_base+0,col_base+0) = 0.f;
					m_W_24_24.coeffRef(row_base+0,col_base+1) = -1.f*shapeDerivativeValue_8_8_3[q][I][2];
					m_W_24_24.coeffRef(row_base+0,col_base+2) = shapeDerivativeValue_8_8_3[q][I][1];

					m_W_24_24.coeffRef(row_base+1,col_base+0) = shapeDerivativeValue_8_8_3[q][I][2];
					m_W_24_24.coeffRef(row_base+1,col_base+1) = 0.f;
					m_W_24_24.coeffRef(row_base+1,col_base+2) = -1.f*shapeDerivativeValue_8_8_3[q][I][0];

					m_W_24_24.coeffRef(row_base+2,col_base+0) = -1.f*shapeDerivativeValue_8_8_3[q][I][1];
					m_W_24_24.coeffRef(row_base+2,col_base+1) = shapeDerivativeValue_8_8_3[q][I][0];
					m_W_24_24.coeffRef(row_base+2,col_base+2) = 0.f;
#else
					m_W_24_24.coeffRef(row_base+0,col_base+0) = 0.f;
					m_W_24_24.coeffRef(row_base+0,col_base+1) = -1.f*shapeDerivativeValue_mapping_8_8_3[q][I][2];
					m_W_24_24.coeffRef(row_base+0,col_base+2) = shapeDerivativeValue_mapping_8_8_3[q][I][1];

					m_W_24_24.coeffRef(row_base+1,col_base+0) = shapeDerivativeValue_mapping_8_8_3[q][I][2];
					m_W_24_24.coeffRef(row_base+1,col_base+1) = 0.f;
					m_W_24_24.coeffRef(row_base+1,col_base+2) = -1.f*shapeDerivativeValue_mapping_8_8_3[q][I][0];

					m_W_24_24.coeffRef(row_base+2,col_base+0) = -1.f*shapeDerivativeValue_mapping_8_8_3[q][I][1];
					m_W_24_24.coeffRef(row_base+2,col_base+1) = shapeDerivativeValue_mapping_8_8_3[q][I][0];
					m_W_24_24.coeffRef(row_base+2,col_base+2) = 0.f;
#endif
				}
			}

			m_W_24_24 *= half_for_curl;
			//m_W_24_24 *= getJxW(0);
			
		}
		//std::cout << m_W_24_24 << std::endl; MyPause;
		/*std::ofstream outfile("d:\\W.txt");
		
		outfile << m_W_24_24 << std::endl;
		MyPause;*/
	}

	void Cell::get_dof_indices_ShareCellCountWeight_Global(std::vector<float> &vecDofsWeights)
	{
		vecDofsWeights.clear();
		Q_ASSERT(FEM == m_CellType);
		if (FEM == m_CellType)
		{
			for (int i=0;i<Geometry::vertexs_per_cell;++i)
			{
				//MyVectorI& ref_Dof = m_elem_vertex[i]->getGlobalDofs();
				int nCount = m_elem_vertex[i]->getShareCellCount();
				vecDofsWeights.push_back(1.f/nCount);
				vecDofsWeights.push_back(1.f/nCount);
				vecDofsWeights.push_back(1.f/nCount);
			}

			//std::copy(vecDofsWeights.begin(), vecDofsWeights.end(), std::ostream_iterator<float>(std::cout, " "));  MyPause;
		}
		else
		{
			MyError("UnSupport Cell Type! (FEM)");
		}
	}

	void Cell::compress_W_Matrix_Global(const MyMatrix& objMatrix,std::vector<int> & vecDofs,std::vector<float> & vecDofsWeights)
	{
		for (unsigned r=0;r<objMatrix.rows();++r)
		{
			const float w = vecDofsWeights[r];
			for (unsigned c=0;c<objMatrix.cols();++c)
			{
				const MyFloat val = objMatrix.coeff(r,c);
				if (!numbers::isZero(val))
				{
					m_global_state.m_TripletNode_ModalWarp[vecDofs[r]][vecDofs[c]].val += val*w;
				}
			}
		}
	}

	void Cell::TestModalWrapMatrix(const MyVector& globlaDisp, MyVector& w, MyVector& translation, MyVector& incremental_displacement)
	{
		incremental_displacement.resize(Geometry::dofs_per_cell);
		//MyVector incremental_displacement(Geometry::dofs_per_cell);
		for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
		{
			MyVectorI dofs = m_elem_vertex[i]->getGlobalDofs();

			incremental_displacement[3*i+0] = globlaDisp[ dofs[0] ];
			incremental_displacement[3*i+1] = globlaDisp[ dofs[1] ];
			incremental_displacement[3*i+2] = globlaDisp[ dofs[2] ];

			//std::cout << "# " << i << " : "<< dofs.transpose() << std::endl;
		}


		w = m_W_24_24 * incremental_displacement;

		MyVector nativePos(Geometry::dofs_per_cell);
		for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
		{
			//MyVectorI dofs = m_elem_vertex[i]->getGlobalDofs();
			MyDenseVector pos = m_elem_vertex[i]->getPos();
			nativePos[3*i+0] =  pos[0] ;
			nativePos[3*i+1] =  pos[1] ;
			nativePos[3*i+2] =  pos[2] ;
		}
		MyMatrix shp(Geometry::dofs_per_cell,Geometry::dofs_per_cell);shp.setZero();
		for (unsigned q=0;q<Geometry::vertexs_per_cell;++q)
		{
			MyMatrix tmpShapeMatrix(3,24);
			tmpShapeMatrix.setZero();
			for (unsigned k=0;k<Geometry::shape_Function_Count_In_FEM;++k)
			{
				const unsigned col = k*MyDIM;
				for (unsigned i=0;i<MyDIM;++i)
				{
					tmpShapeMatrix.coeffRef(0,col+0) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(1,col+1) = shapeFunctionValue_8_8.coeff(q,k);
					tmpShapeMatrix.coeffRef(2,col+2) = shapeFunctionValue_8_8.coeff(q,k);
				}
			}

			shp.block(3*q,0,3,24) = tmpShapeMatrix;
		}

		translation = shp * incremental_displacement + nativePos;

		
	}

	void Cell::TestCellRotationMatrix(const MyVector& globlaDisp, MyMatrix_3X3& RotationMatrix, MyDenseVector& m_FrameTranslationVector)
	{
		static MyDenseVector Signs[Geometry::vertexs_per_cell] = {
																	MyDenseVector(-1,-1,-1), 
																	MyDenseVector(1,-1,-1),
																	MyDenseVector(-1,1,-1),
																	MyDenseVector(1,1,-1),
																	MyDenseVector(-1,-1,1),
																	MyDenseVector(1,-1,1),
																	MyDenseVector(-1,1,1),
																	MyDenseVector(1,1,1)};

			

			MyVector incremental_displacement(Geometry::dofs_per_cell);
			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				MyVectorI dofs = m_elem_vertex[i]->getGlobalDofs();
				
				incremental_displacement[3*i+0] = globlaDisp[ dofs[0] ];
				incremental_displacement[3*i+1] = globlaDisp[ dofs[1] ];
				incremental_displacement[3*i+2] = globlaDisp[ dofs[2] ];
			}
			{
				m_FrameTranslationVector.setZero();
				for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
				{
					m_FrameTranslationVector += (m_elem_vertex[i]->getPos() + incremental_displacement.block(i*3,0,3,1));
				}

				m_FrameTranslationVector /= 8.f;
			}
			MyMatrix_3X3 tmp;
			tmp.setZero();
			for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
			{
				tmp += incremental_displacement.block(i*3,0,3,1) * Signs[i].transpose();
			}
			RotationMatrix = Eigen::Matrix<MyFloat, 3, 3>::Identity();
			RotationMatrix += 0.25f * (1.f / (m_radius * 2)) * tmp;

			for (unsigned i=1;i<5;++i)
			{
				RotationMatrix = 0.5f*(RotationMatrix + RotationMatrix.inverse().transpose());
			}


			
			//return RotationMatrix;
	}
#endif//USE_MODAL_WARP
}//namespace YC
