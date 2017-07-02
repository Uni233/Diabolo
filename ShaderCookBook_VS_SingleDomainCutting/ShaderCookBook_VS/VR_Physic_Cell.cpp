#include "VR_Physic_Cell.h"

#include "constant_numbers.h"
#include <iterator>     // std::back_inserter
#include <vector>       // std::vector
#include <algorithm>    // std::copy
#include "VR_Material.h"
#include "VR_GlobalVariable.h"
#include <fstream>
using namespace YC::GlobalVariable;
namespace YC
{
	namespace Physics
	{
// Class MaterialMatrix
#if 1
		std::vector<MaterialMatrix::MatrixInfo> MaterialMatrix::s_vecMatrixMatrix;

		void MaterialMatrix::makeSymmetry(MyMatrix& objMatrix)
		{
			for (unsigned r=0;r<objMatrix.rows();++r)
			{
				for (unsigned c=0;c<r;++c)
				{
					objMatrix.coeffRef(c,r) = objMatrix.coeff(r,c);
				}
			}
		}

		MyMatrix MaterialMatrix::makeMaterialMatrix(const MyFloat y, const MyFloat p)
		{		

			MyMatrix MaterialMatrix_6_6;
			MaterialMatrix_6_6.resize(MaterialMatrixSize,MaterialMatrixSize);
			MaterialMatrix_6_6.setZero();

			MyFloat E = y/*Material::YoungModulus*/;
			MyFloat mu = p;

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

			return MaterialMatrix_6_6;
		}

		MyMatrix MaterialMatrix::getMaterialMatrix(const MyFloat YoungModule, const MyFloat PossionRatio)
		{
			std::vector< MatrixInfo >::iterator itr = std::find_if(s_vecMatrixMatrix.begin(),s_vecMatrixMatrix.end(),MaterialMatrixCompare(YoungModule,PossionRatio));
			if (itr == s_vecMatrixMatrix.end())
			{
				//no find
				const MyMatrix& tmp = makeMaterialMatrix(YoungModule,PossionRatio);
				s_vecMatrixMatrix.push_back(MatrixInfo(YoungModule,PossionRatio,tmp));
				return s_vecMatrixMatrix[s_vecMatrixMatrix.size()-1].tMatrix;
			}
			else
			{
				//fing
				return (*itr).tMatrix;
			}
		}
#endif


		namespace GPU
		{
//Class Cell
#if 1
			MyPoint Cell::GaussVertex[Geometry::vertexs_per_cell];
			MyFloat   Cell::GaussVertexWeight[Geometry::vertexs_per_cell];
			Physics::MaterialMatrix Cell::s_MaterialMatrix;
			std::vector< CellPtr > Cell::s_Cell_Cache;
			std::map<long,std::map<long,Cell::TripletNode > > Cell::m_TripletNode_Mass,Cell::m_TripletNode_Stiffness;
			std::map<long,Cell::TripletNode >				   Cell::m_TripletNode_Rhs;

			
			MyFloat Cell::scaleExternalForce(g_externalForceFactor);
			MyDenseVector Cell::externalForce(0.f,Material::GravityFactor * Cell::scaleExternalForce * Material::Density,0.f);

			std::vector< Cell::tuple_matrix > Cell::vec_cell_stiffness_matrix;
			std::vector< Cell::tuple_matrix > Cell::vec_cell_mass_matrix;
			
			std::vector< Cell::tuple_vector > Cell::vec_cell_rhs_matrix;
			std::vector< FEMShapeValue > Cell::vec_FEM_ShapeValue;

			void Cell::get_dof_indices(std::vector<int> &vecDofs)
			{
				vecDofs.clear();
				Q_ASSERT(MACRO::FEM == m_CellType);
				if (MACRO::FEM == m_CellType)
				{
					for (int i=0;i<Geometry::vertexs_per_cell;++i)
					{
						MyVectorI& ref_Dof = m_elem_vertex[i]->getDofs();
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

			void Cell::get_postion(MyVector& Pj_FEM)
			{
				Pj_FEM.resize(Geometry::dofs_per_cell);Pj_FEM.setZero();
				for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
				{
					Pj_FEM.block(i*3,0,3,1) = m_elem_vertex[i]->getPos();
				}
			}


			Cell::Cell(MyPoint center, MyFloat radius, VertexPtr vertexes[])
				:m_center(center),m_radius(radius),m_CellType(MACRO::INVALIDTYPE),
				 m_nID(Invalid_Id),m_nRhsIdx(Invalid_Id),m_nMassMatrixIdx(Invalid_Id),
				 m_nStiffnessMatrixIdx(Invalid_Id),m_needBeCutting(false)
			{
				scaleExternalForce = (g_externalForceFactor);
				externalForce = MyDenseVector(0.f,Material::GravityFactor * scaleExternalForce * Material::Density,0.f);
				/*LogInfo("%f,%f,%f,%f\n",externalForce[0],externalForce[1],externalForce[2],scaleExternalForce);
				MyPause;*/
				m_CellType = MACRO::FEM;
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

				incremental_displacement.resize(Geometry::dofs_per_cell);
				incremental_displacement.setZero();
			}

			void Cell::initialize()
			{
				m_Polynomials = Polynomial::generate_complete_basis(1);
				computeGaussPoint();
				computeShapeFunction();
				computeJxW();
				computeShapeGrad();
				//print(std::cout);
				makeStiffnessMatrix();
				makeMassMatrix();
				makeLocalRhs();
			}

			void Cell::clear()
			{
				for (unsigned i=0;i<Geometry::gauss_Sample_Point;++i)
				{
					for (unsigned j=0;j<Geometry::vertexs_per_cell;++j)
					{
						shapeSecondDerivativeValue_8_8_3_3[i][j].resize(0,0);
						shapeDerivativeValue_mapping_8_8_3[i][j].resize(0,0);
						shapeSecondDerivativeValue_8_8_3_3[i][j].resize(0,0);
					}
					contravariant[i].resize(0,0);
					covariant[i].resize(0,0);
				}

				shapeFunctionValue_8_8.resize(0,0);
				StrainMatrix_6_24.resize(0,0);
				m_Polynomials.clear();
				
				
				RotationMatrix.resize(0,0);
				Cell_Corotation_Matrix.resize(0,0);
				Pj_FEM.resize(0,0);
				incremental_displacement.resize(0,0);
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

			void Cell::computeShapeFunction()
			{
				if (MACRO::FEM == m_CellType)
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

			void Cell::makeShapeFunctionMatrix()
			{
				MyError("Call makeShapeFunctionMatrix");
			}


			void Cell::assembleSystemMatrix()
			{
				static std::vector<int> vecDofs;

				get_dof_indices(vecDofs);
#if 0
				compressLocalFEMStiffnessMatrix(m_nDomainId,vec_cell_stiffness_matrix[m_nStiffnessMatrixIdx].matrix,vecDofs);

				compressLocalFEMMassMatrix(m_nDomainId,vec_cell_mass_matrix[m_nMassMatrixIdx].matrix,vecDofs);

				compressLocalFEMRHS(m_nDomainId,vec_cell_rhs_matrix[m_nRhsIdx].vec,vecDofs);
#else
				compressStiffnessMatrix(vec_cell_stiffness_matrix[m_nStiffnessMatrixIdx].matrix,vecDofs);
				compressMassMatrix(vec_cell_mass_matrix[m_nMassMatrixIdx].matrix,vecDofs);
				compressRHS(vec_cell_rhs_matrix[m_nRhsIdx].vec,vecDofs);
#endif
				/*for(int i=0;i<vecDofs.size();++i) printf("%d,",vecDofs[i]);
				printf("\n\n");
				std::cout << vec_cell_stiffness_matrix[m_nStiffnessMatrixIdx].matrix << std::endl;
				std::cout << vec_cell_mass_matrix[m_nMassMatrixIdx].matrix << std::endl;
				std::cout << vec_cell_rhs_matrix[m_nRhsIdx].vec << std::endl;

				printf("%d--%d--%d\n",m_TripletNode_Stiffness.size(),m_TripletNode_Mass.size(),m_TripletNode_Rhs.size());
				MyPause;*/
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
					}

					
					//std::cout << refMaterialMatrix << std::endl;
					/*std::cout << StrainMatrix_6_24 << std::endl;
					std::cout <<"getJxW(q) "<< getJxW(q) << std::endl;*/
					StiffnessMatrix_24_24 += StrainMatrix_6_24.transpose() * refMaterialMatrix * StrainMatrix_6_24*getJxW(q);
					//std::cout << StiffnessMatrix_24_24 << std::endl;
					//MyPause;
				}
				/*std::ofstream outfile("d:\\stiffness.txt");
				outfile << StiffnessMatrix_24_24;
				outfile.close();*/
				m_nStiffnessMatrixIdx = appendMatrix(vec_cell_stiffness_matrix,lRadius,StiffnessMatrix_24_24);
				printf("m_nStiffnessMatrixIdx=%d\n",m_nStiffnessMatrixIdx);
				m_nFEMShapeIdx = appendFEMShapeValue(this);
				Q_ASSERT(m_nStiffnessMatrixIdx == m_nFEMShapeIdx);
			}

			void Cell::makeMassMatrix()
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
				const MyFloat dbDensity = Material::Density;
				MassMatrix_24_24.resize(24,24);
				MassMatrix_24_24.setZero();

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

					MassMatrix_24_24 += dbDensity * tmpShapeMatrix.transpose() * tmpShapeMatrix * getJxW(q) ;
				}
				/*std::ofstream outfile("d:\\massmatrix.txt");
				outfile << MassMatrix_24_24;
				outfile.close();*/
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

				/*std::ofstream outfile("d:\\rhs.txt");
				outfile << RightHandValue_24_1 ;
				outfile.close();*/

				m_nRhsIdx = appendVector(vec_cell_rhs_matrix,lRadius,RightHandValue_24_1);
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
								//printf("YANGCHENDEBUG 3-12 v(%d,%d)[%d]=tmp[%d] {%f} {p(%d) %f}\n",d,i,e,e,tmp[e],d,p(d));
							}
						}
					}
				}

				for (unsigned int i=0; i<Geometry::n_tensor_pols; ++i)
				{
					unsigned int indices[MyDIM];
					compute_index (i, indices);
					//printf("YANGCHENDEBUG 3-12 [%d]:{%d,%d,%d}\n",i,indices[0],indices[1],indices[2]);

					values[i] = 1;
					for (unsigned int x=0; x<MyDIM; ++x)
					{
						values[i] *= v[x][indices[x]]/*(x,indices[x])*/[0];
					}

					//printf("YANGCHENDEBUG 3-12 value[%d]=%f {v(0,indices[0])[0]=%f}{v(1,indices[1])[1]=%f}{v(2,indices[2])[2]=%f}\n",i,values[i],
					//v[0][indices[0]][0],v[1][indices[1]][0],v[2][indices[2]][0]);

					for (unsigned int d=0; d<MyDIM; ++d)
					{
						grads[i][d] = 1.;
						for (unsigned int x=0; x<MyDIM; ++x)
							grads[i][d] *= v[x][indices[x]]/*(x,indices[x])*/[d==x];

						//printf("YANGCHENDEBUG 3-12  grads[%d][%d]=%f\n",i,d,grads[i][d]);
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

			void Cell::assembleRotationMatrix()
			{
				static std::vector<int> vecDofs;
				if (MACRO::FEM == m_CellType)
				{
					Cell_Corotation_Matrix.resize(Geometry::dofs_per_cell,Geometry::dofs_per_cell);
					Cell_Corotation_Matrix.setZero();

					for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
					{
						Cell_Corotation_Matrix.block(i*3,i*3,3,3) = RotationMatrix;
					}

					get_dof_indices(vecDofs);
					
					const MyDenseMatrix& StiffnessMatrix_24_24 = getStiffMatrix(m_nStiffnessMatrixIdx);
					compressStiffnessMatrix(Cell_Corotation_Matrix*StiffnessMatrix_24_24*Cell_Corotation_Matrix.transpose(),vecDofs);

					compressRHS(Cell_Corotation_Matrix * StiffnessMatrix_24_24 * (Cell_Corotation_Matrix.transpose()*Pj_FEM - Pj_FEM),vecDofs);
				}
				else if (MACRO::COUPLE == m_CellType)
				{
					MyError("UnSupport Cell Type! FEM");
				}
				else if (MACRO::EFG == m_CellType)
				{
					MyError("UnSupport Cell Type! FEM");
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


			const MyDenseMatrix& Cell::getStiffMatrix(const int id)
			{
				return vec_cell_stiffness_matrix[id].matrix;
			}

			const MyDenseMatrix& Cell::getMassMatrix(const int id)
			{
				return vec_cell_mass_matrix[id].matrix;
			}

			const MyVector& Cell::getRHSVector(const int id)
			{
				return vec_cell_rhs_matrix[id].vec;
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

			void Cell::basis(const MyPoint& pX, const MyPoint& pXI, MyVector& pP)
			{
				MyError("call Cell::basis");
				/*const MyFloat XI = pXI.x();
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
				}*/
			}

			void Cell::dbasis(const MyPoint& pX, const MyPoint& pXI, MyMatrix& DP_27_3)
			{
				MyError("call Cell::dbasis");
			}

			MyFloat Cell::WeightFun(const MyPoint& pCenterPt, const MyDenseVector& Radius, const MyPoint& pEvPoint, int DerOrder)
			{
				MyError("call Cell::WeightFun");
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

			void Cell::makeGaussPoint(double gauss[2],double w)
			{
				static unsigned int index[8][3] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};

				for (unsigned int idx = 0; idx <Geometry::vertexs_per_cell;++idx)
				{
					GaussVertex[idx] = MyPoint(gauss[index[idx][0]],gauss[index[idx][1]],gauss[index[idx][2]]);
					GaussVertexWeight[idx] = w;
				}
			}

			void Cell::compressMassMatrix(const MyMatrix& objMatrix,std::vector<int> & vecDofs)
			{
				compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_Mass);
			}

			void Cell::compressStiffnessMatrix(const MyMatrix& objMatrix,std::vector<int> & vecDofs)
			{
				compressMatrix(objMatrix,vecDofs,Cell::m_TripletNode_Stiffness);
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

			int Cell::appendFEMShapeValue(Cell * pThis)
			{
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
				return vec_FEM_ShapeValue.size()-1;
			}
#endif
		}
	}//namespace Physics
}//namespace YC