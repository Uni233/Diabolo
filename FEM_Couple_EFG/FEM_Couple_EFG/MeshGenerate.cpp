#include "stdafx.h"
#include "MeshGenerate.h"
#include "solvercontrol.h"
#include "solvercg.h"
#include "preconditionssor.h"
#include "BlasOperator.h"
#include <Eigen/Cholesky>
#include "constant_numbers.h"
#include <fstream>
#include <map>

using namespace std;
extern float element_steak[3136][4];
namespace VR_FEM
{
	MeshGenerate::MeshGenerate(MyFloat cellRadius, const int X_Count, const int Y_Count, const int Z_Count, std::vector< Plane >& vecPlanes)
		:m_radius(cellRadius),m_axis_count(X_Count,Y_Count,Z_Count),m_vecPlanes(vecPlanes),m_nDof(0)
	{
		m_BoundaryType = EFG;
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
		/*m_db_NewMarkConstant [0]=16384;
		m_db_NewMarkConstant [1]=128;
		m_db_NewMarkConstant [2]=256;
		m_db_NewMarkConstant [3]=1;
		m_db_NewMarkConstant [4]=1;
		m_db_NewMarkConstant [5]=0;
		m_db_NewMarkConstant [6]=0.0078125;
		m_db_NewMarkConstant [7]=0.0078125;*/
		m_vecPlanes.clear();
		m_vecPlanes.push_back(VR_FEM::Plane(MyPoint(CellRaidus*2*8.5,1,0),MyPoint(CellRaidus*2*8.5,-1,-1),MyPoint(CellRaidus*2*8.5,-1,1) ));
		m_vecPlanes.push_back(VR_FEM::Plane(MyPoint(CellRaidus*2*16.5,1,0),MyPoint(CellRaidus*2*12.5,-1,-1),MyPoint(CellRaidus*2*12.5,-1,1) ));
	}

	void MeshGenerate::generate()
	{
		const MyFloat diameter = 2*m_radius;
		std::vector< MyFloat > vec_X_SamplePoint,vec_Y_SamplePoint,vec_Z_SamplePoint;
		for (unsigned i=0;i<m_axis_count[X_AXIS];++i)
		{
			vec_X_SamplePoint.push_back(i*diameter+m_radius);
		}
		for (unsigned i=0;i<m_axis_count[Y_AXIS];++i)
		{
			vec_Y_SamplePoint.push_back(i*diameter+m_radius);
		}
		for (unsigned i=0;i<m_axis_count[Z_AXIS];++i)
		{
			vec_Z_SamplePoint.push_back(i*diameter+m_radius);
		}

		printf("vec_X_SamplePoint.size(%d) vec_Y_SamplePoint.size(%d) vec_Z_SamplePoint.size(%d)\n",vec_X_SamplePoint.size(),vec_Y_SamplePoint.size(),vec_Z_SamplePoint.size());
		//MyPause;
		for (unsigned i=0;i<vec_X_SamplePoint.size();++i)
		{
			for (unsigned j=0;j<vec_Y_SamplePoint.size();++j)
			{
				for (unsigned k=0;k<vec_Z_SamplePoint.size();++k)
				{
					m_vec_cell.push_back(Cell::makeCell(MyPoint(vec_X_SamplePoint[i],vec_Y_SamplePoint[j],vec_Z_SamplePoint[k]),m_radius));
					m_vec_cell[m_vec_cell.size()-1]->computeCellType(m_vecPlanes);
				}
			}
		}

		printf("vertex is %d  cell is %d\n",Vertex::getVertexSize(),Cell::getCellSize());
		distributeDof();
		
		printf("nDof is %d\n",m_nDof);
		
		for (unsigned i=0; i < Cell::getCellSize();++i)
		{
			printf("Cell %d\n",i);
			
			Cell::getCell(i)->initialize();
			Cell::getCell(i)->clear();
			//Cell::getCell(i)->print(std::cout);
		}

		
		/*MyPause;

		std::map< unsigned,bool >::const_iterator ci = Cell::s_map_InfluncePointSize.begin();
		std::map< unsigned,bool >::const_iterator endc = Cell::s_map_InfluncePointSize.end();
		for (;ci != endc; ++ci)
		{
			printf("Influnce %d\t",ci->first);
		}
		printf("\n");

		MyPause;*/
		printf("create_boundary \n");
		create_boundary();

		printf("createGlobalMassAndStiffnessAndDampingMatrix \n");
		//createGlobalMassAndStiffnessAndDampingMatrix();
		createGlobalMassAndStiffnessAndDampingMatrixEFG();

		printf("createNewMarkMatrix\n");
		
		//apply_boundary_values();
		createNewMarkMatrix();
		
		
		//MyPause;
	}
	
	void MeshGenerate::generate_steak(const char* lpszSteakFile,const char* lpszTexturePath)
	{
		loadObjDataSteak(lpszSteakFile);
		//loadObjDataBunny(lpszSteakFile);
		makeScalarObjDataSteak();
		LoadGLTextures(lpszTexturePath);

		const unsigned nSamplePointSize = 3136;//element_steak[3136][4]
		for (unsigned i=0;i<nSamplePointSize; ++i)
		{
			m_vec_cell.push_back(Cell::makeCell(MyPoint(element_steak[i][0],element_steak[i][1],element_steak[i][2]),element_steak[i][3]));
			m_vec_cell[m_vec_cell.size()-1]->computeCellType_Steak();
		}
		

		printf("vertex is %d  cell is %d\n",Vertex::getVertexSize(),Cell::getCellSize());
		distributeDof();		
		printf("nDof is %d\n",m_nDof);

		printf("makeLineSet_Steak\n");
		makeLineSet_Steak();

		printf("makeTriangleMeshInterpolation_Steak\n");
		//makeTriangleMeshInterpolation_Steak();
		
		for (unsigned i=0; i < Cell::getCellSize();++i)
		{
			printf("Cell %d\n",i);
			
			Cell::getCell(i)->initialize();
			Cell::getCell(i)->clear();
		}
		printf("FEM(%d) EFG(%d) Couple(%d)\n",Cell::s_nFEM_Cell_Count,Cell::s_nEFG_Cell_Count,Cell::s_nCOUPLE_Cell_Count);
		MyPause;

		printf("create_boundary \n");
		create_boundary_steak();

		printf("createGlobalMassAndStiffnessAndDampingMatrix \n");

		//createGlobalMassAndStiffnessAndDampingMatrixEFG();

		printf("createNewMarkMatrix\n");
		
		//createNewMarkMatrix();
		
		assembleFEMPreComputeMatrix();
		assembleOnCuda();
		initVBOStructure();
		
		initCuttingStructure();
		//MyPause;
	}

	void MeshGenerate::solve_timestep(unsigned nTimeStep)
	{
		/*if (false == numbers::isSymmetric(m_computeMatrix))
		{
			Q_ASSERT(false);
		}*/

		double dbCurrentTime = 0.;
		if (nTimeStep < MaxTimeStep) //for (unsigned currentTimeStep = 0;currentTimeStep < MaxTimeStep; ++currentTimeStep)
		{
			update_rhs(nTimeStep);

			apply_boundary_values();

			solve_linear_problem ();

			update_u_v_a();

			//assembleRotationSystemMatrix();
		}
	}

	void MeshGenerate::printCuttingCell()
	{
		static int cuttingCell[] = {81,83,85,87,97,99,101,103,265,267,269,271,281,283,285,287,329,331,333,335,345,347,349,351,953,955,957,959,969,971,973,975,1649,1651,1653,1655,1665,1667,1669,1671,1833,1835,1837,1839,1849,1851,1853,1855,1897,1899,1901,1903,1913,1915,1917,1919,2521,2523,2525,2527,2537,2539,2541,2543};
		::glPointSize(3.5f);
		glColor3f(1.f,0.f,0.f);
		glBegin(GL_POINTS);

		const int nCuttingCellSize = sizeof(cuttingCell) / sizeof(int);
		for (unsigned v=0;v<nCuttingCellSize;++v)
		{
			CellPtr& refCell =  m_vec_cell[cuttingCell[v]];
			for (unsigned j=0;j<Geometry::vertexs_per_cell;++j)
			{
				MyPoint & refPoint = refCell->getVertex(j)->getPos();
				MyVectorI & refDofs = refCell->getVertex(j)->getDofs();
				int curColor = refCell->getVertex(j)->getFromDomainId();
				glColor3f(Colors::colorTemplage[curColor][0],Colors::colorTemplage[curColor][1],Colors::colorTemplage[curColor][2]);
				glVertex3f(refPoint[0] ,
					refPoint[1] ,
					refPoint[2] );
			}
		}
		glEnd();
	}

	void MeshGenerate::print()
	{
		/*std::ofstream outfile("d:\\incremental_displacement.txt");
		outfile << incremental_displacement ;
		exit(0);*/
		
		::glPointSize(1.5f);
		glColor3f(1.f,0.f,0.f);
		glBegin(GL_POINTS);
		for (unsigned v=0;v<m_vec_cell.size();++v)
		{
			CellPtr& refCell =  m_vec_cell[v];
			for (unsigned j=0;j<Geometry::vertexs_per_cell;++j)
			{
				MyPoint & refPoint = refCell->getVertex(j)->getPos();
				MyVectorI & refDofs = refCell->getVertex(j)->getDofs();
				//printf("{%f,%f,%f}\n",refPoint[0],refPoint[1],refPoint[2]);
				/*glVertex3f(refPoint[0] ,
						   refPoint[1] ,
						   refPoint[2] );*/
				/*int curColor = refCell->getVertex(j)->getFromDomainId();
				glColor3f(Colors::colorTemplage[curColor][0],Colors::colorTemplage[curColor][1],Colors::colorTemplage[curColor][2]);*/
				glVertex3f(refPoint[0] /* + incremental_displacement[refDofs[0]]*/,
					refPoint[1]  /*+ incremental_displacement[refDofs[1]]*/,
					refPoint[2]  /*+ incremental_displacement[refDofs[2]]*/);
			}

			/*int curColor = refCell->getDomainId();
			glColor3f(Colors::colorTemplage[curColor][0],Colors::colorTemplage[curColor][1],Colors::colorTemplage[curColor][2]);
			MyPoint centerPoint = refCell->getCenterPoint();
			glVertex3f(centerPoint[0],centerPoint[1],centerPoint[2]);*/
		}
		glEnd();

		/*const unsigned nVertexSize = Vertex::getVertexSize();
		for (unsigned v=0;v < nVertexSize;++v)
		{
			Vertex::getVertex(v)->printFrame();
		}*/
		
	}

	void MeshGenerate::distributeDof()
	{

		for (unsigned i=0;i<Vertex::getVertexSize();++i)
		{
			VertexPtr currentVertex = Vertex::getVertex(i);
			currentVertex->setDof(m_nDof+0,m_nDof+1,m_nDof+2);
			m_nDof +=3;
		}
	}

	void MeshGenerate::create_boundary()
	{
		//std::cout << "boundary dof : ";
		for (unsigned i=0;i<Vertex::getVertexSize();++i)
		{
			VertexPtr currentVertex = Vertex::getVertex(i);

			MyPoint& point = currentVertex->getPos();
			if (point[0] < (0.1*CellRaidus))
			{
				m_vec_boundary.push_back(currentVertex->getDof(0));
				m_vec_boundary.push_back(currentVertex->getDof(1));
				m_vec_boundary.push_back(currentVertex->getDof(2));

				m_vec_boundaryVertex.push_back(currentVertex);
				//std::cout << m_vec_boundary[m_vec_boundary.size()-1] << ",";
			}
			else if (point[0] > (47.5*CellRaidus) )
			{
				m_vecForceBoundary.push_back(currentVertex->getDof(1));
			}
		}
		//std::cout << std::endl;
		printf("boundary.size() is %d; m_vecForceBoundary.size()=%d\n",m_vec_boundary.size(),m_vecForceBoundary.size());
		//MyPause;
	}

	void MeshGenerate::create_boundary_steak()
	{
		m_vecForceBoundary;
		map<int,bool> vecMapVertexId,vecMapVertexId_boundary;
		for (unsigned c=0;c<Cell::getCellSize();++c)
		{
			CellPtr curCellPtr = Cell::getCell(c);
			for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
			{
				MyPoint & refPos = curCellPtr->getVertex(v)->getPos();

				/*const MyFloat curX = refPos[1];
				const MyFloat curY = refPos[0];*/

				const MyFloat curX = ((refPos[1]) ) ;//Take care of rotation 90
				const MyFloat curY = ((refPos[0]) ) ;

				if ( (curY < 0.8f && curY > 0.75f && curX < 0.85f && curX > 0.15f) /*|| (curY < 0.8f && curY > 0.1f && curX>0.48f && curX < 0.53f)*/ )
				{
					vecMapVertexId_boundary[curCellPtr->getVertex(v)->getId()] = true;
				}
				else if (curX < 0.2f && curY < 0.2f)
				{
					vecMapVertexId[curCellPtr->getVertex(v)->getId()] = true;
				}
			}
		}

		map<int,bool>::const_iterator ci = vecMapVertexId.begin();
		for (;ci != vecMapVertexId.end();++ci)
		{
			const int vertexId = ci->first;
			VertexPtr currentVertex = Vertex::getVertex(vertexId);
			//m_vecForceBoundary.push_back(currentVertex->getDof(0));
			//m_vecForceBoundary.push_back(currentVertex->getDof(1));
			m_vecForceBoundary.push_back(currentVertex->getDof(2));

			
			//m_vec_ForceBoundaryVertex.push_back(currentVertex);
		}

		ci = vecMapVertexId_boundary.begin();
		for (;ci != vecMapVertexId_boundary.end();++ci)
		{
			const int vertexId = ci->first;
			VertexPtr currentVertex = Vertex::getVertex(vertexId);
			m_vec_boundary.push_back(currentVertex->getDof(0));
			m_vec_boundary.push_back(currentVertex->getDof(1));
			m_vec_boundary.push_back(currentVertex->getDof(2));

			m_vec_boundaryVertex.push_back(currentVertex);
		}

		printf("m_vec_boundaryVertex.size is %d m_vec_boundary.size is %d\n",m_vec_boundaryVertex.size(),m_vec_boundary.size());
		//MyPause;
	}


	void MeshGenerate::apply_boundary_values()
	{
		if (true/*FEM == m_BoundaryType*/)
		{
			printf("apply_boundary_values begin \n");
			if (m_vec_boundary.size() == 0)
				return;


			const unsigned int n_dofs = m_computeMatrix.rows();
			std::vector<MyFloat> diagnosticValue(n_dofs,0.0);

			for (int v=0;v < n_dofs;++v)
			{
				diagnosticValue[v] = m_computeMatrix.coeff(v,v);
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

			for (unsigned i=0; i < m_vec_boundary.size(); ++i)
			{
				const unsigned int dof_number = m_vec_boundary[i];
				//std::cout << m_computeMatrix << std::endl;
				setMatrixRowZeroWithoutDiag(m_computeMatrix,dof_number);
				//std::cout << m_computeMatrix << std::endl;
				//exit(66);

				MyFloat new_rhs;
				if ( !numbers::isZero(diagnosticValue[dof_number] ) )
				{
					new_rhs = 0 * diagnosticValue[dof_number];
					R_rhs(dof_number) = new_rhs;
				}
				else
				{
					m_computeMatrix.coeffRef(dof_number,dof_number) = first_nonzero_diagonal_entry;
					new_rhs = 0 * first_nonzero_diagonal_entry;
					R_rhs(dof_number) = new_rhs;
				}
				incremental_displacement(dof_number) = 0;
			}
		}
		else
		{
			//EFG Boundary type
			for (unsigned i=0; i < m_vec_boundaryVertex.size(); ++i)
			{
				VertexPtr currentVertex = m_vec_boundaryVertex[i];
				MyVector u(3);
				u.setZero();

				unsigned InflPtsNb;
				std::vector<VertexPtr> InflPts;
				MyVector SHP;
				ApproxAtPoint(currentVertex->getPos(),InflPtsNb,InflPts,SHP);

				

				const unsigned Nb = dim*InflPtsNb;
				MyMatrix S(Nb,Nb);
				MyVector R(Nb);

				{
					MyMatrix G(dim, dim*InflPtsNb);
					G.setZero();

					double alpha = 1.0e2 * Material::YoungModulus;

					for (unsigned shapeIdx=0; shapeIdx<InflPtsNb; shapeIdx++)
					{
						MyFloat f = (SHP)(shapeIdx);

						G.coeffRef(0, 3*shapeIdx+0) = f;						
						G.coeffRef(1, 3*shapeIdx+1) = f;
						G.coeffRef(2, 3*shapeIdx+2) = f;
					}

					
					blasOperator::Blas_R1_Trans_Update(S, G, alpha, 0.0);  // Transpose[N] * N
					blasOperator::Blas_Mat_Trans_Vec_Mult(G, u, R, alpha, 0.0);

					std::vector< unsigned > vecDofs;
					for (unsigned k=0;k<InflPts.size();++k)
					{
						vecDofs.push_back(InflPts[k]->getDof(0));
						vecDofs.push_back(InflPts[k]->getDof(1));
						vecDofs.push_back(InflPts[k]->getDof(2));
					}

					for (unsigned r =0;r< vecDofs.size();++r)
					{
						for (unsigned c=0;c<vecDofs.size();++c)
						{
							m_computeMatrix.coeffRef(vecDofs[r],vecDofs[c]) = S.coeff(r,c);
						}
						//m_computeRhs[vecDofs[r]] += R[r];
						m_computeRhs[vecDofs[r]] = 0.f;
					}
				}
			}
		}
	}

	unsigned MeshGenerate::InfluentPoints(const MyPoint& gaussPointInGlobalCoordinate, std::vector<VertexPtr>& vecInflunceVertex )
	{
		const unsigned Npts = Vertex::getVertexSize();

		vecInflunceVertex.clear();
		for( unsigned i = 0; i< Npts; i++ )
		{
			VertexPtr currentVertex = Vertex::getVertex(i);
			const MyPoint& vertexPos = currentVertex->getPos();
			if( abs( gaussPointInGlobalCoordinate.x() - vertexPos.x()) < SupportSize && 
				abs( gaussPointInGlobalCoordinate.y() - vertexPos.y() ) < SupportSize && 
				abs( gaussPointInGlobalCoordinate.z() - vertexPos.z() ) < SupportSize)
			{
				vecInflunceVertex.push_back(currentVertex);
			}
		}
		return vecInflunceVertex.size();
	}

	void MeshGenerate::ApproxAtPoint(const MyPoint& evPoint, unsigned& nInfluncePoint, std::vector<VertexPtr>& vecInflunceVertex, MyVector& vecShapeFunction )
	{
		const MyPoint& pX = evPoint;
		MyVector& N_ =  vecShapeFunction;

		MyFloat WI;         // Value of weight function at node I
		MyFloat WxI, WyI, WzI;   // Value of derivatives of weight function at node I
		MyFloat WxxI, WxyI, WyyI, WzzI, WxzI, WyzI;

		nInfluncePoint = InfluentPoints(pX,vecInflunceVertex);

		const unsigned InflPtsNb_ = nInfluncePoint;
		
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
		//MyMatrix  Bx_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对x的偏导数
		//MyMatrix  By_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对y的偏导数
		//MyMatrix  Bz_(EFG_BasisNb_,InflPtsNb_);  // B矩阵对z的偏导数

		//MyMatrix  Bxx_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xx的偏导数
		//MyMatrix  Byy_(EFG_BasisNb_,InflPtsNb_); // B矩阵对yy的偏导数
		//MyMatrix  Bzz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对zz的偏导数
		//MyMatrix  Bxy_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数
		//MyMatrix  Bxz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数
		//MyMatrix  Byz_(EFG_BasisNb_,InflPtsNb_); // B矩阵对xy的偏导数

		MyVector			P_(EFG_BasisNb_);
		//MyMatrix            D1P_(EFG_BasisNb_,dim);

		N_.resize(InflPtsNb_);

		MyDenseVector Radius_;
		Radius_[0] = SupportSize;Radius_[1] = SupportSize;Radius_[2] = SupportSize;

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
			/*Bx_.setZero();
			By_.setZero();
			Bz_.setZero();
			Bxx_.setZero();
			Byy_.setZero();
			Bzz_.setZero();
			Bxy_.setZero();
			Bxz_.setZero();
			Byz_.setZero();*/
			P_.setZero();
			N_.setZero();
			//D1P_.setZero();
		}
		
		std::vector< VertexPtr >& refCurGaussPointInfls = vecInflunceVertex;
		Q_ASSERT(refCurGaussPointInfls.size() == InflPtsNb_);
		for (unsigned i=0; i<InflPtsNb_; i++)
		{
			VertexPtr curInflsPoint = refCurGaussPointInfls[i];
			MyPoint pts_j = curInflsPoint->getPos();

			//     A  = Sum of WI(X)*P(XI)*Transpose[P(XI)], I=1,N

			Cell::basis(pX,curInflsPoint->getPos(), P_);
			
			WI = Cell::WeightFun(pts_j, Radius_, pX, 0);
			

			//  (*A_) = (*A_) + WI * (*P_) * Transpose [ (*P_) ]
			MyFloat cof = ( i == 0 ? 0:1 );
			blasOperator::Blas_R1_Update(A_, P_, WI, cof);
			

			{
				WxI = Cell::WeightFun(pts_j, Radius_, pX, 1);
				WyI = Cell::WeightFun(pts_j, Radius_, pX, 2);
				WzI = Cell::WeightFun(pts_j, Radius_, pX, 3);

				//  (*Ax_) = (*Ax_) + WxI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ax_, P_, WxI, cof);

				//  (*Ay_) = (*Ay_) + WyI * (*P_) * Transpose [ (*P_) ]
				blasOperator::Blas_R1_Update(Ay_, P_, WyI, cof);
				blasOperator::Blas_R1_Update(Az_, P_, WzI, cof);
			}

			{
				WxxI = Cell::WeightFun(pts_j, Radius_, pX, 4);
				WyyI = Cell::WeightFun(pts_j, Radius_, pX, 5);
				WzzI = Cell::WeightFun(pts_j, Radius_, pX, 6);
				WxyI = Cell::WeightFun(pts_j, Radius_, pX, 7);
				WxzI = Cell::WeightFun(pts_j, Radius_, pX, 8);
				WyzI = Cell::WeightFun(pts_j, Radius_, pX, 9);
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
			

			//{
			//	for(unsigned k = 0; k < EFG_BasisNb_; k++ )
			//	{
			//		/*(*Bx_)(k, i) = (*P_)(k) * WxI;
			//		(*By_)(k, i) = (*P_)(k) * WyI;
			//		(*Bz_)(k, i) = (*P_)(k) * WzI;*/

			//		Bx_.coeffRef(k, i) = (P_)(k) * WxI;
			//		By_.coeffRef(k, i) = (P_)(k) * WyI;
			//		Bz_.coeffRef(k, i) = (P_)(k) * WzI;
			//	}
			//}

			//
			//{	
			//	for( int k = 0; k< EFG_BasisNb_; k++ )
			//		(Bxx_).coeffRef(k, i) = (P_)(k) * WxxI;
			//	for( int k = 0; k< EFG_BasisNb_; k++ )
			//		(Byy_).coeffRef(k, i) = (P_)(k) * WyyI;
			//	for( int k = 0; k< EFG_BasisNb_; k++ )
			//		(Bzz_).coeffRef(k, i) = (P_)(k) * WzzI;
			//	for( int k = 0; k< EFG_BasisNb_; k++ )
			//		(Bxy_).coeffRef(k, i) = (P_)(k) * WxyI;
			//	for(int  k = 0; k< EFG_BasisNb_; k++ )
			//		(Bxz_).coeffRef(k, i) = (P_)(k) * WxzI;
			//	for(int  k = 0; k< EFG_BasisNb_; k++ )
			//		(Byz_).coeffRef(k, i) = (P_)(k) * WyzI;
			//}
		}

		Eigen::LLT<MyMatrix> lltOfA(A_);//LaLUFactor(*A_);

		//  r = Inverse[A] * P

		Cell::basis(pX, pX, P_);
		P_ = lltOfA.solve(P_);//LaLUSolve(*A_, *P_);  // r is stored in P_

		//  Transpose[N] = Transpose[r] * B
		blasOperator::Blas_Mat_Trans_Vec_Mult(B_, P_, N_, 1.0, 0.0);

		return ;
	}

	void MeshGenerate::createGlobalMassAndStiffnessAndDampingMatrix()
	{		
		m_computeMatrix.resize(m_nDof,m_nDof);
		m_global_MassMatrix.resize(m_nDof,m_nDof);
		m_global_StiffnessMatrix.resize(m_nDof,m_nDof);
		m_global_DampingMatrix.resize(m_nDof,m_nDof);
		m_computeRhs.resize(m_nDof);

		{
			R_rhs.resize(m_nDof);
			mass_rhs.resize(m_nDof);
			damping_rhs.resize(m_nDof);
			displacement.resize(m_nDof);
			velocity.resize(m_nDof);
			acceleration.resize(m_nDof);
			old_acceleration.resize(m_nDof);
			old_displacement.resize(m_nDof);
			incremental_displacement.resize(m_nDof);

			R_rhs.setZero();
			mass_rhs.setZero();
			damping_rhs.setZero();
			displacement.setZero();
			velocity.setZero();
			acceleration.setZero();
			old_acceleration.setZero();
			old_displacement.setZero();
			incremental_displacement.setZero();
		}

		m_computeMatrix.setZero();
		m_global_MassMatrix.setZero();
		m_global_StiffnessMatrix.setZero();
		m_global_DampingMatrix.setZero();
		m_computeRhs.setZero();

		std::map<long,std::map<long,TripletNode > > TripletNode4Mass,TripletNode4Stiffness;

		std::vector<int> localDofs;
		for (unsigned i=0; i < Cell::getCellSize();++i)
		{
			CellPtr cellPtr = Cell::getCell(i);
			cellPtr->get_dof_indices(localDofs);
			const unsigned localDofCount = localDofs.size();
			const MyMatrix & curMassMatrix = cellPtr->getMassMatrix();
			const MyMatrix & curStiffnessMatrix = cellPtr->getStiffnessMatrix();
			const MyVector & curRhs = cellPtr->getRhsVector();
			
			distribute_local_to_global(curStiffnessMatrix,localDofs,m_global_StiffnessMatrix);
			distribute_local_to_global(curRhs,localDofs,m_computeRhs);
		}


		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;

		std::map<long,std::map<long,TripletNode > >::const_iterator itr_tri =  TripletNode4Mass.begin();
		for (;itr_tri != TripletNode4Mass.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}

		m_global_MassMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		TripletNode4Mass.clear();
		vec_triplet.clear();

		itr_tri =  m_TripletNode.begin();
		for (;itr_tri != m_TripletNode.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}

		m_global_StiffnessMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		m_TripletNode.clear();
		vec_triplet.clear();


		create_mass_matrix(m_global_MassMatrix);
		//m_global_MassMatrix *= 100000;

		
		m_global_DampingMatrix = Material::damping_alpha * m_global_MassMatrix + Material::damping_beta * m_global_StiffnessMatrix;
	}

	void MeshGenerate::createNewMarkMatrix()
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
	}

	void MeshGenerate::update_rhs(int nStep)
	{
		R_rhs.setZero();
		mass_rhs.setZero();
		damping_rhs.setZero();

		//incremental_displacement,velocity,acceleration
		mass_rhs += m_db_NewMarkConstant[0] * displacement;
		mass_rhs += m_db_NewMarkConstant[2] * velocity;
		mass_rhs += m_db_NewMarkConstant[3] * acceleration;

		damping_rhs += m_db_NewMarkConstant[1] * displacement;
		damping_rhs += m_db_NewMarkConstant[4] * velocity;
		damping_rhs += m_db_NewMarkConstant[5] * acceleration;

		R_rhs += m_computeRhs;

		R_rhs += m_global_MassMatrix * mass_rhs;
		R_rhs += m_global_DampingMatrix * damping_rhs;

		R_rhs -= m_RotationRHS;

		if ( (nStep%50) < 10 )
		{
			for (unsigned i=0;i<m_vecForceBoundary.size();++i)
			{
				R_rhs[m_vecForceBoundary[i]] += 345.772125 * 10;
			}
		}
		
		
		
	}

	void MeshGenerate::update_u_v_a()
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

		for (unsigned i=0; i < Cell::getCellSize();++i)
		{
			CellPtr cellPtr = Cell::getCell(i);
			cellPtr->computeRotationMatrix(incremental_displacement);
		}
	}

	void MeshGenerate::setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx )
	{
		{
			for (MySpMat::InnerIterator it(matrix,rowIdx); it; ++it)
			{
				Q_ASSERT(rowIdx == it.row());
				const int r = it.row();
				const int c = it.col();
				if ( r == rowIdx && (r != c) )
				{
					it.valueRef() = 0.;
				}
			}
		}
	}

	void MeshGenerate::solve_linear_problem()
	{
		SolverControl           solver_control (m_nDof*3,  1e-3*numbers::l2_norm(R_rhs));
		SolverCG              cg (solver_control);

		PreconditionSSOR preconditioner;
		preconditioner.initialize(m_computeMatrix, 1.2);

		cg.solve (m_computeMatrix, incremental_displacement, R_rhs,	preconditioner);

		const unsigned nVertexSize = Vertex::getVertexSize();

		/*for (unsigned v=0;v<nVertexSize;++v)
		{
			Vertex::getVertex(v)->computeRotationMatrix(incremental_displacement);
		}*/
		/*std::ofstream outR_rhs("d:\\R_rhs.txt");
		outR_rhs << R_rhs;
		outR_rhs.close();
		std::ofstream outComputeRhs("d:\\computRhs.txt");
		outComputeRhs << m_computeRhs;
		outComputeRhs.close();
		exit(0);*/
	}

	void MeshGenerate::distribute_local_to_global (
		const MyMatrix                  &local_matrix,
		const std::vector< int>         &local_dof_indices,
		const VR_FEM::MySpMat                   &global_matrix
		)
	{
		const bool use_vectors = false;//= (local_vector.size() == 0 && global_vector.size() == 0) ? false : true;
		typedef MyFloat number;
		const bool use_dealii_matrix =true;// types_are_equal<MatrixType,SparseMatrix<number> >::value;

		const unsigned int n_local_dofs =24;// local_dof_indices.size();

		GlobalRowsFromLocal global_rows (n_local_dofs);

		make_sorted_row_list (local_dof_indices, global_rows);

		const unsigned int n_actual_dofs = global_rows.size();//n_active_rows

		for (unsigned int i=0; i<n_actual_dofs; ++i)
		{
			const unsigned int row = global_rows.global_row(i);


			resolve_matrix_row (global_rows, i, 0, n_actual_dofs,local_matrix, global_matrix);
		}

		set_matrix_diagonals (global_rows, local_dof_indices,local_matrix, global_matrix);
	}

	void MeshGenerate::distribute_local_to_global (const MyVector& local_vector,const std::vector<int> &local_dof_indices, MyVector& global_vector) const
	{
		Q_ASSERT (local_vector.size() == local_dof_indices.size());
		const unsigned local_size = local_vector.size();
		for ( unsigned v =0;v < local_size;++v )
		{
			if (is_constrained(local_dof_indices[v]) == false)
			{
				global_vector(local_dof_indices[v]) += local_vector[v];
			}
			else
			{
			}
		}
	}

	void MeshGenerate::make_sorted_row_list (const std::vector< int> &local_dof_indices,GlobalRowsFromLocal  &global_rows) const
	{
		const unsigned int n_local_dofs = local_dof_indices.size();
		Q_ASSERT (n_local_dofs == global_rows.size());
		unsigned int added_rows = 0;
		for (unsigned int i = 0; i<n_local_dofs; ++i)
		{
			if (is_constrained(local_dof_indices[i]) == false)
			{
				global_rows.global_row(added_rows)  = local_dof_indices[i];
				global_rows.local_row(added_rows++) = i;
				continue;
			}
			global_rows.insert_constraint(i);
		}

		global_rows.sort();

		const unsigned int n_constrained_rows = n_local_dofs-added_rows;
		for (unsigned int i=0; i<n_constrained_rows; ++i)
		{
			const unsigned int local_row = global_rows.constraint_origin(i);//total_row_indices[n_active_rows+i].local_row;
			Q_ASSERT(local_row >=0 && local_row < n_local_dofs);

			const unsigned int global_row = local_dof_indices[local_row];
			Q_ASSERT (is_constrained(global_row));
		}
	}

	void  MeshGenerate::set_matrix_diagonals (const GlobalRowsFromLocal &global_rows,
		const std::vector< int>              &local_dof_indices,
		const MyDenseMatrix                       &local_matrix,
		const VR_FEM::MySpMat                        &global_matrix)
	{
		if (global_rows.n_constraints() > 0)
		{
			MyFloat average_diagonal = 0;
			for (unsigned int i=0; i<local_matrix.rows(); ++i)
				average_diagonal += std::fabs (local_matrix(i,i));
			average_diagonal /= static_cast<MyFloat>(local_matrix.rows());

			for (unsigned int i=0; i<global_rows.n_constraints(); i++)
			{
				const unsigned int local_row = global_rows.constraint_origin(i);
				const unsigned int global_row = local_dof_indices[local_row];

				const MyFloat new_diagonal = (std::fabs(local_matrix(local_row,local_row)) != 0 ? std::fabs(local_matrix(local_row,local_row)) : average_diagonal);
				printf("#####(%d,%d)(%f)-->(%f)\n",global_row,global_row,m_TripletNode[global_row][global_row].val,new_diagonal);
				m_TripletNode[global_row][global_row].val += new_diagonal;
				//global_matrix.coeffRef(global_row, global_row) += new_diagonal;
			}
		}
	}

	void   MeshGenerate::resolve_matrix_row (const GlobalRowsFromLocal&global_rows,
		const unsigned int        i,/*1-24*/
		const unsigned int        column_start/*0*/,
		const unsigned int        column_end/*24*/,
		const MyDenseMatrix           &local_matrix,
		const MySpMat                    &sparse_matrix)
	{
		Q_ASSERT ( (column_end-1) >=0 && (column_end-1) < global_rows.size());
		const unsigned int global_row/*global dof*/ = global_rows.global_row(i);
		const unsigned int loc_row /*i*/= global_rows.local_row(i);
		Q_ASSERT( loc_row >=0 && loc_row < local_matrix.rows() );

		/*i = 0-23*/
		for (unsigned int j=column_start/*0*/; j<i; ++j)
		{
			const unsigned int loc_col/*j*/ = global_rows.local_row(j);
			const MyFloat col_val /*local_matrix(i,j)*/= local_matrix(loc_row,loc_col);
			const unsigned int global_col = global_rows.global_row(j);
			m_TripletNode[global_row][global_col].val += col_val;
		}

		m_TripletNode[global_row][global_row].val += local_matrix(loc_row,loc_row);

		for (unsigned int j=i+1; j<column_end/*24*/; ++j)
		{
			const unsigned int loc_col = global_rows.local_row(j);
			const MyFloat col_val = local_matrix(loc_row,loc_col);
			const unsigned int global_col = global_rows.global_row(j);
			m_TripletNode[global_row][global_col].val += col_val;
		}
	}

	void MeshGenerate::create_mass_matrix(VR_FEM::MySpMat &matrix)
	{
		m_TripletNode.clear();
		std::vector<int> localDofs;
		for (unsigned i=0; i < Cell::getCellSize();++i)
		{
			CellPtr cellPtr = Cell::getCell(i);
			cellPtr->get_dof_indices(localDofs);
			const unsigned localDofCount = localDofs.size();
			const MyMatrix & curMassMatrix = cellPtr->getMassMatrix();

			copy_local_to_global(localDofs,curMassMatrix,matrix);
		}

		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;

		std::map<long,std::map<long,TripletNode > >::const_iterator itr_tri =  m_TripletNode.begin();
		for (;itr_tri != m_TripletNode.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}

		matrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		m_TripletNode.clear();

	}

	void MeshGenerate::copy_local_to_global(const std::vector< int> &indices,const MyDenseMatrix  &vals,const VR_FEM::MySpMat &matrix)
	{
		for (unsigned v = 0;v < indices.size(); ++v)
		{
			for (unsigned w = 0;w < indices.size();++w)
			{
				if (!numbers::isZero(vals(v,w)) )
				{
					m_TripletNode[indices[v]][indices[w]].val += vals(v,w);
				}
			}
		}
	}

	void MeshGenerate::createGlobalMassAndStiffnessAndDampingMatrixEFG()
	{
		m_computeMatrix.resize(m_nDof,m_nDof);
		m_global_MassMatrix.resize(m_nDof,m_nDof);
		m_global_StiffnessMatrix.resize(m_nDof,m_nDof);
		m_global_DampingMatrix.resize(m_nDof,m_nDof);
		m_computeRhs.resize(m_nDof);
		m_RotationRHS.resize(m_nDof);

		{
			R_rhs.resize(m_nDof);
			mass_rhs.resize(m_nDof);
			damping_rhs.resize(m_nDof);
			displacement.resize(m_nDof);
			velocity.resize(m_nDof);
			acceleration.resize(m_nDof);
			old_acceleration.resize(m_nDof);
			old_displacement.resize(m_nDof);
			incremental_displacement.resize(m_nDof);

			R_rhs.setZero();
			mass_rhs.setZero();
			damping_rhs.setZero();
			displacement.setZero();
			velocity.setZero();
			acceleration.setZero();
			old_acceleration.setZero();
			old_displacement.setZero();
			incremental_displacement.setZero();
		}

		m_computeMatrix.setZero();
		m_global_MassMatrix.setZero();
		m_global_StiffnessMatrix.setZero();
		m_global_DampingMatrix.setZero();
		m_computeRhs.setZero();
		m_RotationRHS.setZero();

		std::vector<int> localDofs;
		//std::map<long,std::map<long,TripletNode > > TripletNode4Mass,TripletNode4Stiffness;
		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;
		
		std::map<long,std::map<long,TripletNode > >& TripletNode4Mass = Cell::m_TripletNode_Mass;
		std::map<long,std::map<long,TripletNode > >::const_iterator itr_tri =  TripletNode4Mass.begin();
		for (;itr_tri != TripletNode4Mass.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}

		m_global_MassMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		TripletNode4Mass.clear();
		vec_triplet.clear();

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::map<long,std::map<long,TripletNode > >& TripletNode4Stiffness = Cell::m_TripletNode_Stiffness;
		itr_tri =  TripletNode4Stiffness.begin();
		for (;itr_tri != TripletNode4Stiffness.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
			for (;itr_2 != ref_map.end();++itr_2)
			{
				vec_triplet.push_back( Eigen::Triplet<MyFloat,long>(itr_tri->first,itr_2->first,(itr_2->second).val) );
			}
		}

		m_global_StiffnessMatrix.setFromTriplets(vec_triplet.begin(),vec_triplet.end());
		TripletNode4Stiffness.clear();
		vec_triplet.clear();

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::map<long,TripletNode >& TripletNode4RHS = Cell::m_TripletNode_Rhs;
		std::map<long,TripletNode >::const_iterator itr_rhs = TripletNode4RHS.begin();
		for (; itr_rhs != TripletNode4RHS.end();++itr_rhs)
		{
			m_computeRhs[itr_rhs->first] = (itr_rhs->second).val;
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		m_global_DampingMatrix = Material::damping_alpha * m_global_MassMatrix + Material::damping_beta * m_global_StiffnessMatrix;
	}

	void MeshGenerate::compareMatrix(MySpMat& leftMatrix,MySpMat& rightMatrix)
	{
		Q_ASSERT(leftMatrix.rows() == rightMatrix.rows());
		Q_ASSERT(leftMatrix.cols() == rightMatrix.cols());
		for (unsigned r=0;r<leftMatrix.rows();++r)
		{
			for (unsigned c=0;c<leftMatrix.cols();++c)
			{
				const MyFloat leftVal = leftMatrix.coeff(r,c);
				const MyFloat rightVal = rightMatrix.coeff(r,c);
				if (!numbers::isEqual(leftVal,rightVal))
				{
					printf("[%d,%d]{%f,%f}\n",r,c,leftVal,rightVal);
				}
			}
		}
	}

	void MeshGenerate::assembleRotationSystemMatrix()
	{
		Cell::m_TripletNode_Stiffness.clear();
		Cell::m_TripletNode_Rhs.clear();
		
		m_RotationRHS.setZero();

		const unsigned nCellSize = Cell::getCellSize();
		for (unsigned v = 0; v < nCellSize; ++v)
		{
			printf("co-ro cell %d\n",v);
			Cell::getCell(v)->assembleRotationMatrix();
		}

		m_computeMatrix.setZero();
		//m_global_MassMatrix.setZero();
		m_global_StiffnessMatrix.setZero();
		m_global_DampingMatrix.setZero();
		//m_computeRhs.setZero();

		std::vector<int> localDofs;
		//std::map<long,std::map<long,TripletNode > > TripletNode4Mass,TripletNode4Stiffness;
		std::vector< Eigen::Triplet<MyFloat,long> > vec_triplet;


		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::map<long,std::map<long,TripletNode > >& TripletNode4Stiffness = Cell::m_TripletNode_Stiffness;
		std::map<long,std::map<long,TripletNode > >::const_iterator itr_tri =  TripletNode4Stiffness.begin();
		for (;itr_tri != TripletNode4Stiffness.end();++itr_tri)
		{
			const std::map<long,TripletNode >&  ref_map = itr_tri->second;
			std::map<long,TripletNode >::const_iterator itr_2 =  ref_map.begin();
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
		std::map<long,TripletNode >& TripletNode4RHS = Cell::m_TripletNode_Rhs;
		std::map<long,TripletNode >::const_iterator itr_rhs = TripletNode4RHS.begin();
		for (; itr_rhs != TripletNode4RHS.end();++itr_rhs)
		{
			m_RotationRHS[itr_rhs->first] = (itr_rhs->second).val;
		}
#endif

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		m_global_DampingMatrix = Material::damping_alpha * m_global_MassMatrix + Material::damping_beta * m_global_StiffnessMatrix;

		m_computeMatrix = m_global_StiffnessMatrix;
		m_computeMatrix += m_db_NewMarkConstant[0] * m_global_MassMatrix;
		m_computeMatrix += m_db_NewMarkConstant[1] * m_global_DampingMatrix;
	}
}