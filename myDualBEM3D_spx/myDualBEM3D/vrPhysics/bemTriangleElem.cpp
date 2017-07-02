#include "bemTriangleElem.h"
#include "vrBase/vrLog.h"
#include <boost/math/constants/constants.hpp> // pi
#include "discontinuousColletionPoint.h"
#include "vrGlobalConf.h"

VR::DisContinuousCollectionPoint tmp;
const VR::MyFloat s1 = tmp.s1;
const VR::MyFloat s2 = tmp.s2;
//const MyFloat pi = boost::math::constants::pi<MyFloat>();
#include "lgwt.h"
namespace VR
{
	std::vector< TriangleElemPtr > TriangleElem::s_Triangle_Cache;

	MyVec2ParamSpace TriangleElem::s_paramSpace[Geometry::vertexs_per_tri] = { MyVec2ParamSpace(0.0,0.0), MyVec2ParamSpace(1.0,0.0), MyVec2ParamSpace(0.0,1.0)};//[0,+1]
	// 0 : (ξ1, η1) = (1/6, 1/6) , (ξ2, η2) = (2/3 , 1/6) , (ξ3, η3) = (1/6 , 2/3) , w1 = w2 = w3 = 1/3
	MyVec2ParamSpace TriangleElem::s_gaussPtInParamSpace[Geometry::vertexs_per_tri] = { MyVec2ParamSpace(1.0 / 6.0, 1.0 / 6.0), MyVec2ParamSpace(2.0 / 3.0, 1.0 / 6.0), MyVec2ParamSpace(1.0 / 6.0, 2.0 / 3.0) };//[-sqrt(2/3), +sqrt(2/3)]
	MyFloat TriangleElem::s_gaussPtWeigth[Geometry::vertexs_per_tri] = { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};


	void TriangleElem::CountSurfaceType()
	{
		vrInt nCount[3] = {0,0,0};
		for (iterAllOf(ci,s_Triangle_Cache))
		{
			nCount[(*ci)->m_TriSetType]++;
		}
		printf("nCount [%d][%d][%d]\n",nCount[0],nCount[1],nCount[2]);vrPause;
	}

	TriangleElemPtr TriangleElem::makeTriangleElem4DualPlus(VertexPtr vtxPtr[], VertexPtr endVtxPtr[], const TriangleSetType type)
	{
		std::vector< TriangleElemPtr >::reverse_iterator itr = std::find_if(s_Triangle_Cache.rbegin(), s_Triangle_Cache.rend(), TriangleElemCompare4DualPlus(vtxPtr));
		if (s_Triangle_Cache.rend() == itr)
		{
			//no find
			TriangleElemPtr curTriangleElemPtr(new TriangleElem(vtxPtr, type));
			s_Triangle_Cache.push_back(curTriangleElemPtr);
			curTriangleElemPtr->setId(s_Triangle_Cache.size() - 1);

			for (int v = 0; v < Geometry::vertexs_per_tri; ++v)
			{
				vtxPtr[v]->addShareTriangleElement(curTriangleElemPtr); 

			}
			curTriangleElemPtr->setEndVtxPtr(endVtxPtr);

			//vrPause;
			return curTriangleElemPtr;
		}
		else
		{
			MyError("same triangle info.");
			return (*itr);
		}
	}



	MyInt TriangleElem::searchVtxIndexByVtxId(const MyInt nId)
	{
		for (int v = 0; v < Geometry::vertexs_per_tri;++v)
		{
			if (nId == (m_elem_vertex[v]->getId()))
			{
				return v;
			}
		}
		return Invalid_Id;
	}

	MyInt TriangleElem::searchVtxIndexByEndVtxId(const MyInt nId)
	{
		for (int v = 0; v < Geometry::vertexs_per_tri;++v)
		{
			if (nId == (m_elem_vertex_endVtx[v]->getId()))
			{
				return v;
			}
		}
		return Invalid_Id;
	}

	MyInt TriangleElem::searchVtxIndexByVtxPos(const MyVec3& pos)
	{
		for (int v = 0; v < Geometry::vertexs_per_tri; ++v)
		{
			if (numbers::isEqual(m_elem_vertex[v]->getPos(), pos))
			{
				return v;
			}
		}
		return Invalid_Id;
	}

	MyFloat TriangleElem::lineShapeFunction(const MyVec2ParamSpace& localCoords, const MyInt n)
	{
		switch (n)
		{
		case 0:
		{
				  return 1.0 - localCoords[0] - localCoords[1];
				  break;
		}
		case 1:
		{
				  return localCoords[0];
				  break;
		}
		case 2:
		{
				  return localCoords[1];
				  break;
		}
		default:
		{
				   vrError("n > 2");
				   return -1.0;
				   break;
		}
		}
	}

	MyFloat TriangleElem::lineDeris(const MyVec2ParamSpace& localCoords, const MyInt n, const MyInt eta)
	{
		vrASSERT((n>=0)&&(n<=2));
		vrASSERT((eta>=0)&&(eta<=1));
		if (0 == n)
		{
			if (0 == eta)
			{
				return -1.0;
			}
			else
			{
				return -1.0;
			}

		}
		else if (1 == n)
		{
			if (0 == eta)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		}
		else 
		{
			if (0 == eta)
			{
				return 0.0;
			}
			else
			{
				return 1.0;
			}
		}
	}

	MyFloat TriangleElem::jacobian(const MyVec3& vtxEndPt0, const MyVec3& vtxEndPt1, const MyVec3& vtxEndPt2)
	{
		//bem-short 4.39
		MyMatrix_2X2 m11, m12, m13;
		const MyFloat x1 = vtxEndPt0[0];
		const MyFloat y1 = vtxEndPt0[1];
		const MyFloat z1 = vtxEndPt0[2];

		const MyFloat x2 = vtxEndPt1[0];
		const MyFloat y2 = vtxEndPt1[1];
		const MyFloat z2 = vtxEndPt1[2];

		const MyFloat x3 = vtxEndPt2[0];
		const MyFloat y3 = vtxEndPt2[1];
		const MyFloat z3 = vtxEndPt2[2];

		m11.coeffRef(0, 0) = (y2-y1);
		m11.coeffRef(0, 1) = (z2-z1);
		m11.coeffRef(1, 0) = (y3-y1);
		m11.coeffRef(1, 1) = (z3-z1);

		m12.coeffRef(0, 0) = (x2 - x1);
		m12.coeffRef(0, 1) = (z2 - z1);
		m12.coeffRef(1, 0) = (x3 - x1);
		m12.coeffRef(1, 1) = (z3 - z1);

		m13.coeffRef(0, 0) = (x2 - x1);
		m13.coeffRef(0, 1) = (y2 - y1);
		m13.coeffRef(1, 0) = (x3 - x1);
		m13.coeffRef(1, 1) = (y3 - y1);

		const MyFloat m11_det = m11.determinant();
		const MyFloat m12_det = m12.determinant();
		const MyFloat m13_det = m13.determinant();

		return std::sqrt(m11_det*m11_det + m12_det*m12_det + m13_det*m13_det);
	}
	
	const MyFloat my_pi = boost::math::constants::pi<MyFloat>();
	const MyFloat E = 1e5;
	const MyFloat mu = 0.3;
	const MyFloat shearMod = E / (2 * (1 + mu));
	
	const MyFloat const1_3d = 16 * my_pi*shearMod * (1 - mu);
	const MyFloat const2_3d = (3 - 4 * mu);
	const MyFloat const3_3d = (8 * my_pi*(1 - mu));
	const MyFloat const4_3d = (1 - 2 * mu);

	TriangleElem::TriangleElem(VertexPtr vertexes[], const TriangleSetType type)
	{
		//m_TriElemType = Continuous;// default Line Element Type
		
		m_elem_vertex[0] = vertexes[0];
		m_elem_vertex[1] = vertexes[1];
		m_elem_vertex[2] = vertexes[2];

		m_TriSetType = type;

		m_TriElemType = TriElemType::DisContinuous;
		m_ElemDisContinuousType = dis_regular;
#if USE_VDB
		m_tri4RegionId = Invalid_Id;
#endif
	}

	TriangleElem::~TriangleElem()
	{}

	void TriangleElem::computeTriElemContinuousType()
	{
		bool isDisContinuousVertex[Geometry::vertexs_per_tri];
		for (int i=0;i<Geometry::vertexs_per_tri;++i)
		{
			isDisContinuousVertex[i] = getVertex(i)->isDisContinuousVertex();
		}

		if ( isDisContinuousVertex[0] || 
			 isDisContinuousVertex[1] || 
			 isDisContinuousVertex[2] )
		{
			m_TriElemType = DisContinuous;
			
			if (!isDisContinuousVertex[0] && isDisContinuousVertex[1] && !isDisContinuousVertex[2])
			{
				m_ElemDisContinuousType = dis_1_1;
				//printf("elem id of dis 1 1 [%d]\n",getID());
			}
			else if (!isDisContinuousVertex[0] && !isDisContinuousVertex[1] && isDisContinuousVertex[2])
			{
				m_ElemDisContinuousType = dis_1_2;
			}
			else if (isDisContinuousVertex[0] && !isDisContinuousVertex[1] && !isDisContinuousVertex[2])
			{
				m_ElemDisContinuousType = dis_1_3;
				//printf("elem id of dis 1 3 [%d]\n",getID());
			}
			else if (!isDisContinuousVertex[0] && isDisContinuousVertex[1] && isDisContinuousVertex[2])
			{
				m_ElemDisContinuousType = dis_2_3;
			}
			else if (isDisContinuousVertex[0] && isDisContinuousVertex[1] && !isDisContinuousVertex[2])
			{
				m_ElemDisContinuousType = dis_2_2;
			}
			else if (isDisContinuousVertex[0] && !isDisContinuousVertex[1] && isDisContinuousVertex[2])
			{
				m_ElemDisContinuousType = dis_2_1;
			}
			else if (isDisContinuousVertex[0] && isDisContinuousVertex[1] && isDisContinuousVertex[2])
			{
				m_ElemDisContinuousType = dis_3_3;
				//printf("elem id of dis 3 3 [%d]\n",getID());
			}
			else
			{
				MyError("Un-support discontinuous type!");
			}
		}
		else
		{
			m_ElemDisContinuousType = dis_regular;
			m_TriElemType = Continuous;
		}
	}

	void TriangleElem::get_dof_indices(MyVector9I &vecDofs)
	{
		for (int v = 0; v < MyDim;++v)
		{
			vecDofs.block(v*MyDim,0,MyDim,1) = getVertex(v)->getDofs();
		}
	}

	MyVec3 TriangleElem::makeDistance_r_positive(const MyVec3 & r)
	{
		return MyVec3(std::fabs(r.x()),std::fabs(r.y()),std::fabs(r.z()));
	}

	void TriangleElem::getKernelParameters(const MyInt gpt/*0,1,2*/, const MyVec3& source_global, MyFloat& JxW, MyVec3& fieldNormals, MyFloat& r, MyVec3& dr, MyFloat& drdn)
	{
		
		MyVec3 fieldPt /*gauss point*/= shapeFunctionN.coeff(gpt, 0)*(getElemVtxPoint(0)) +
			shapeFunctionN.coeff(gpt, 1)*(getElemVtxPoint(1)) + shapeFunctionN.coeff(gpt, 2)*(getElemVtxPoint(2));

		JxW = JxW_values[gpt];
		fieldNormals = m_TriElemNormal;

		MyVec3 relDist = fieldPt - source_global;
		r = relDist.norm();
		//relDist = makeDistance_r_positive(relDist); 距离应该是正负值

		//vrASSERT(r > 0.01f);
		dr = (1.0 / r) * relDist;

		drdn = dr.transpose() * fieldNormals;
	}
	
	void TriangleElem::calculateJumpTerm_smooth(MyMatrix_3X3& jumpTerm)
	{
		jumpTerm = MyMatrix_3X3::Identity();
		jumpTerm *= 0.5;
	}

#if USE_Mantic_CMat
	vrFloat theta_bar(const vrFloat theta, const vrFloat theta_0)
	{
		return theta - theta_0;
	}
	vrFloat compute_angle_ret_cos(const MyVec2ParamSpace& vec_a, const MyVec2ParamSpace& vec_b)
	{
		//could not use abs in dot product
		return (vec_a.dot(vec_b))/(vec_a.norm()*vec_b.norm());
	}

	vrFloat compute_angle_ret_cos(const MyVec3& vec_a, const MyVec3& vec_b)
	{
		return (vec_a.dot(vec_b))/(vec_a.norm()*vec_b.norm());
	}

	vrFloat rho_hat_with_bar(const vrFloat h, const vrFloat theta_bar_arc)
	{
		/*if ( std::abs(std::abs(theta_bar_arc) - numbers::MyPI / 2.0) < 0.1 )
		{
			printf("h/cos(theta_bar_arc) =[%f] [theta_bar_arc=%f] [90 degrees = %f]\n",h/cos(theta_bar_arc), theta_bar_arc, numbers::MyPI / 2.0);vrPause;
		}*/
		/*if ((h/cos(theta_bar_arc)) > 10.0)
		{
			printf("h/cos(theta_bar_arc) =[%f] [theta_bar_arc=%f] [90 degrees = %f]\n",h/cos(theta_bar_arc), theta_bar_arc, numbers::MyPI / 2.0);vrPause;
		}*/
		return h/cos(theta_bar_arc);
	}

	void TriangleElem::compute_Shape_Deris_Jacobi_SST_3D()
	{
#if USE_UniformSampling
		MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];
		const DisContinuousType curTriElemDisContinuousType = getTriContinuousType();
		for (int srcPtIdx=0; srcPtIdx < Geometry::vertexs_per_tri; ++srcPtIdx)
		{
			const DisContinuousType tmpTriElemDisContinuousType = 
				TriangleElemData_DisContinuous::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);

			for (int v=0;v<Geometry::vertexs_per_tri;++v)
			{
				//srcPt_SST_LookUp[v] = srcPtIdx;
				vtx_globalCoord[v] = getVertex((srcPtIdx+v) % Geometry::vertexs_per_tri)->getPos();
			}
			m_data_SST_3D_Elem[srcPtIdx].compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
		}
#else//USE_UniformSampling
		MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];
		for (int v=0;v<Geometry::vertexs_per_tri;++v)
		{
			vtx_globalCoord[v] = m_elem_vertex[v]->getPos();
		}
		m_data_SST_3D.compute_Shape_Deris_Jacobi_SST_3D( getTriContinuousType(), vtx_globalCoord);

#endif//USE_UniformSampling
	}

	vrFloat shapefunction_xi_regular(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return 1-xi_1-xi_2;
				break;
			}
		case 1:
			{
				return xi_1;
				break;
			}
		case 2:
			{
				return xi_2;
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}
	


	vrFloat derisShapefunction_xi_regular(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();
		switch(idx)
		{
		case 0:
			{
				//return 1-xi_1-xi_2;
				switch(idx_local)
				{
				case 0:
					{
						return -1.0;
						break;
					}
				case 1:
					{
						return -1.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				//return xi_1;
				switch(idx_local)
				{
				case 0:
					{
						return 1.0;
						break;
					}
				case 1:
					{
						return 0.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				//return xi_2;
				switch(idx_local)
				{
				case 0:
					{
						return 0.0;
						break;
					}
				case 1:
					{
						return 1.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}



	bool areEqualRel(vrFloat a, vrFloat b, vrFloat epsilon) {
		return (fabs(a - b) <= epsilon * std::max(fabs(a), fabs(b)));
	}

	MyVec3 compute_UnitNormal(const MyVec3 vtx_globalCoord[])
	{
		MyVec3 triNormal = (vtx_globalCoord[1] - vtx_globalCoord[0]).cross(vtx_globalCoord[2] - vtx_globalCoord[0]);
		triNormal.normalize();
		return triNormal;
	}



	MyVec2ParamSpace TriangleElemData_DisContinuous::pc2xi(const MyVec2ParamSpace& srcImage, const MyVec2ParamSpace& pc/*(rho,theta)*/)
	{
		const vrFloat rho = pc[TriangleElem::idx_rho_doubleLayer];
		const vrFloat theta = pc[TriangleElem::idx_theta_doubleLayer];
		
		return MyVec2ParamSpace(srcImage[0] MYNOTICE + rho*cos(theta),srcImage[1] MYNOTICE + rho*sin(theta));
	}

	MyVec2ParamSpace TriangleElemData_DisContinuous::pc2eta(const MyVec2ParamSpace& srcImage, const MyVec2ParamSpace& pc)
	{
		const vrFloat rho = pc[TriangleElem::idx_rho_doubleLayer];
		const vrFloat theta = pc[TriangleElem::idx_theta_doubleLayer];
		
		return MyVec2ParamSpace(srcImage[0] MYNOTICE + rho*cos(theta),srcImage[1] MYNOTICE + rho*sin(theta));
	}

	void TriangleElemData_DisContinuous::testAssist(DisContinuousType currentContinuousType, const MyMatrix& tmp_gaussQuadrature_xi_polar, MyVec2ParamSpace paramCoordsInDiscontinuous[])
	{
		

		
#if 1
		for (int idx_i=0;idx_i<Geometry::vertexs_per_tri;++idx_i)
		{
			for (int idx_j=0;idx_j<Geometry::vertexs_per_tri;++idx_j)
			{
				const vrFloat i_j = delta_ij(idx_i, idx_j);
				//printf("i_j[%f] shp[%f]\n",i_j, s_shapefunction_xi(currentContinuousType, idx_i, paramCoordsInDiscontinuous[idx_j]));
				Q_ASSERT( numbers::isEqual(i_j, s_shapefunction_xi(currentContinuousType, idx_i, paramCoordsInDiscontinuous[idx_j])));
				
			}
		}

		const vrInt gaussPtSize = tmp_gaussQuadrature_xi_polar.rows();

		MyVec2ParamSpace cur_xi;
		vrFloat weight = 0.0;
		for (int gpt=0;gpt<gaussPtSize;++gpt)
		{
			cur_xi[TriangleElem::idx_theta_doubleLayer] = tmp_gaussQuadrature_xi_polar.coeff(gpt,TriangleElem::idx_theta_doubleLayer);
			cur_xi[TriangleElem::idx_rho_doubleLayer] = tmp_gaussQuadrature_xi_polar.coeff(gpt,TriangleElem::idx_rho_doubleLayer);
			cur_xi = pc2xi(MyVec2ParamSpace(0.0,0.0) ,cur_xi);
			weight = 0.0;
			for (int idx_i = 0;idx_i < Geometry::vertexs_per_tri;++idx_i)
			{
				//vrFloat tmpWeight = s_shapefunction_xi(currentContinuousType, idx_i, cur_xi);
				//if (tmpWeight < 0.0)
				//{
				//	printf("[%d] shp(%f,%f)=%f \n", currentContinuousType, cur_xi[0], cur_xi[1], tmpWeight); //vrPause;
				//}
				/*printf("general [%f]---- [%f]\n",shp.shape_function(cur_xi, (GeneralShapeFunction_TriangleElement::globalCoord)idx_i),
					s_shapefunction_xi(currentContinuousType, idx_i, cur_xi));*/
				
				weight += s_shapefunction_xi(currentContinuousType, idx_i, cur_xi);

				
			}

			if (!numbers::isEqual(1.0,weight))
			{
				printf("weight [%f]  currentContinuousType[%d] cur_xi(%f,%f)\n",weight, currentContinuousType,cur_xi[0],cur_xi[1]);vrPause;
			}
			//Q_ASSERT(numbers::isEqual(1.0,weight));
		}
#endif
	}

	vrVec3 TriangleElemData_DisContinuous::interpolation_displacement(const vrInt idx, vrVec3 srcDisp[])
	{
		//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
		
		static MyVec2ParamSpace s_xi[3] = {MyVec2ParamSpace(0.0,0.0),MyVec2ParamSpace(1.0,0.0),MyVec2ParamSpace(0.0,1.0)};


		return shapefunction_xi(0, s_xi[idx]) * srcDisp[0] + 
			   shapefunction_xi(1, s_xi[idx]) * srcDisp[1] + 
			   shapefunction_xi(2, s_xi[idx]) * srcDisp[2];
		/*
		return shapefunction_xi( 0, xi)*m_vtx_globalCoord[0]
		+ shapefunction_xi( 1, xi)*m_vtx_globalCoord[1]
		+ shapefunction_xi( 2, xi)*m_vtx_globalCoord[2];
		*/
	}

	void TriangleElemData_DisContinuous::TestShapeFunction()
	{
#if 0

		MyVector gaussPoint_xi_In_Theta, gaussPoint_xi_In_Theta_Weight, gaussPoint_xi_In_Rho[TriangleElem::GaussPointSize_xi_In_Theta],gaussPoint_xi_In_Rho_Weight[TriangleElem::GaussPointSize_xi_In_Theta];
		const vrFloat theta_xi = my_pi / 2.0; //90 degree in radian
		const vrFloat h_Pedal_tmp = sqrt(2.0) / 2.0;
		const vrFloat theta_0 = my_pi / 4.0;
		lgwt(TriangleElem::GaussPointSize_xi_In_Theta,0.0/*arc*/,theta_xi,gaussPoint_xi_In_Theta,gaussPoint_xi_In_Theta_Weight);
		//printf("gaussPoint_xi_In_Theta_Weight sum %f\n",gaussPoint_xi_In_Theta_Weight.sum());
		for (int v=0;v<TriangleElem::GaussPointSize_xi_In_Theta;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_In_Theta[v];
			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_In_Rho_Weight[v];

			const vrFloat cur_rho_hat = rho_hat_with_bar(h_Pedal_tmp,cur_theta-theta_0);
			Q_ASSERT(cur_rho_hat > 0.0);
			lgwt(TriangleElem::GaussPointSize_xi_In_Rho,0.0,cur_rho_hat,cur_gaussPointIn_Rho,cur_gaussPointIn_Rho_Weight);
			//printf("cur_gaussPointIn_Rho_Weight[%d] sum %f\n",v,cur_gaussPointIn_Rho_Weight.sum());
		}

		MyMatrix tmp_gaussQuadrature_xi_polar;
		tmp_gaussQuadrature_xi_polar.resize(TriangleElem::GaussPointSize_xi_In_Theta*TriangleElem::GaussPointSize_xi_In_Rho,3);
		tmp_gaussQuadrature_xi_polar.setZero();

		for (int v=0,rowIdx=0;v<TriangleElem::GaussPointSize_xi_In_Theta;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_In_Theta[v];
			const vrFloat cur_theta_weight = gaussPoint_xi_In_Theta_Weight[v];

			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_In_Rho_Weight[v];

			for (int r=0;r<TriangleElem::GaussPointSize_xi_In_Rho;++r,++rowIdx)
			{
				tmp_gaussQuadrature_xi_polar.coeffRef(rowIdx,TriangleElem::idx_theta) = cur_theta;
				tmp_gaussQuadrature_xi_polar.coeffRef(rowIdx,TriangleElem::idx_rho) = cur_gaussPointIn_Rho[r];
				tmp_gaussQuadrature_xi_polar.coeffRef(rowIdx,TriangleElem::idx_weight) = cur_theta_weight * cur_gaussPointIn_Rho_Weight[r];
			}
		}

		DisContinuousType currentContinuousType;
		//MyVec2ParamSpace paramCoordsInDiscontinuous[MyDim] = 
		//{ MyVec2ParamSpace(s2, s2), MyVec2ParamSpace(s1, s2), MyVec2ParamSpace(s2, s1) };
		{
			currentContinuousType = dis_regular;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(0.0, 0.0);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(1.0, 0.0);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(0.0, 1.0);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}
		{
			currentContinuousType = dis_1_1;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(0.0, 0.0);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(s1, s2);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(0.0, 1.0);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}
		{
			currentContinuousType = dis_1_2;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(0.0, 0.0);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(1.0, 0.0);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(s2, s1);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}
		{
			currentContinuousType = dis_1_3;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(s2, s2);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(1.0, 0.0);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(0.0, 1.0);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}
		{
			currentContinuousType = dis_2_3;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(0.0, 0.0);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(s1, s2);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(s2, s1);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}
		{
			currentContinuousType = dis_2_2;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(s2, s2);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(s1, s2);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(0.0, 1.0);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}
		{
			currentContinuousType = dis_2_1;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(s2, s2);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(1.0, 0.0);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(s2, s1);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}
		{
			currentContinuousType = dis_3_3;
			MyVec2ParamSpace paramCoordsInDiscontinuous[Geometry::vertexs_per_tri]; 
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(s2, s2);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(s1, s2);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(s2, s1);
			testAssist(currentContinuousType, tmp_gaussQuadrature_xi_polar, paramCoordsInDiscontinuous);
		}

#endif
	}
#endif

#if 1
	
	MyVec2ParamSpace TriangleElemData_DisContinuous::s_paramCoordsInDiscontinuous[9][MyDim]=
	{
		/*dis_regular*/{MyVec2ParamSpace(0,0),MyVec2ParamSpace(0,0),MyVec2ParamSpace(0,0)},
		/*dis_1_1*/{MyVec2ParamSpace(0,0),MyVec2ParamSpace(s1,s2),MyVec2ParamSpace(0,1)},
		/*dis_1_2*/{MyVec2ParamSpace(0,0),MyVec2ParamSpace(1,0),MyVec2ParamSpace(s2,s1)},
		/*dis_1_3*/{MyVec2ParamSpace(s2,s2),MyVec2ParamSpace(1,0),MyVec2ParamSpace(0,1)},
		/*dis_2_3*/{MyVec2ParamSpace(0,0),MyVec2ParamSpace(s1,s2),MyVec2ParamSpace(s2,s1)},
		/*dis_2_2*/{MyVec2ParamSpace(s2,s2),MyVec2ParamSpace(s1,s2),MyVec2ParamSpace(0,1)},
		/*dis_2_1*/{MyVec2ParamSpace(s2,s2),MyVec2ParamSpace(1,0),MyVec2ParamSpace(s2,s1)},
		/*dis_3_3*/{MyVec2ParamSpace(s2,s2),MyVec2ParamSpace(s1,s2),MyVec2ParamSpace(s2,s1)},
		/*dis_regular*/{MyVec2ParamSpace(0,0),MyVec2ParamSpace(1,0),MyVec2ParamSpace(0,1)}
	};
	vrFloat shapefunction_xi_dis_1_1(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return ((s2-1.0) / s1) * xi_1 - xi_2 + 1.0;
				break;
			}
		case 1:
			{
				return (1.0/s1) * xi_1;
				break;
			}
		case 2:
			{
				return -1.0 * (s2 / s1) * xi_1 + xi_2;
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}

	vrFloat shapefunction_xi_dis_1_2(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return -1.0 * xi_1 +((s2-1.0)/s1)* xi_2 + 1.0;
				break;
			}
		case 1:
			{
				return xi_1 - (s2/s1)*xi_2;
				break;
			}
		case 2:
			{
				return (1.0 / s1) * xi_2;
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}

	vrFloat shapefunction_xi_dis_1_3(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return (1.0 / (2.0*s2-1.0)) * xi_1 + (1.0 / (2.0*s2-1.0)) * xi_2 - (1.0 / (2.0*s2-1.0));
				break;
			}
		case 1:
			{
				return ((s2-1.0) / (2.0*s2-1.0))*xi_1 - (s2 / (2.0*s2-1.0))*xi_2 + (s2 / (2.0*s2-1.0));
				break;
			}
		case 2:
			{
				return -1.0 * (s2 / (2.0*s2-1.0)) * xi_1 + ((s2-1.0) / (2.0*s2-1.0)) * xi_2 + (s2 / (2.0*s2-1.0));
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}

	vrFloat shapefunction_xi_dis_2_1(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return (s1/((s1-s2)*(s2-1.0))) * xi_1 - (1.0/(s1-s2)) *xi_2 - (s1/((s1-s2)*(s2-1.0)));
				break;
			}
		case 1:
			{
				return -1.0 * (1.0 / (s2 - 1.0)) * xi_1 + (s2 / (s2 - 1.0));
				break;
			}
		case 2:
			{
				return -1.0 * (s2 / ((s1-s2)*(s2-1.0))) * xi_1 + (1.0 / (s1-s2)) * xi_2 + (s2 / ((s1-s2)*(s2-1.0)));
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}

	vrFloat shapefunction_xi_dis_2_2(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return -1.0 * (1.0 / (s1-s2)) * xi_1 + (s1 / ((s1-s2)*(s2-1.0)))*xi_2 - (s1)/((s1-s2)*(s2-1.0));
				break;
			}
		case 1:
			{
				return (1.0 / (s1-s2)) * xi_1 - ((s2)/((s1-s2)*(s2-1.0)))*xi_2 + ((s2)/((s1-s2)*(s2-1.0)));
				break;
			}
		case 2:
			{
				return -1.0 * (1.0 / (s2-1.0))*xi_2 + s2 / (s2-1.0);
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}

	vrFloat shapefunction_xi_dis_2_3(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return -1.0 * (1.0 / (s1 + s2)) * xi_1 - (1.0 / (s1 + s2)) * xi_2 + 1.0;
				break;
			}
		case 1:
			{
				return (s1 / (s1*s1 - s2*s2)) * xi_1 - (s2 / (s1*s1 - s2*s2)) * xi_2;
				break;
			}
		case 2:
			{
				return -1.0 * (s2 / (s1*s1 - s2*s2)) * xi_1 + (s1 / (s1*s1 - s2*s2)) * xi_2;
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}

	vrFloat shapefunction_xi_dis_3_3(vrInt idx,const MyVec2ParamSpace& xi)
	{
		const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();

		switch(idx)
		{
		case 0:
			{
				return -1.0 * (1.0/(s1-s2)) * xi_1 - (1.0/(s1-s2)) * xi_2 + (s1 + s2)/(s1-s2);
				break;
			}
		case 1:
			{
				return (1.0 / (s1-s2))* xi_1 - (s2 / (s1 - s2));
				break;
			}
		case 2:
			{
				return (1.0 / (s1-s2)) * xi_2 - (s2 / (s1 - s2));
				break;
			}
		default:
			{
				MyError("unsupport index in shapefunction_xi");
				return 0.0;
				break;
			}
		}
	}

	vrFloat TriangleElemData_DisContinuous::delta_ij(vrInt i, vrInt j)
	{
		if (i == j)
		{
			return 1.0;
		}
		else
		{
			return 0.0;
		}
	}

	vrFloat derisShapefunction_xi_dis_1_1(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		/*const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();*/
		switch(idx)
		{
		case 0:
			{
				// ((s2-1.0) / s1) * xi_1 - xi_2 + 1.0;
				switch(idx_local)
				{
				case 0:
					{
						return ((s2-1.0) / s1);
						break;
					}
				case 1:
					{
						return -1.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_1");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				// (1.0/s1) * xi_1
				switch(idx_local)
				{
				case 0:
					{
						return (1.0/s1);
						break;
					}
				case 1:
					{
						return 0.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_1");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				// -1.0 * (s2 / s1) * xi_1 + xi_2;
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (s2 / s1);
						break;
					}
				case 1:
					{
						return 1.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_1");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in derisShapefunction_xi_dis_1_1");
				return 0.0;
				break;
			}
		}
	}

	vrFloat derisShapefunction_xi_dis_1_2(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		/*const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();*/
		switch(idx)
		{
		case 0:
			{
				// -1.0 * xi_1 +((s2-1.0)/s1)* xi_2 + 1.0;
				switch(idx_local)
				{
				case 0:
					{
						return -1.0;
						break;
					}
				case 1:
					{
						return ((s2-1.0)/s1);
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_2");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				// xi_1 - (s2/s1)*xi_2;
				switch(idx_local)
				{
				case 0:
					{
						return 1.0;
						break;
					}
				case 1:
					{
						return -1.0 * (s2/s1);
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_2");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				// (1.0 / s1) * xi_2;
				switch(idx_local)
				{
				case 0:
					{
						return 0.0;
						break;
					}
				case 1:
					{
						return (1.0 / s1);
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_2");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in derisShapefunction_xi_dis_1_2");
				return 0.0;
				break;
			}
		}
	}

	vrFloat derisShapefunction_xi_dis_1_3(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		/*const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();*/
		switch(idx)
		{
		case 0:
			{
				// (1.0 / (2.0*s2-1.0)) * xi_1 + (1.0 / (2.0*s2-1.0)) * xi_2 - (1.0 / (2.0*s2-1.0));
				switch(idx_local)
				{
				case 0:
					{
						return (1.0 / (2.0*s2-1.0));
						break;
					}
				case 1:
					{
						return (1.0 / (2.0*s2-1.0));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				// ((s2-1.0) / (2.0*s2-1.0))*xi_1 - (s2 / (2.0*s2-1.0))*xi_2 + (s2 / (2.0*s2-1.0));
				switch(idx_local)
				{
				case 0:
					{
						return ((s2-1.0) / (2.0*s2-1.0));
						break;
					}
				case 1:
					{
						return -1.0 * (s2 / (2.0*s2-1.0));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				// -1.0 * (s2 / (2.0*s2-1.0)) * xi_1 + ((s2-1.0) / (2.0*s2-1.0)) * xi_2 + (s2 / (2.0*s2-1.0));
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (s2 / (2.0*s2-1.0));
						break;
					}
				case 1:
					{
						return ((s2-1.0) / (2.0*s2-1.0));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_1_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in derisShapefunction_xi_dis_1_3");
				return 0.0;
				break;
			}
		}
	}

	vrFloat derisShapefunction_xi_dis_2_3(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		/*const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();*/
		switch(idx)
		{
		case 0:
			{
				// -1.0 * (1.0 / (s1 + s2)) * xi_1 - (1.0 / (s1 + s2)) * xi_2 + 1.0;
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (1.0 / (s1 + s2));
						break;
					}
				case 1:
					{
						return -1.0 * (1.0 / (s1 + s2));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				// (s1 / (s1*s1 - s2*s2)) * xi_1 - (s2 / (s1*s1 - s2*s2)) * xi_2;
				switch(idx_local)
				{
				case 0:
					{
						return (s1 / (s1*s1 - s2*s2));
						break;
					}
				case 1:
					{
						return -1.0 * (s2 / (s1*s1 - s2*s2));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				// -1.0 * (s2 / (s1*s1 - s2*s2)) * xi_1 + (s1 / (s1*s1 - s2*s2)) * xi_2;
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (s2 / (s1*s1 - s2*s2));
						break;
					}
				case 1:
					{
						return (s1 / (s1*s1 - s2*s2));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in derisShapefunction_xi_dis_2_3");
				return 0.0;
				break;
			}
		}
	}

	vrFloat derisShapefunction_xi_dis_2_2(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		/*const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();*/
		switch(idx)
		{
		case 0:
			{
				// -1.0 * (1.0 / (s1-s2)) * xi_1 + (s1 / ((s1-s2)*(s2-1.0)))*xi_2 - (s1)/((s1-s2)*(s2-1.0));
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (1.0 / (s1-s2));
						break;
					}
				case 1:
					{
						return (s1 / ((s1-s2)*(s2-1.0)));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_2");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				// (1.0 / (s1-s2)) * xi_1 - ((s2)/((s1-s2)*(s2-1.0)))*xi_2 + ((s2)/((s1-s2)*(s2-1.0)));
				switch(idx_local)
				{
				case 0:
					{
						return (1.0 / (s1-s2)) ;
						break;
					}
				case 1:
					{
						return -1.0 * ((s2)/((s1-s2)*(s2-1.0)));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_2");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				// -1.0 * (1.0 / (s2-1.0))*xi_2 + s2 / (s2-1.0);
				switch(idx_local)
				{
				case 0:
					{
						return 0.0;
						break;
					}
				case 1:
					{
						return -1.0 * (1.0 / (s2-1.0));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_2");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in derisShapefunction_xi_dis_2_2");
				return 0.0;
				break;
			}
		}
	}

	vrFloat derisShapefunction_xi_dis_2_1(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		/*const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();*/
		switch(idx)
		{
		case 0:
			{
				// (s1/((s1-s2)*(s2-1.0))) * xi_1 - (1.0/(s1-s2)) *xi_2 - (s1/((s1-s2)*(s2-1.0)));
				switch(idx_local)
				{
				case 0:
					{
						return (s1/((s1-s2)*(s2-1.0)));
						break;
					}
				case 1:
					{
						return -1.0 * (1.0/(s1-s2));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_1");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				// -1.0 * (1.0 / (s2 - 1.0)) * xi_1 + (s2 / (s2 - 1.0));
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (1.0 / (s2 - 1.0)) ;
						break;
					}
				case 1:
					{
						return 0.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_1");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				// -1.0 * (s2 / ((s1-s2)*(s2-1.0))) * xi_1 + (1.0 / (s1-s2)) * xi_2 + (s2 / ((s1-s2)*(s2-1.0)));
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (s2 / ((s1-s2)*(s2-1.0)));
						break;
					}
				case 1:
					{
						return (1.0 / (s1-s2));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_2_1");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in derisShapefunction_xi_dis_2_1");
				return 0.0;
				break;
			}
		}
	}

	vrFloat derisShapefunction_xi_dis_3_3(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		/*const vrFloat& xi_1 = xi.x();
		const vrFloat& xi_2 = xi.y();*/
		switch(idx)
		{
		case 0:
			{
				// -1.0 * (1.0/(s1-s2)) * xi_1 - (1.0/(s1-s2)) * xi_2 + (s1 + s2)/(s1-s2);
				switch(idx_local)
				{
				case 0:
					{
						return -1.0 * (1.0/(s1-s2));
						break;
					}
				case 1:
					{
						return -1.0 * (1.0/(s1-s2));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_3_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 1:
			{
				// (1.0 / (s1-s2))* xi_1 - (s2 / (s1 - s2));
				switch(idx_local)
				{
				case 0:
					{
						return (1.0 / (s1-s2)) ;
						break;
					}
				case 1:
					{
						return 0.0;
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_3_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		case 2:
			{
				// (1.0 / (s1-s2)) * xi_2 - (s2 / (s1 - s2));
				switch(idx_local)
				{
				case 0:
					{
						return 0.0;
						break;
					}
				case 1:
					{
						return (1.0 / (s1-s2));
						break;
					}
				default:
					{
						MyError("unsupport index in derisShapefunction_xi_dis_3_3");
						return 0.0;
						break;
					}
				}
				break;
			}
		default:
			{
				MyError("unsupport index in derisShapefunction_xi_dis_3_3");
				return 0.0;
				break;
			}
		}
	}
	vrFloat TriangleElemData_DisContinuous::s_derisShapefunction_xi(DisContinuousType currentContinuousType, vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)
	{
		switch (currentContinuousType)
		{
		case dis_regular:
			{
				//MyError("dis_regular in TriangleElemData_DisContinuous::s_derisShapefunction_xi.");
				
				return derisShapefunction_xi_regular(idx, idx_local, xi);
			}
		case dis_1_1:
			{
				return derisShapefunction_xi_dis_1_1(idx, idx_local, xi);
			}
		case dis_1_2:
			{
				return derisShapefunction_xi_dis_1_2(idx, idx_local, xi);
			}
		case dis_1_3:
			{
				return derisShapefunction_xi_dis_1_3(idx, idx_local, xi);
			}
		case dis_2_3:
			{
				return derisShapefunction_xi_dis_2_3(idx, idx_local, xi);
			}
		case dis_2_2:
			{
				return derisShapefunction_xi_dis_2_2(idx, idx_local, xi);
			}
		case dis_2_1:
			{
				return derisShapefunction_xi_dis_2_1(idx, idx_local, xi);
			}
		case dis_3_3:
			{
				return derisShapefunction_xi_dis_3_3(idx, idx_local, xi);
			}
		default:
			{
				MyError("un support triangle type in TriangleElemData::derisShapefunction_xi.");
			}
		}
	}

	vrFloat TriangleElemData_DisContinuous::s_shapefunction_xi(DisContinuousType continuousType, vrInt idx,const MyVec2ParamSpace& xi)
	{
		//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
		switch (continuousType)
		{
		case dis_regular:
			{
				//MyError("dis_regular in TriangleElemData_DisContinuous::s_shapefunction_xi.");
				return shapefunction_xi_regular(idx, xi);
			}
		case dis_1_1:
			{
				return shapefunction_xi_dis_1_1(idx, xi);
			}
		case dis_1_2:
			{
				return shapefunction_xi_dis_1_2(idx, xi);
			}
		case dis_1_3:
			{
				return shapefunction_xi_dis_1_3(idx, xi);
			}
		case dis_2_3:
			{
				return shapefunction_xi_dis_2_3(idx, xi);
			}
		case dis_2_2:
			{
				return shapefunction_xi_dis_2_2(idx, xi);
			}
		case dis_2_1:
			{
				return shapefunction_xi_dis_2_1(idx, xi);
			}
		case dis_3_3:
			{
				return shapefunction_xi_dis_3_3(idx, xi);
			}
		default:
			{
				printf("continuousType [%d]\n",continuousType);
				MyError("un support triangle type in TriangleElemData::shapefunction_xi.");
			}
		}
	}

	vrFloat TriangleElemData_DisContinuous::shapefunction_xi(vrInt idx,const MyVec2ParamSpace& xi)const
	{
		return s_shapefunction_xi(m_DisContinuousType,idx,xi);
	}

	vrFloat TriangleElemData_DisContinuous::derisShapefunction_xi(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi)const
	{
		return s_derisShapefunction_xi(m_DisContinuousType,idx,idx_local,xi);
	}

	

	void TriangleElemData_DisContinuous::compute_Shape_Deris_Jacobi_SST_3D(DisContinuousType type, const MyVec3 vtx_globalCoord[])
	{
		m_DisContinuousType = type;
		//Q_ASSERT(m_DisContinuousType != dis_regular);

		m_vtx_globalCoord[0]=vtx_globalCoord[0];
		m_vtx_globalCoord[1]=vtx_globalCoord[1];
		m_vtx_globalCoord[2]=vtx_globalCoord[2];
		//compute unit normal of field point
		unitNormal_fieldPt = compute_UnitNormal(m_vtx_globalCoord);
		//XI SPACE
		compute_Shape_Deris_Jacobi_SST_3D_xi();

		switch (type)
		{
		case dis_regular:
		case dis_1_1:
		case dis_1_2:
		case dis_2_3:
			{
#if DEBUG_3_3
				compute_Shape_Deris_Jacobi_SST_3D_eta( );
#endif
			
				break;
			}
		case dis_1_3:
		case dis_2_2:
		case dis_2_1:
		case dis_3_3:
			{
				compute_Shape_Deris_Jacobi_SST_3D_eta( );
				compute_Shape_Deris_Jacobi_SST_3D_eta_DisContinuousPt( );
				break;
			}
		}
		//ETA space
		
	}

	void TriangleElemData_DisContinuous::compute_Shape_Deris_Jacobi_SST_3D_xi()
	{
#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
		//1. gauss point
		MyVector gaussPoint_xi_In_Theta, gaussPoint_xi_In_Theta_Weight;
		std::vector< MyVector > gaussPoint_xi_In_Rho; gaussPoint_xi_In_Rho.resize(nGaussPointSize_xi_In_Theta);
		std::vector< MyVector > gaussPoint_xi_In_Rho_Weight; gaussPoint_xi_In_Rho_Weight.resize(nGaussPointSize_xi_In_Theta);
		const vrFloat theta_xi = my_pi / 2.0; //90 degree in radian
		const vrFloat h_Pedal_tmp = sqrt(2.0) / 2.0;
		const vrFloat theta_0 = my_pi / 4.0;
		lgwt(nGaussPointSize_xi_In_Theta,0.0/*arc*/,theta_xi,gaussPoint_xi_In_Theta,gaussPoint_xi_In_Theta_Weight);
		//printf("gaussPoint_xi_In_Theta_Weight sum %f\n",gaussPoint_xi_In_Theta_Weight.sum());
		for (int v=0;v<nGaussPointSize_xi_In_Theta;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_In_Theta[v];
			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_In_Rho_Weight[v];

			const vrFloat cur_rho_hat = rho_hat_with_bar(h_Pedal_tmp,cur_theta-theta_0);
			Q_ASSERT(cur_rho_hat > 0.0);
			lgwt(nGaussPointSize_xi_In_Rho,0.0,cur_rho_hat,cur_gaussPointIn_Rho,cur_gaussPointIn_Rho_Weight);
			//printf("cur_gaussPointIn_Rho_Weight[%d] sum %f\n",v,cur_gaussPointIn_Rho_Weight.sum());
		}

		m_gaussQuadrature_xi_polar.resize(nGaussPointSize_xi_In_Theta*nGaussPointSize_xi_In_Rho,3);
		m_gaussQuadrature_xi_polar.setZero();

		for (int v=0,rowIdx=0;v<nGaussPointSize_xi_In_Theta;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_In_Theta[v];
			const vrFloat cur_theta_weight = gaussPoint_xi_In_Theta_Weight[v];

			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_In_Rho_Weight[v];

			for (int r=0;r<nGaussPointSize_xi_In_Rho;++r,++rowIdx)
			{
				m_gaussQuadrature_xi_polar.coeffRef(rowIdx,TriangleElem::idx_theta_doubleLayer) = cur_theta;
				m_gaussQuadrature_xi_polar.coeffRef(rowIdx,TriangleElem::idx_rho_doubleLayer) = cur_gaussPointIn_Rho[r];
				m_gaussQuadrature_xi_polar.coeffRef(rowIdx,TriangleElem::idx_weight_doubleLayer) = cur_theta_weight * cur_gaussPointIn_Rho_Weight[r];
			}
		}

		//jacobi
		MyVec2ParamSpace xi_nouse;
		/* 2. Jacobi u1=@x/@xi_1 */u1_xi = derisShapefunction_xi(0,0,xi_nouse) * m_vtx_globalCoord[0] + 
			derisShapefunction_xi(1,0,xi_nouse) * m_vtx_globalCoord[1] + 
			derisShapefunction_xi(2,0,xi_nouse) * m_vtx_globalCoord[2] ; 
		/* 3. Jacobi u2=@x/@xi_2 */u2_xi =  derisShapefunction_xi(0,1,xi_nouse) * m_vtx_globalCoord[0] + 
			derisShapefunction_xi(1,1,xi_nouse) * m_vtx_globalCoord[1] + 
			derisShapefunction_xi(2,1,xi_nouse) * m_vtx_globalCoord[2] ; 
		/* 4. Jacobi u1 * u2 */u1_xi_cross_u2_xi = u1_xi.cross(u2_xi);
		/* 4. Jacobi J_xi */Jacobi_xi = u1_xi_cross_u2_xi.norm();

#endif
	}

	

	vrFloat TriangleElemData_DisContinuous::rho_hat_SubTri(const vrInt subTriId, const vrFloat theta)const
	{
#if USE_Sigmoidal
		MyError("TriangleElemData_DisContinuous::rho_hat_SubTri.");
#endif
		
		return rho_hat_with_bar(Pedal_Length_in[subTriId],theta - theta_0_eta_SubTri[subTriId]);
	}
#if DEBUG_3_3
	vrFloat TriangleElemData_DisContinuous::rho_hat(const vrFloat theta)const
	{
		return rho_hat_with_bar(h_Pedal,theta - theta_0_eta);
	}

	void TriangleElemData_DisContinuous::compute_Shape_Deris_Jacobi_SST_3D_eta()
	{
#if SPEEDUP_5_31
		//1. compute source point local pos in eta space
		m_SrcPt_in_xi = MyVec2ParamSpace(0.0,0.0);
		//2. compute \lambda, cos(\gamma )
		const vrFloat lambda = u1_xi.norm() / u2_xi.norm();
		const vrFloat cos_gamma = compute_angle_ret_cos(u1_xi,u2_xi);//(u1_xi.dot(u2_xi)) / (u1_xi.norm() * u2_xi.norm());
		const vrFloat sin_gamma = std::sqrt( (1-cos_gamma*cos_gamma) );

		eta_2[0] = cos_gamma / lambda;	
		eta_2[1] = sin_gamma / lambda;
		Q_ASSERT(eta_2[1] > 0.0);

		mat_T.coeffRef(0,0) = 1.0;mat_T.coeffRef(0,1) = eta_2.x();
		mat_T.coeffRef(1,0) = 0.0;mat_T.coeffRef(1,1) = eta_2.y();

		mat_T_Inv.coeffRef(0,0) = 1.0;mat_T_Inv(0,1) = -1.0*eta_2.x()/eta_2.y();
		mat_T_Inv.coeffRef(1,0) = 0.0;mat_T_Inv.coeffRef(1,1) = 1.0/eta_2.y();

		det_Mat_T = mat_T.determinant();
		det_Mat_T_Inv = mat_T_Inv.determinant();

		m_SrcPt_in_eta = xi2eta(m_SrcPt_in_xi);
#if !DEBUG_5_28
		MyVec2ParamSpace xi_2 = eta2xi(eta_2);

		Q_ASSERT(numbers::isEqual(0.0,xi_2[0]) && numbers::isEqual(1.0,xi_2[1]));
#endif

		/* 2-1. Jacobi u1=@x/@xi_1 */u1_eta = u1_xi;
		/* 3-1. Jacobi u2=@x/@xi_2 */u2_eta = (u1_xi)*(-1.0*eta_2.x() / eta_2.y())+(u2_xi)*(1.0/eta_2.y());
		/* 4-1. Jacobi u1 * u2 */u1_eta_cross_u2_eta = u1_eta.cross(u2_eta);
		/* 4-1. Jacobi J_xi */Jacobi_eta = u1_eta_cross_u2_eta.norm();

#if DEBUG_SST
		//printf("u1_eta.norm(%f) u2_eta.norm(%f)\n",u1_eta.norm(),u2_eta.norm());
		Q_ASSERT(numbers::isZero(u1_eta.norm()-u2_eta.norm()));
		Q_ASSERT(numbers::isZero(u1_eta.dot(u2_eta)));
		MyMatrix_2X2 mustbeIdentity = mat_T * mat_T_Inv;
		//std::cout << mustbeIdentity << std::endl;
		Q_ASSERT(areEqualRel(mustbeIdentity.coeff(0,0),1.0,1e-6));
		Q_ASSERT(areEqualRel(mustbeIdentity.coeff(1,1),1.0,1e-6));
		//Q_ASSERT(areEqualRel(mustbeIdentity.coeff(0,1),0.0,1e-1));
		Q_ASSERT(areEqualRel(mustbeIdentity.coeff(1,0),0.0,1e-6));

		//printf("Jacobi_eta * det_Mat_T_Inv [%f] == [%f]\n",Jacobi_eta ,Jacobi_xi* det_Mat_T_Inv);
		Q_ASSERT(numbers::isEqual(Jacobi_eta ,Jacobi_xi* det_Mat_T_Inv));
		//printf("Jacobi_eta * det_Mat_T_Inv [%f] == [%f]\n",Jacobi_eta*det_Mat_T ,Jacobi_xi);
		Q_ASSERT(numbers::isEqual(Jacobi_eta*det_Mat_T ,Jacobi_xi));
#endif
		//3. compute  \theta 
		MyVec2ParamSpace vtx_eta[Geometry::vertexs_per_tri];
		vtx_eta[0][0] = 0.0; vtx_eta[0][1] = 0.0;
		vtx_eta[1][0] = 1.0; vtx_eta[1][1] = 0.0; MYNOTICE;
		vtx_eta[2][0] = eta_2.x(); vtx_eta[2][1] = eta_2.y();

		MyVec2ParamSpace vec_a = vtx_eta[2] - vtx_eta[0];
		MyVec2ParamSpace vec_b = vtx_eta[1] - vtx_eta[0];

		const vrFloat cos_theta = compute_angle_ret_cos(vec_a,vec_b);

		theta_eta = acos(cos_theta);

		//4. compute theta_bar
		MyVec2ParamSpace vec_aa = vtx_eta[0] - vtx_eta[2];
		MyVec2ParamSpace vec_bb = vtx_eta[1] - vtx_eta[2];
		vrFloat cos_aabb = compute_angle_ret_cos(vec_aa,vec_bb);//(vec_aa.dot(vec_bb))/(vec_aa.norm()*vec_bb.norm());

		MyVec2ParamSpace vec_bb_unit = vec_bb; vec_bb_unit.normalize();
		vrFloat cc = vec_aa.norm() * cos_aabb;

		Pedal_in_eta = vtx_eta[2] + cc * vec_bb_unit;

		h_Pedal = (Pedal_in_eta-vtx_eta[0]).norm();

		const vrFloat cos_theta_0 = h_Pedal / 1.0; 

		theta_0_eta = acos(cos_theta_0);//acos(x) return value in [0,pi]

#if DEBUG_SST
		//abs((vec_a.x()*vec_b.y())-(vec_a.y()*vec_b.x()));
		//(vec_a.cross(vec_b)).norm();
		//((Pedal_in_eta-vtx_eta[0]).norm() * vec_bb.norm());
		Q_ASSERT(numbers::isEqual(abs((vec_a.x()*vec_b.y())-(vec_a.y()*vec_b.x())),((Pedal_in_eta-vtx_eta[0]).norm() * vec_bb.norm())));
#endif
		int tmp_GaussPointSize_eta_In_Theta = 0;
		int tmp_GaussPointSize_eta_In_Rho = 0;
		if (dis_regular == m_DisContinuousType)
		{
			tmp_GaussPointSize_eta_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta;
			tmp_GaussPointSize_eta_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_eta_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_DisContinuous;
			tmp_GaussPointSize_eta_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_eta_In_Theta = tmp_GaussPointSize_eta_In_Theta;
		const vrInt nGaussPointSize_eta_In_Rho = tmp_GaussPointSize_eta_In_Rho;

		MyVector gaussPoint_xi_eta_In_Theta, gaussPoint_xi_eta_In_Theta_Weight;
		std::vector< MyVector > gaussPoint_xi_eta_In_Rho; gaussPoint_xi_eta_In_Rho.resize(nGaussPointSize_eta_In_Theta);
		std::vector< MyVector > gaussPoint_xi_eta_In_Rho_Weight; gaussPoint_xi_eta_In_Rho_Weight.resize(nGaussPointSize_eta_In_Theta);
		lgwt(nGaussPointSize_eta_In_Theta,0.0/*arc*/,theta_eta,gaussPoint_xi_eta_In_Theta,gaussPoint_xi_eta_In_Theta_Weight);
		for (int v=0;v<nGaussPointSize_eta_In_Theta;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_eta_In_Theta[v];
			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_eta_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_eta_In_Rho_Weight[v];
			const vrFloat cur_rho_hat_compare = rho_hat_with_bar(h_Pedal,cur_theta-theta_0_eta);
			const vrFloat cur_rho_hat = rho_hat(cur_theta);
			Q_ASSERT(numbers::isEqual(cur_rho_hat, cur_rho_hat_compare));
			lgwt(nGaussPointSize_eta_In_Rho,0.0,cur_rho_hat,cur_gaussPointIn_Rho,cur_gaussPointIn_Rho_Weight);
		}

		//single Layer 
		m_gaussQuadrature_eta_theta_singleLayer.resize(nGaussPointSize_eta_In_Theta,2);//(theta,weight)
		m_gaussQuadrature_xi_eta_polar.resize(nGaussPointSize_eta_In_Theta*nGaussPointSize_eta_In_Rho,3);
		m_gaussQuadrature_xi_eta_polar.setZero();
		for (int v=0,rowIdx=0;v<nGaussPointSize_eta_In_Theta;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_eta_In_Theta[v];
			const vrFloat cur_theta_weight = gaussPoint_xi_eta_In_Theta_Weight[v];

			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_eta_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_eta_In_Rho_Weight[v];

			for (int r=0;r<nGaussPointSize_eta_In_Rho;++r,++rowIdx)
			{
				m_gaussQuadrature_xi_eta_polar.coeffRef(rowIdx,TriangleElem::idx_theta_doubleLayer) = cur_theta;
				m_gaussQuadrature_xi_eta_polar.coeffRef(rowIdx,TriangleElem::idx_rho_doubleLayer) = cur_gaussPointIn_Rho[r];
				m_gaussQuadrature_xi_eta_polar.coeffRef(rowIdx,TriangleElem::idx_weight_doubleLayer) = cur_theta_weight * cur_gaussPointIn_Rho_Weight[r];
			}

			m_gaussQuadrature_eta_theta_singleLayer.coeffRef(v,TriangleElem::idx_theta_singleLayer) = cur_theta;
			m_gaussQuadrature_eta_theta_singleLayer.coeffRef(v,TriangleElem::idx_weight_singleLayer) = cur_theta_weight;

		}
#endif
	}
#endif

	DisContinuousType TriangleElemData_DisContinuous::computeTmpDisContinuousType(DisContinuousType srcType, const vrInt srcIdx)
	{
		switch (srcType)
		{
		case dis_1_1:
			{
				Q_ASSERT( (0 == srcIdx)||(2 == srcIdx) );
				if (0 == srcIdx)
				{
					return dis_1_1;
				}
				else
				{
					return dis_1_2;
				}
				break;
			}
		case dis_1_2:
			{
				Q_ASSERT( (0 == srcIdx)||(1 == srcIdx) );
				if (0 == srcIdx)
				{
					return dis_1_2;
				}
				else
				{
					return dis_1_1;
				}
				break;
			}
		case dis_1_3:
			{
				Q_ASSERT( (1 == srcIdx)||(2 == srcIdx) );
				if (1 == srcIdx)
				{
					return dis_1_2;
				}
				else
				{
					return dis_1_1;
				}
				break;
			}
		case dis_2_3:
			{
				Q_ASSERT( (0 == srcIdx) );
				return dis_2_3;
				break;
			}
		case dis_2_2:
			{
				Q_ASSERT( (2 == srcIdx) );
				return dis_2_3;
				break;
			}
		case dis_2_1:
			{
				Q_ASSERT( (1 == srcIdx) );
				return dis_2_3;
				break;
			}
		default:
			{
				MyError("error source point discontinuous type in TriangleElemData_DisContinuous::computeTmpDisContinuousType.");
				break;
			}
		}
	}

	DisContinuousType TriangleElemData_DisContinuous::computeTmpDisContinuousTypePlus(DisContinuousType srcType, const vrInt srcIdx)
	{
		Q_ASSERT( (srcIdx >= 0) && (srcIdx < 3) );
		switch (srcType)
		{
		case dis_1_1:
			{				
				if (0 == srcIdx)
				{
					return dis_1_1;
				}
				else if (2 == srcIdx)
				{
					return dis_1_2;
				}
				else
				{
					return dis_1_3;
				}
				break;
			}
		case dis_1_2:
			{
				if (0 == srcIdx)
				{
					return dis_1_2;
				}
				else if (1 == srcIdx)
				{
					return dis_1_1;
				}
				else
				{
					return dis_1_3;
				}
				break;
			}
		case dis_1_3:
			{
				if (1 == srcIdx)
				{
					return dis_1_2;
				}
				else if (2 == srcIdx)
				{
					return dis_1_1;
				}
				else
				{
					return dis_1_3;
				}
				break;
			}
		case dis_2_3:
			{
				
				if (0 == srcIdx)
				{
					return dis_2_3;
				}
				else if (1 == srcIdx)
				{
					return dis_2_2;
				}
				else
				{
					return dis_2_1;
				}
				break;
			}
		case dis_2_2:
			{
				if (2 == srcIdx)
				{
					return dis_2_3;
				}
				else if (0 == srcIdx)
				{
					return dis_2_2;
				}
				else
				{
					return dis_2_1;
				}
				break;
			}
		case dis_2_1:
			{
				if (1 == srcIdx)
				{
					return dis_2_3;
				}
				else if (0 == srcIdx)
				{
					return dis_2_1;
				}
				else
				{
					return dis_2_2;
				}
				break;
			}
		case dis_3_3:
			{
				return dis_3_3;
				break;
			}
		case dis_regular:
			{
				return dis_regular;
				break;
			}
		default:
			{
				MyError("error source point discontinuous type in TriangleElemData_DisContinuous::computeTmpDisContinuousType.");
				break;
			}
		}
	}

#if DEBUG_3_3

	MyVec2ParamSpace TriangleElemData_DisContinuous::eta2xi(const MyVec2ParamSpace& eta)const
	{
		return mat_T_Inv * eta;
	}

	/*MyVec3 TriangleElemData_DisContinuous::xi2global(const MyVec2ParamSpace& xi)const
	{
		return shapefunction_xi( 0, xi)*m_vtx_globalCoord[0]
		+ shapefunction_xi( 1, xi)*m_vtx_globalCoord[1]
		+ shapefunction_xi( 2, xi)*m_vtx_globalCoord[2];
	}*/
#endif	

	MyVec2ParamSpace TriangleElemData_DisContinuous::xi2eta_SubTri(const MyVec2ParamSpace& xi)const
	{
		return mat_T_SubTri * xi;
	}

	MyVec2ParamSpace TriangleElemData_DisContinuous::eta2xi_SubTri(const MyVec2ParamSpace& eta)const
	{
		return mat_T_Inv_SubTri * eta;
	}

	MyVec3 TriangleElemData_DisContinuous::xi2global(const MyVec2ParamSpace& xi)const
	{
		/*printf("shapefunction_xi( 0, s_paramCoordsInDiscontinuous[m_DisContinuousType][0]) = [%f] m_DisContinuousType[%d] src[%f, %f]\n",
			shapefunction_xi( 0, s_paramCoordsInDiscontinuous[m_DisContinuousType][0]), m_DisContinuousType,
			s_paramCoordsInDiscontinuous[m_DisContinuousType][0][0], s_paramCoordsInDiscontinuous[m_DisContinuousType][0][1]);*/
		/*Q_ASSERT( numbers::isEqual(1.0, shapefunction_xi( 0, s_paramCoordsInDiscontinuous[m_DisContinuousType][0])) );
		Q_ASSERT( numbers::isEqual(0.0, shapefunction_xi( 0, s_paramCoordsInDiscontinuous[m_DisContinuousType][1])) );
		Q_ASSERT( numbers::isEqual(0.0, shapefunction_xi( 0, s_paramCoordsInDiscontinuous[m_DisContinuousType][2])) );

		Q_ASSERT( numbers::isEqual(1.0, shapefunction_xi( 1, s_paramCoordsInDiscontinuous[m_DisContinuousType][1])) );
		Q_ASSERT( numbers::isEqual(0.0, shapefunction_xi( 1, s_paramCoordsInDiscontinuous[m_DisContinuousType][2])) );
		Q_ASSERT( numbers::isEqual(0.0, shapefunction_xi( 1, s_paramCoordsInDiscontinuous[m_DisContinuousType][0])) );

		Q_ASSERT( numbers::isEqual(1.0, shapefunction_xi( 2, s_paramCoordsInDiscontinuous[m_DisContinuousType][2])) );
		Q_ASSERT( numbers::isEqual(0.0, shapefunction_xi( 2, s_paramCoordsInDiscontinuous[m_DisContinuousType][0])) );
		Q_ASSERT( numbers::isEqual(0.0, shapefunction_xi( 2, s_paramCoordsInDiscontinuous[m_DisContinuousType][1])) );*/

		return shapefunction_xi( 0, xi)*m_vtx_globalCoord[0]
		+ shapefunction_xi( 1, xi)*m_vtx_globalCoord[1]
		+ shapefunction_xi( 2, xi)*m_vtx_globalCoord[2];
	}
#if DEBUG_3_3
	vrFloat TriangleElemData_DisContinuous::A_i_theta(const vrInt idx_i, const vrFloat theta)const
	{
		return u1_eta[idx_i]*cos(theta) + u2_eta[idx_i]*sin(theta);
	}

	vrFloat TriangleElemData_DisContinuous::B_i_theta(const vrInt idx_i, const vrFloat theta)const
	{
		return 0.0;
	}
	

	vrFloat TriangleElemData_DisContinuous::A_theta(const vrFloat theta)const
	{
		MyVec3 A_i;
		for (int idx_i=0;idx_i<MyDim;++idx_i)
		{
			A_i[idx_i] = A_i_theta(idx_i,theta);
		}

		vrFloat retValue = A_i.norm();
		Q_ASSERT(numbers::isEqual(retValue, u2_eta.norm()));
		return retValue;
	}

	vrFloat TriangleElemData_DisContinuous::B_theta(const vrFloat theta)const
	{
		return 0.0;
	}

	vrFloat TriangleElemData_DisContinuous::r_i(const vrInt idx_i, const vrFloat theta)const
	{
		return A_i_theta(idx_i,theta) / A_theta(theta);
	}

	vrFloat TriangleElemData_DisContinuous::N_I_0_eta(const vrInt idx_I/*, const MyVec2ParamSpace& eta*/)const
	{
		//const MyVec2ParamSpace xi = eta2xi(eta);
		const vrFloat retVal = shapefunction_xi(idx_I,m_SrcPt_in_xi);
		Q_ASSERT(numbers::isEqual(retVal,0.0) || numbers::isEqual(retVal,1.0) );
		return retVal;
	}

	vrFloat TriangleElemData_DisContinuous::N_I_1_eta(const vrInt idx_I, const vrFloat theta)const
	{
		MyError("TriangleElemData_DisContinuous::N_I_1_eta. use xi_0_eta_0");
		return 0.0;

#if DEBUG_3_3

		/*const MyVec2ParamSpace& xi = m_SrcPt_in_xi;
		return (
			(derisShapefunction_xi( idx_I, 0, xi) * xi_0_eta_0() + derisShapefunction_xi( idx_I, 1, xi) * xi_1_eta_0()) * std::cos(theta) + 
			(derisShapefunction_xi( idx_I, 0, xi) * xi_0_eta_1() + derisShapefunction_xi( idx_I, 1, xi) * xi_1_eta_1()) * std::sin(theta) );*/
#endif
	}

	MyVec2ParamSpace TriangleElemData_DisContinuous::xi2eta(const MyVec2ParamSpace& xi)const
	{
		return mat_T * xi;
	}
#endif

	vrFloat TriangleElemData_DisContinuous::A_i_theta_SubTri(const vrInt idx_i, const vrFloat theta)const
	{
		return u1_eta_SubTri[idx_i]*cos(theta) + u2_eta_SubTri[idx_i]*sin(theta);
	}

	vrFloat TriangleElemData_DisContinuous::B_i_theta_SubTri(const vrInt idx_i, const vrFloat theta)const
	{
		return 0.0;
	}

	vrFloat TriangleElemData_DisContinuous::A_theta_SubTri(const vrFloat theta)const
	{
		MyVec3 A_i;
		for (int idx_i=0;idx_i<MyDim;++idx_i)
		{
			A_i[idx_i] = A_i_theta_SubTri(idx_i,theta);
		}

		vrFloat retValue = A_i.norm();
		Q_ASSERT(numbers::isEqual(retValue, u2_eta_SubTri.norm()));
		return retValue;
	}

	vrFloat TriangleElemData_DisContinuous::B_theta_SubTri(const vrFloat theta)const
	{
		return 0.0;
	}

	vrFloat TriangleElemData_DisContinuous::r_i_SubTri(const vrInt idx_i, const vrFloat theta)const
	{
		return A_i_theta_SubTri(idx_i,theta) / A_theta_SubTri(theta);
	}

	vrFloat TriangleElemData_DisContinuous::N_I_0_eta_SubTri(const vrInt idx_I/*, const MyVec2ParamSpace& eta*/)const
	{
		//const MyVec2ParamSpace xi = eta2xi(eta);
		const vrFloat retVal = shapefunction_xi(idx_I,m_SrcPt_in_xi_SubTri);
		Q_ASSERT(numbers::isEqual(retVal,0.0) || numbers::isEqual(retVal,1.0) );
		return retVal;
	}

	vrFloat TriangleElemData_DisContinuous::N_I_1_eta_SubTri(const vrInt idx_I, const vrFloat theta)const
	{
		const MyVec2ParamSpace& xi = m_SrcPt_in_xi_SubTri;
		return (
			(derisShapefunction_xi( idx_I, 0, xi) * xi_0_eta_0_SubTri() + derisShapefunction_xi( idx_I, 1, xi) * xi_1_eta_0_SubTri()) * std::cos(theta) + 
			(derisShapefunction_xi( idx_I, 0, xi) * xi_0_eta_1_SubTri() + derisShapefunction_xi( idx_I, 1, xi) * xi_1_eta_1_SubTri()) * std::sin(theta) );
		/*MyError("TriangleElemData_DisContinuous::N_I_1_eta_SubTri.");
		return 0.0;*/
	}

	vrFloat Heron_formula(const vrFloat a, const vrFloat b, const vrFloat c)
	{
		Q_ASSERT(a > 0 && b > 0 && c > 0);
		const vrFloat p = (a + b + c) / 2.0;
		return std::sqrt( p * (p-a) * (p-b) * (p-c) );
	}

	MyVec2ParamSpace Frenet_Serret_formulas_2d_UnitNormal(const MyVec2ParamSpace& vec)
	{
		MyVec2ParamSpace retVec(vec[1], -1.0*vec[0]);
		retVec.normalize();
		return retVec;
	}

	bool point_in_line(const MyVec2ParamSpace& end0, const MyVec2ParamSpace& end1, const MyVec2ParamSpace& pt)
	{
		return numbers::isEqual( (pt - end0).norm() + (end1 - pt).norm(), (end1 - end0).norm());
	}

	const vrInt TriangleElemData_DisContinuous::search_Theta_in_eta_belong_SubTri_Index(const vrFloat curTheta)const
	{
		const vrFloat curTheta_less_zero = curTheta - 2.0 * numbers::MyPI;
		const vrFloat curTheta_large_2pi = curTheta + 2.0 * numbers::MyPI;

		for (int idx=0; idx < SubTriSize; ++idx)
		{
			if ( ((subtri_Theta_eta[idx][0] <= curTheta)&&(curTheta <= subtri_Theta_eta[idx][1])) ||
				((subtri_Theta_eta[idx][0] <= curTheta_less_zero)&&(curTheta_less_zero <= subtri_Theta_eta[idx][1])) ||
				((subtri_Theta_eta[idx][0] <= curTheta_large_2pi)&&(curTheta_large_2pi <= subtri_Theta_eta[idx][1])) )
			{
				return idx;
			}
		}
		MyError("error in TriangleElemData_DisContinuous::search_Theta_in_eta_belong_SubTri_Index.");
		return 0;
	}

	const vrInt TriangleElemData_DisContinuous::search_Theta_in_eta_belong_SubTri_Index(vrFloat subtri_Theta_eta[][2], const vrFloat curTheta)
	{
		const vrFloat curTheta_less_zero = curTheta - 2.0 * numbers::MyPI;
		const vrFloat curTheta_large_2pi = curTheta + 2.0 * numbers::MyPI;

		for (int idx=0; idx < SubTriSize; ++idx)
		{
			if ( ((subtri_Theta_eta[idx][0] <= curTheta)&&(curTheta <= subtri_Theta_eta[idx][1])) ||
				 ((subtri_Theta_eta[idx][0] <= curTheta_less_zero)&&(curTheta_less_zero <= subtri_Theta_eta[idx][1])) ||
				 ((subtri_Theta_eta[idx][0] <= curTheta_large_2pi)&&(curTheta_large_2pi <= subtri_Theta_eta[idx][1])) )
			{
				return idx;
			}
		}
		MyError("error in TriangleElemData_DisContinuous::search_Theta_in_eta_belong_SubTri_Index.");
		return 0;
	}

	const vrFloat radian2angle(const vrFloat radian)
	{
		return (180.0 / numbers::MyPI) * radian;
	}

	const vrFloat angle2radian(const vrFloat angle)
	{
		return (numbers::MyPI / 180.0) * angle;
	}

	void TriangleElemData_DisContinuous::compute_Shape_Deris_Jacobi_SST_3D_eta_DisContinuousPt()
	{
		//1. compute 
		m_SrcPt_in_xi_SubTri = MyVec2ParamSpace(s2, s2);
		//2. compute \lambda, cos(\gamma )
		const vrFloat lambda = u1_xi.norm() / u2_xi.norm();
		const vrFloat cos_gamma = compute_angle_ret_cos(u1_xi,u2_xi);//(u1_xi.dot(u2_xi)) / (u1_xi.norm() * u2_xi.norm());
		const vrFloat sin_gamma = std::sqrt( (1-cos_gamma*cos_gamma) );

		eta_2_SubTri[0] = cos_gamma / lambda;	
		eta_2_SubTri[1] = sin_gamma / lambda;
		Q_ASSERT(eta_2_SubTri[1] > 0.0);

		mat_T_SubTri.coeffRef(0,0) = 1.0; mat_T_SubTri.coeffRef(0,1) = eta_2_SubTri.x();
		mat_T_SubTri.coeffRef(1,0) = 0.0;mat_T_SubTri.coeffRef(1,1) = eta_2_SubTri.y();

		mat_T_Inv_SubTri.coeffRef(0,0) = 1.0;mat_T_Inv_SubTri.coeffRef(0,1) = -1.0*eta_2_SubTri.x()/eta_2_SubTri.y();
		mat_T_Inv_SubTri.coeffRef(1,0) = 0.0;mat_T_Inv_SubTri.coeffRef(1,1) = 1.0/eta_2_SubTri.y();

		det_Mat_T_SubTri = mat_T_SubTri.determinant();
		det_Mat_T_Inv_SubTri = mat_T_Inv_SubTri.determinant();

		m_SrcPt_in_eta_SubTri = xi2eta_SubTri(m_SrcPt_in_xi_SubTri);
		/* 2-1. Jacobi u1=@x/@xi_1 */u1_eta_SubTri = u1_xi;
		/* 3-1. Jacobi u2=@x/@xi_2 */u2_eta_SubTri = (u1_xi)*(-1.0*eta_2_SubTri.x() / eta_2_SubTri.y())+(u2_xi)*(1.0/eta_2_SubTri.y());
		/* 4-1. Jacobi u1 * u2 */u1_eta_cross_u2_eta_SubTri = u1_eta_SubTri.cross(u2_eta_SubTri);
#if USE_Jacobi_Weight
		vrFloat total_Jacobi_eta_SubTri = u1_eta_cross_u2_eta_SubTri.norm();
		
#else//USE_Jacobi_Weight
		/* 4-1. Jacobi J_xi */Jacobi_eta_SubTri = u1_eta_cross_u2_eta_SubTri.norm();
#endif//USE_Jacobi_Weight
		

#if DEBUG_SST
		//printf("u1_eta.norm(%f) u2_eta.norm(%f)\n",u1_eta.norm(),u2_eta.norm());
		Q_ASSERT(numbers::isZero(u1_eta_SubTri.norm()-u2_eta_SubTri.norm()));
		Q_ASSERT(numbers::isZero(u1_eta_SubTri.dot(u2_eta_SubTri)));
		MyMatrix_2X2 mustbeIdentity = mat_T_SubTri * mat_T_Inv_SubTri;
		//std::cout << mustbeIdentity << std::endl;
		Q_ASSERT(areEqualRel(mustbeIdentity.coeff(0,0),1.0,1e-6));
		Q_ASSERT(areEqualRel(mustbeIdentity.coeff(1,1),1.0,1e-6));
		//Q_ASSERT(areEqualRel(mustbeIdentity.coeff(0,1),0.0,1e-1));
		Q_ASSERT(areEqualRel(mustbeIdentity.coeff(1,0),0.0,1e-6));

		//printf("Jacobi_eta * det_Mat_T_Inv [%f] == [%f]\n",Jacobi_eta ,Jacobi_xi* det_Mat_T_Inv);
		Q_ASSERT(numbers::isEqual(Jacobi_eta_SubTri ,Jacobi_xi* det_Mat_T_Inv_SubTri));
		//printf("Jacobi_eta * det_Mat_T_Inv [%f] == [%f]\n",Jacobi_eta*det_Mat_T ,Jacobi_xi);
		Q_ASSERT(numbers::isEqual(Jacobi_eta_SubTri*det_Mat_T_SubTri ,Jacobi_xi));
#endif
		//3. compute  \theta 
		MyVec2ParamSpace vtx_eta[Geometry::vertexs_per_tri];
		vtx_eta[0][0] = 0.0; vtx_eta[0][1] = 0.0;
		vtx_eta[1][0] = 1.0; vtx_eta[1][1] = 0.0; MYNOTICE;
		vtx_eta[2][0] = eta_2_SubTri.x(); vtx_eta[2][1] = eta_2_SubTri.y();

		//printf("DEBUG SST : ETA (%f,%f), (%f,%f), (%f,%f)\n", vtx_eta[0][0],vtx_eta[0][1], vtx_eta[1][0],vtx_eta[1][1], vtx_eta[2][0],vtx_eta[2][1]);
		
		
		//printf("DEBUG SST : src image in xi(%f,%f) eta(%f,%f)\n",srcPt_in_xi[0],srcPt_in_xi[1],srcPt_in_eta[0],srcPt_in_eta[1]);
		
		MyVec2ParamSpace eta_vtx_edge[Geometry::vertexs_per_tri];
		eta_vtx_edge[0] = vtx_eta[0] - m_SrcPt_in_eta_SubTri;
		eta_vtx_edge[1] = vtx_eta[1] - m_SrcPt_in_eta_SubTri;
		eta_vtx_edge[2] = vtx_eta[2] - m_SrcPt_in_eta_SubTri;
		const MyVec2ParamSpace srcPt_in_eta_horizontal_line(1,0);

		vrFloat included_angle[Geometry::vertexs_per_tri];
		included_angle[0] = acos(compute_angle_ret_cos(eta_vtx_edge[0], srcPt_in_eta_horizontal_line));
		included_angle[1] = acos(compute_angle_ret_cos(eta_vtx_edge[1], srcPt_in_eta_horizontal_line));
		included_angle[2] = acos(compute_angle_ret_cos(eta_vtx_edge[2], srcPt_in_eta_horizontal_line));

		/*printf("DEBUG SST : included_angle (%f,%f,%f)  angle (%f,%f,%f)\n",
			included_angle[0], included_angle[1], included_angle[2],
			radian2angle(included_angle[0]),radian2angle(included_angle[1]),radian2angle(included_angle[2]));*/

		//Q_ASSERT(included_angle[0]+included_angle[1]+included_angle[2]);
#if USE_Sigmoidal
		subtri_Theta_eta[0][0] = 0.0 - included_angle[0];
		subtri_Theta_eta[0][1] = 0.0 - included_angle[1];
		Q_ASSERT(subtri_Theta_eta[0][0] < subtri_Theta_eta[0][1]);
		//printf("DEBUG SST : SUB TRI %d (%f)-(%f)\n",0, radian2angle(subtri_Theta_eta[0][0]), radian2angle(subtri_Theta_eta[0][1]));
#else
		subtri_Theta_eta[0][0] = numbers::MyPI * 2.0 - included_angle[0];
		subtri_Theta_eta[0][1] = numbers::MyPI * 2.0 - included_angle[1];
		Q_ASSERT(subtri_Theta_eta[0][0] < subtri_Theta_eta[0][1]);
		//printf("DEBUG SST : SUB TRI %d (%f)-(%f)\n",0, radian2angle(subtri_Theta_eta[0][0]), radian2angle(subtri_Theta_eta[0][1]));
#endif
		

		subtri_Theta_eta[1][0] = /*numbers::MyPI * 2.0*/ -1.0 * included_angle[1]; MYNOTICE;
		subtri_Theta_eta[1][1] = included_angle[2];
		Q_ASSERT(subtri_Theta_eta[1][0] < subtri_Theta_eta[1][1]);
		//printf("DEBUG SST : SUB TRI %d (%f)-(%f)\n",1, radian2angle(subtri_Theta_eta[1][0]), radian2angle(subtri_Theta_eta[1][1]));

		subtri_Theta_eta[2][0] = included_angle[2];
		subtri_Theta_eta[2][1] = numbers::MyPI * 2.0 - included_angle[0];
		Q_ASSERT(subtri_Theta_eta[2][0] < subtri_Theta_eta[2][1]);
		//printf("DEBUG SST : SUB TRI %d (%f)-(%f)\n",2, radian2angle(subtri_Theta_eta[2][0]), radian2angle(subtri_Theta_eta[2][1]));

#if DEBUG_SST
		vrFloat angleCycle = (subtri_Theta_eta[0][1] - subtri_Theta_eta[0][0]) + (subtri_Theta_eta[1][1] - subtri_Theta_eta[1][0]) + (subtri_Theta_eta[2][1] - subtri_Theta_eta[2][0]); 
		Q_ASSERT(numbers::isEqual(angleCycle, numbers::MyPI * 2.0));
#endif

		const MyVec2ParamSpace vec_0_1 = vtx_eta[1] - vtx_eta[0];
		const MyVec2ParamSpace vec_1_2 = vtx_eta[2] - vtx_eta[1];
		const MyVec2ParamSpace vec_2_0 = vtx_eta[0] - vtx_eta[2];

		const MyVec2ParamSpace normal_0_1_Unit = Frenet_Serret_formulas_2d_UnitNormal(vec_0_1);
		const MyVec2ParamSpace normal_1_2_Unit = Frenet_Serret_formulas_2d_UnitNormal(vec_1_2);
		const MyVec2ParamSpace normal_2_0_Unit = Frenet_Serret_formulas_2d_UnitNormal(vec_2_0);

		const vrFloat Area = Heron_formula(vec_0_1.norm(), vec_1_2.norm(), vec_2_0.norm()); 
		vrFloat subTriArea[SubTriSize];
		//sub tri 0
		{
			subTriArea[0] = Heron_formula(eta_vtx_edge[0].norm(), vec_0_1.norm(), eta_vtx_edge[1].norm()); 
			const vrFloat area_0 = subTriArea[0];
			Pedal_Length_in[0] = 2.0* area_0 / vec_0_1.norm();
			Pedal_Vector_In_eta[0] = Pedal_Length_in[0] * normal_0_1_Unit;
			Pedal_in_edge_eta[0] = m_SrcPt_in_eta_SubTri + Pedal_Vector_In_eta[0];
#if USE_Sigmoidal
			theta_0_eta_SubTri[0] = 0.0 MYNOTICE - acos(compute_angle_ret_cos(normal_0_1_Unit, srcPt_in_eta_horizontal_line));
#else
			theta_0_eta_SubTri[0] = numbers::MyPI * 2.0 MYNOTICE - acos(compute_angle_ret_cos(normal_0_1_Unit, srcPt_in_eta_horizontal_line));
#endif
			
			Q_ASSERT(point_in_line(vtx_eta[0],vtx_eta[1],Pedal_in_edge_eta[0]));
			Q_ASSERT(numbers::isEqual(theta_0_eta_SubTri[0], (-1.0 * numbers::PI_2)));

			/*printf("DEBUG SST : Pedal_in %d (%f,%f) theta-0 [%f]\n",
				0, Pedal_in_edge_eta[0][0], Pedal_in_edge_eta[0][1], radian2angle(theta_0_eta_SubTri[0]));*/
		}

		//sub tri 1
		{
			subTriArea[1] = Heron_formula(eta_vtx_edge[1].norm(), vec_1_2.norm(), eta_vtx_edge[2].norm()); 
			const vrFloat area_1 = subTriArea[1];
			Pedal_Length_in[1] = 2.0 * area_1 / vec_1_2.norm();
			Pedal_Vector_In_eta[1] = Pedal_Length_in[1] * normal_1_2_Unit;
			Pedal_in_edge_eta[1] = m_SrcPt_in_eta_SubTri + Pedal_Vector_In_eta[1];
			theta_0_eta_SubTri[1] = acos(compute_angle_ret_cos(normal_1_2_Unit, srcPt_in_eta_horizontal_line));
			Q_ASSERT(point_in_line(vtx_eta[1],vtx_eta[2],Pedal_in_edge_eta[1]));

			/*printf("DEBUG SST : Pedal_in %d (%f,%f) theta-0 [%f]\n",
				1, Pedal_in_edge_eta[1][0], Pedal_in_edge_eta[1][1], radian2angle(theta_0_eta_SubTri[1]));*/
		}

		//sub tri 2
		{
			subTriArea[2] = Heron_formula(eta_vtx_edge[2].norm(), vec_2_0.norm(), eta_vtx_edge[0].norm()); 
			const vrFloat area_2  = subTriArea[2];
			Pedal_Length_in[2] = 2.0 * area_2 / vec_2_0.norm();
			Pedal_Vector_In_eta[2] = Pedal_Length_in[2] * normal_2_0_Unit;
			Pedal_in_edge_eta[2] = m_SrcPt_in_eta_SubTri + Pedal_Vector_In_eta[2];
			theta_0_eta_SubTri[2] = acos(compute_angle_ret_cos(normal_2_0_Unit, srcPt_in_eta_horizontal_line));
			Q_ASSERT(point_in_line(vtx_eta[2],vtx_eta[0],Pedal_in_edge_eta[2]));

			/*printf("DEBUG SST : Pedal_in %d (%f,%f) theta-0 [%f]\n",
				2, Pedal_in_edge_eta[2][0], Pedal_in_edge_eta[2][1], radian2angle(theta_0_eta_SubTri[2]));*/
		}
		
#if USE_Jacobi_Weight
		for (int subTriIdx=0; subTriIdx < SubTriSize; ++subTriIdx)
		{
			Jacobi_eta_SubTri[subTriIdx] = total_Jacobi_eta_SubTri * (subTriArea[subTriIdx] / Area);
		}
#endif//USE_Jacobi_Weight
		//gauss point
#if USE_360_Sample
		MyMatrix& gaussQuadrature_xi_eta_polar_360 = m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal;
		MyMatrix& gaussQuadrature_eta_theta_singleLayer_360 = m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal;

		MyVector gaussPoint_xi_eta_In_Theta, gaussPoint_xi_eta_In_Theta_Weight, gaussPoint_xi_eta_In_Rho_Hat_End;
		std::vector< MyVector > gaussPoint_xi_eta_In_Rho(GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri);
		std::vector< MyVector > gaussPoint_xi_eta_In_Rho_Weight(GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri);

		lgwt(GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri,
			0.0 , 2.0*numbers::MyPI,MyNoticeMsg("360 sample")
			gaussPoint_xi_eta_In_Theta,gaussPoint_xi_eta_In_Theta_Weight);

		gaussPoint_xi_eta_In_Rho_Hat_End.resize(GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri);
		gaussPoint_xi_eta_In_Rho_Hat_End.setZero();

		for (int v=0;v<GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_eta_In_Theta[v];
			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_eta_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_eta_In_Rho_Weight[v];
			

			const vrInt triId = search_Theta_in_eta_belong_SubTri_Index(subtri_Theta_eta, cur_theta);
			const vrFloat cur_rho_hat_compare = rho_hat_with_bar(Pedal_Length_in[triId], MYNOTICE
				cur_theta-theta_0_eta_SubTri[triId]);

			gaussPoint_xi_eta_In_Rho_Hat_End[v] = cur_rho_hat_compare;

			lgwt(GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri,0.0 MYNOTICE,cur_rho_hat_compare ,cur_gaussPointIn_Rho,cur_gaussPointIn_Rho_Weight);
		}//for (int v=0;v<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++v)

		gaussQuadrature_eta_theta_singleLayer_360.resize(GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri, TriangleElem::singleLayerSize);
		gaussQuadrature_xi_eta_polar_360.resize(GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri*GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri,TriangleElem::doubleLayerSize);
		gaussQuadrature_xi_eta_polar_360.setZero();

		for (int v=0,rowIdx=0;v<GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;++v)
		{
			const vrFloat cur_theta = gaussPoint_xi_eta_In_Theta[v];
			const vrFloat cur_theta_weight = gaussPoint_xi_eta_In_Theta_Weight[v];
			const vrFloat cur_theta_rhoHat = gaussPoint_xi_eta_In_Rho_Hat_End[v];

			MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_eta_In_Rho[v];
			MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_eta_In_Rho_Weight[v];

			for (int r=0;r<GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;++r,++rowIdx)
			{
				gaussQuadrature_xi_eta_polar_360.coeffRef(rowIdx, TriangleElem::idx_theta_doubleLayer) = cur_theta;
				gaussQuadrature_xi_eta_polar_360.coeffRef(rowIdx, TriangleElem::idx_rho_doubleLayer) = cur_gaussPointIn_Rho[r];
				gaussQuadrature_xi_eta_polar_360.coeffRef(rowIdx, TriangleElem::idx_weight_doubleLayer) = cur_theta_weight * cur_gaussPointIn_Rho_Weight[r];
			}//for (int r=0;r<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++r,++rowIdx)

			gaussQuadrature_eta_theta_singleLayer_360.coeffRef(v, TriangleElem::idx_theta_singleLayer) = cur_theta;
			gaussQuadrature_eta_theta_singleLayer_360.coeffRef(v, TriangleElem::idx_weight_singleLayer) = cur_theta_weight;

			gaussQuadrature_eta_theta_singleLayer_360.coeffRef(v,TriangleElem::idx_rho_bar_singleLayer) = cur_theta_rhoHat;
		}//for (int v=0,rowIdx=0;v<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++v)

#else//USE_360_Sample
#if USE_Aliabadi_RegularSample
		for (int triId=0, nGaussIdx = 0, global_row_idx=0; triId < SubTriSize; ++triId)
		{
			const vrFloat currentTriAreaWeight = 1.0;//subTriArea[triId] / Area;

			MyMatrix& gaussQuadrature_xi_eta_polar = m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[triId];
			MyMatrix& gaussQuadrature_eta_theta_singleLayer = m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[triId];

			MyVector gaussPoint_xi_eta_In_Theta, gaussPoint_xi_eta_In_Theta_Weight, gaussPoint_xi_eta_In_Theta_rho_hat;
			MyVector gaussPoint_xi_eta_In_Rho[TriangleElem::GaussPointSize_eta_In_Theta_SubTri],gaussPoint_xi_eta_In_Rho_Weight[TriangleElem::GaussPointSize_eta_In_Theta_SubTri];
			lgwt(TriangleElem::GaussPointSize_eta_In_Theta_SubTri,
				subtri_Theta_eta[triId][0], subtri_Theta_eta[triId][1],MYNOTICE
				gaussPoint_xi_eta_In_Theta,gaussPoint_xi_eta_In_Theta_Weight);

			gaussPoint_xi_eta_In_Theta_rho_hat.resize(TriangleElem::GaussPointSize_eta_In_Theta_SubTri);
			for (int v=0;v<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++v)
			{
				const vrFloat cur_theta = gaussPoint_xi_eta_In_Theta[v];
				MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_eta_In_Rho[v];
				MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_eta_In_Rho_Weight[v];

				const vrFloat cur_rho_hat_compare = rho_hat_with_bar(Pedal_Length_in[triId], MYNOTICE
					cur_theta-theta_0_eta_SubTri[triId]);

				gaussPoint_xi_eta_In_Theta_rho_hat[v] = cur_rho_hat_compare;

				lgwt(TriangleElem::GaussPointSize_eta_In_Rho_SubTri,0.0,cur_rho_hat_compare ,cur_gaussPointIn_Rho,cur_gaussPointIn_Rho_Weight);
			}

			gaussQuadrature_eta_theta_singleLayer.resize(TriangleElem::GaussPointSize_eta_In_Theta_SubTri, TriangleElem::singleLayerSize);
			gaussQuadrature_xi_eta_polar.resize(TriangleElem::GaussPointSize_eta_In_Theta_SubTri*TriangleElem::GaussPointSize_eta_In_Rho_SubTri,TriangleElem::doubleLayerSize);
			gaussQuadrature_xi_eta_polar.setZero();

			for (int v=0,rowIdx=0;v<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++v)
			{
				const vrFloat cur_theta = gaussPoint_xi_eta_In_Theta[v];
				const vrFloat cur_theta_weight = gaussPoint_xi_eta_In_Theta_Weight[v] * currentTriAreaWeight;


				const vrFloat cur_theta_rho_hat = gaussPoint_xi_eta_In_Theta_rho_hat[v];

				MyVector& cur_gaussPointIn_Rho = gaussPoint_xi_eta_In_Rho[v];
				MyVector& cur_gaussPointIn_Rho_Weight = gaussPoint_xi_eta_In_Rho_Weight[v];

				for (int r=0;r<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++r,++rowIdx)
				{
					gaussQuadrature_xi_eta_polar.coeffRef(rowIdx,TriangleElem::idx_theta_doubleLayer) = cur_theta;
					gaussQuadrature_xi_eta_polar.coeffRef(rowIdx,TriangleElem::idx_rho_doubleLayer) = cur_gaussPointIn_Rho[r];
					//printf("[%d]{%f,%f},\n",triId,cur_theta,cur_gaussPointIn_Rho[r]);
					gaussQuadrature_xi_eta_polar.coeffRef(rowIdx,TriangleElem::idx_weight_doubleLayer) = cur_theta_weight * cur_gaussPointIn_Rho_Weight[r] ;
				}

				gaussQuadrature_eta_theta_singleLayer.coeffRef(v,TriangleElem::idx_theta_singleLayer) = cur_theta;
				gaussQuadrature_eta_theta_singleLayer.coeffRef(v,TriangleElem::idx_weight_singleLayer) = cur_theta_weight;
				gaussQuadrature_eta_theta_singleLayer.coeffRef(v,TriangleElem::idx_rho_bar_singleLayer) = cur_theta_rho_hat;

			}

		}
#endif//USE_Aliabadi_RegularSample
#endif//USE_360_Sample
	}
#endif
}//namespace VR