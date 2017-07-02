#include "quad_rule.h"

namespace VR
{
	/*
	gaussPoint - a (n by 3) matrix:
	n - the order of the Gaussian quadrature (n<=12)
	1st column gives the x-coordinates of points
	2nd column gives the y-coordinates of points
	3rd column gives the weights
	*/
	MyFloat vrGaussPoint::gaussPoint_order_1[1][3]={{0.33333333333333, 0.33333333333333, 1.00000000000000}};
	MyFloat vrGaussPoint::gaussPoint_order_2[3][3]={{0.16666666666667,0.16666666666667,0.33333333333333},
									  {0.16666666666667,0.66666666666667,0.33333333333333},
									  {0.66666666666667,0.16666666666667,0.33333333333333}};
	MyFloat vrGaussPoint::gaussPoint_order_3[4][3]={	{0.33333333333333,0.33333333333333,-0.56250000000000},
										{0.20000000000000,0.20000000000000,0.52083333333333},
										{0.20000000000000,0.60000000000000,0.52083333333333},
										{0.60000000000000,0.20000000000000,0.52083333333333}};
	MyFloat vrGaussPoint::gaussPoint_order_4[6][3]={	{0.44594849091597,0.44594849091597,0.22338158967801},
										{0.44594849091597,0.10810301816807,0.22338158967801},
										{0.10810301816807,0.44594849091597,0.22338158967801},
										{0.09157621350977,0.09157621350977,0.10995174365532},
										{0.09157621350977,0.81684757298046,0.10995174365532},
										{0.81684757298046,0.09157621350977,0.10995174365532}};
	MyFloat vrGaussPoint::gaussPoint_order_5[7][3]={	{0.33333333333333,0.33333333333333,0.22500000000000},
										{0.47014206410511,0.47014206410511,0.13239415278851},
										{0.47014206410511,0.05971587178977,0.13239415278851},
										{0.05971587178977,0.47014206410511,0.13239415278851},
										{0.10128650732346,0.10128650732346,0.12593918054483},
										{0.10128650732346,0.79742698535309,0.12593918054483},
										{0.79742698535309,0.10128650732346,0.12593918054483}};
	MyFloat vrGaussPoint::gaussPoint_order_6[12][3]={	{0.24928674517091,0.24928674517091,0.11678627572638},
										{0.24928674517091,0.50142650965818,0.11678627572638},
										{0.50142650965818,0.24928674517091,0.11678627572638},
										{0.06308901449150,0.06308901449150,0.05084490637021},
										{0.06308901449150,0.87382197101700,0.05084490637021},
										{0.87382197101700,0.06308901449150,0.05084490637021},
										{0.31035245103378,0.63650249912140,0.08285107561837},
										{0.63650249912140,0.05314504984482,0.08285107561837},
										{0.05314504984482,0.31035245103378,0.08285107561837},
										{0.63650249912140,0.31035245103378,0.08285107561837},
										{0.31035245103378,0.05314504984482,0.08285107561837},
										{0.05314504984482,0.63650249912140,0.08285107561837}};
	MyFloat vrGaussPoint::gaussPoint_order_7[13][3]={	{0.33333333333333,0.33333333333333,-0.14957004446768},
										{0.26034596607904,0.26034596607904,0.17561525743321},
										{0.26034596607904,0.47930806784192,0.17561525743321},
										{0.47930806784192,0.26034596607904,0.17561525743321},
										{0.06513010290222,0.06513010290222,0.05334723560884},
										{0.06513010290222,0.86973979419557,0.05334723560884},
										{0.86973979419557,0.06513010290222,0.05334723560884},
										{0.31286549600487,0.63844418856981,0.07711376089026},
										{0.63844418856981,0.04869031542532,0.07711376089026},
										{0.04869031542532,0.31286549600487,0.07711376089026},
										{0.63844418856981,0.31286549600487,0.07711376089026},
										{0.31286549600487,0.04869031542532,0.07711376089026},
										{0.04869031542532,0.63844418856981,0.07711376089026}};
	MyFloat vrGaussPoint::gaussPoint_order_8[16][3]={	{0.33333333333333,0.33333333333333,0.14431560767779},
										{0.45929258829272,0.45929258829272,0.09509163426728},
										{0.45929258829272,0.08141482341455,0.09509163426728},
										{0.08141482341455,0.45929258829272,0.09509163426728},
										{0.17056930775176,0.17056930775176,0.10321737053472},
										{0.17056930775176,0.65886138449648,0.10321737053472},
										{0.65886138449648,0.17056930775176,0.10321737053472},
										{0.05054722831703,0.05054722831703,0.03245849762320},
										{0.05054722831703,0.89890554336594,0.03245849762320},
										{0.89890554336594,0.05054722831703,0.03245849762320},
										{0.26311282963464,0.72849239295540,0.02723031417443},
										{0.72849239295540,0.00839477740996,0.02723031417443},
										{0.00839477740996,0.26311282963464,0.02723031417443},
										{0.72849239295540,0.26311282963464,0.02723031417443},
										{0.26311282963464,0.00839477740996,0.02723031417443},
										{0.00839477740996,0.72849239295540,0.02723031417443}};

	vrGaussPoint::vrGaussPoint(const vrInt order):m_order(order)
	{
		if (m_order<=MAX_ORDER_NUM && m_order > 0)
		{
			init();
		}
		else
		{
			vrPrintf("unsupport quad order %d\n",m_order);
			vrPause;
		}
	}

	vrGaussPoint::~vrGaussPoint()
	{

	}

	void vrGaussPoint::init()
	{
		m_vecGaussPointPosInParamSpace.clear();
		m_vecWeightOfGaussPoint.clear();

		MyFloat  (*ptr2gaussPoint)[3]=NULL;
		switch (m_order)
		{
		case 1:
			m_nGaussPointSize = sizeof(gaussPoint_order_1) / sizeof(gaussPoint_order_1[0]);
			ptr2gaussPoint = gaussPoint_order_1;
			break;
		case 2:
			m_nGaussPointSize = sizeof(gaussPoint_order_2) / sizeof(gaussPoint_order_2[0]);
			ptr2gaussPoint = gaussPoint_order_2;
			break;
		case 3:
			m_nGaussPointSize = sizeof(gaussPoint_order_3) / sizeof(gaussPoint_order_3[0]);
			ptr2gaussPoint = gaussPoint_order_3;
			break;
		case 4:
			m_nGaussPointSize = sizeof(gaussPoint_order_4) / sizeof(gaussPoint_order_4[0]);
			ptr2gaussPoint = gaussPoint_order_4;
			break;
		case 5:
			m_nGaussPointSize = sizeof(gaussPoint_order_5) / sizeof(gaussPoint_order_5[0]);
			ptr2gaussPoint = gaussPoint_order_5;
			break;
		case 6:
			m_nGaussPointSize = sizeof(gaussPoint_order_6) / sizeof(gaussPoint_order_6[0]);
			ptr2gaussPoint = gaussPoint_order_6;
			break;
		case 7:
			m_nGaussPointSize = sizeof(gaussPoint_order_7) / sizeof(gaussPoint_order_7[0]);
			ptr2gaussPoint = gaussPoint_order_7;
			break;
		case 8:
			m_nGaussPointSize = sizeof(gaussPoint_order_8) / sizeof(gaussPoint_order_8[0]);
			ptr2gaussPoint = gaussPoint_order_8;
			break;
		default:
			{
				vrPrintf("unsupport quad order %d\n",m_order);
				vrPause;;
				break;
			}
		};

		m_vecGaussPointPosInParamSpace.resize(m_nGaussPointSize);
		m_vecWeightOfGaussPoint.resize(m_nGaussPointSize);
		for (vrInt i=0;i<m_nGaussPointSize;++i)
		{
			m_vecGaussPointPosInParamSpace[i][0] = ptr2gaussPoint[i][0];
			m_vecGaussPointPosInParamSpace[i][1] = ptr2gaussPoint[i][1];
			m_vecWeightOfGaussPoint[i] = ptr2gaussPoint[i][2];
		}

		m_matShapeFunction.resize(m_nGaussPointSize,MyDim); m_matShapeFunction.setZero();
		for (vrInt nGaussPtIdx=0; nGaussPtIdx < m_nGaussPointSize; ++nGaussPtIdx)
		{
			for (vrInt nVtxIdx=0; nVtxIdx < MyDim; ++nVtxIdx)
			{
				m_matShapeFunction.coeffRef(nGaussPtIdx,nVtxIdx) = shapeFunc_value(nGaussPtIdx,nVtxIdx);
			}
		}

	}

	MyFloat vrGaussPoint::shapeFunc_value(const vrInt nGaussPtIdx/*[0-GaussPointSize)*/, const vrInt nVtxIdx/*[0-2]*/)
	{
		Q_ASSERT((nGaussPtIdx >= 0)&&(nGaussPtIdx < m_nGaussPointSize));
		Q_ASSERT((nVtxIdx >= 0)&&(nVtxIdx < MyDim));

		switch(nVtxIdx)
		{
		case 0:
			return 1.0 - m_vecGaussPointPosInParamSpace[nGaussPtIdx][0]/* s */ - m_vecGaussPointPosInParamSpace[nGaussPtIdx][1]/* t */;
			break;
		case 1:
			return m_vecGaussPointPosInParamSpace[nGaussPtIdx][0]/* s */;
			break;
		case 2:
			return m_vecGaussPointPosInParamSpace[nGaussPtIdx][1]/* t */;
			break;
		}
	}

	MyVec3  vrGaussPoint::shapeFunc4GaussPt(const vrInt nGaussPtIdx/*[0-GaussPointSize]*/)
	{
		MyVec3 retVec;
		for (vrInt vtxIdx=0;vtxIdx < MyDim;++vtxIdx)
		{
			retVec[vtxIdx] = shapeFunc_value(nGaussPtIdx,vtxIdx);
		}
		return retVec;
	}

	MyVector vrGaussPoint::shapeFunc4Vtx(const vrInt nVtxIdx/*[0-MyDim)*/)
	{
		MyVector retVec; retVec.resize(m_nGaussPointSize);
		for (vrInt nGaussPtIdx=0; nGaussPtIdx<m_nGaussPointSize;++nGaussPtIdx)
		{
			retVec[nGaussPtIdx] = shapeFunc_value(nGaussPtIdx,nVtxIdx);
		}
		return retVec;
	}

	MyVec3 vrGaussPoint::reference_to_physical_t3(const MyVec3& gloablCoords_0, const MyVec3& gloablCoords_1, const MyVec3& gloablCoords_2, const vrInt nGaussPtIdx/*[0-GaussPointSize]*/)
	{
		MyVec3 phyCoords;
		Q_ASSERT((nGaussPtIdx >= 0)&&(nGaussPtIdx < m_nGaussPointSize));
		MyVec3 shapeFunction = m_matShapeFunction.row(nGaussPtIdx).transpose();

		phyCoords = shapeFunction[0] * gloablCoords_0 + shapeFunction[1] * gloablCoords_1 + shapeFunction[2] * gloablCoords_2;
		return phyCoords;
	}


	//maps reference points to physical points.
	void reference_to_physical_t3(const MyVec3& gloablCoords_0, const MyVec3& gloablCoords_1, const MyVec3& gloablCoords_2,
		const MyVec2ParamSpace& referenceCoords, MyVec3& phyCoords)
	{
		MyFloat shapeFunction[MyDim];
		shapeFunction[0] = 1.0 - referenceCoords[0] - referenceCoords[1];
		shapeFunction[1] = referenceCoords[0];
		shapeFunction[2] = referenceCoords[1];

		phyCoords = shapeFunction[0] * gloablCoords_0 + shapeFunction[1] * gloablCoords_1 + shapeFunction[2] * gloablCoords_2;
		//for (int i = 0; i < 2; i++)
		//{
		//	for (int j = 0; j < MyDim; j++)
		//	{
		//		/*phyCoords.coeffRef();
		//		phy[i + j * 2] = t[i + 0 * 2] * (1.0 - ref[0 + j * 2] - ref[1 + j * 2])
		//			+ t[i + 1 * 2] * +ref[0 + j * 2]
		//			+ t[i + 2 * 2] * +ref[1 + j * 2];*/
		//	}
		//}
	}
	//quad_order : 
	// 0 : (¦Î1, ¦Ç1) = (1/6, 1/6) , (¦Î2, ¦Ç2) = (2/3 , 1/6) , (¦Î3, ¦Ç3) = (1/6 , 2/3) , w1 = w2 = w3 = 1/3
	// 1 : (¦Î1, ¦Ç1) = (0, 1/2)   , (¦Î2, ¦Ç2) = (1/2 , 0)   , (¦Î3, ¦Ç3) = (1/2 , 1/2) , w1 = w2 = w3 = 1/3
	void quad_rule(const MyInt quad_order, MyDenseVector& quad_w, MyCoords_3X2& quad_xy, MyCoords_3X2& local_Coords)
	{
		//           2
		//          / .
		//         /   .
		//        /     .
		//       0-------1


		//      ^
		//    1 | 2
		//      | |.
		//    Y | | .
		//      | |  .
		//    0 | 0---1
		//      +------->
		//        0 X 1
		local_Coords.row(0) = MyVec2ParamSpace(0.0,0.0);
		local_Coords.row(1) = MyVec2ParamSpace(1.0, 0.0);
		local_Coords.row(2) = MyVec2ParamSpace(0.0, 1.0);
		if (0 == quad_order)
		{
			const MyFloat const_1_6 = 1.0 / 6.0;
			const MyFloat const_2_3 = 2.0 / 3.0;
			const MyFloat const_w = 1.0 / 3.0;

			quad_w[0] = quad_w[1] = quad_w[2] = const_w;
			quad_xy.row(0) = MyVec2ParamSpace(const_1_6, const_1_6);
			quad_xy.row(1) = MyVec2ParamSpace(const_2_3, const_1_6);
			quad_xy.row(2) = MyVec2ParamSpace(const_1_6, const_2_3);
		}
		else if (1 == quad_order)
		{
			MyError("unknow quad_order value!");
			const MyFloat const_1_2 = 0.5;
			const MyFloat const_0 = 0.0;
			const MyFloat const_w = 1.0 / 3.0;
			quad_w[0] = quad_w[1] = quad_w[2] = const_w;
			quad_xy.row(0) = MyVec2ParamSpace(const_0, const_1_2);
			quad_xy.row(1) = MyVec2ParamSpace(const_1_2, const_0);
			quad_xy.row(2) = MyVec2ParamSpace(const_1_2, const_1_2);
		}
		else
		{
			MyError("unknow quad_order value!");
		}
	}
}//namespace VR