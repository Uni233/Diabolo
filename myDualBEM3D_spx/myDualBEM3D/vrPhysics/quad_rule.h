#ifndef _quad_rule_h_
#define _quad_rule_h_

#include "bemDefines.h"
#include <vector>
#include <Eigen/StdVector>

//param space variable (XSI,ETA)
namespace VR
{
	class vrGaussPoint
	{
		enum{MAX_ORDER_NUM=8};
	public:
		vrGaussPoint(const vrInt order);
		~vrGaussPoint();
		void init();
		MyFloat shapeFunc_value(const vrInt nGaussPtIdx/*[0-GaussPointSize]*/, const vrInt nVtxIdx/*[0-2]*/);
		MyVec3  shapeFunc4GaussPt(const vrInt nGaussPtIdx/*[0-GaussPointSize]*/);
		MyVector shapeFunc4Vtx(const vrInt nVtxIdx/*[0-2]*/);
		const std::vector< MyFloat >& gaussPtWeight()const{return m_vecWeightOfGaussPoint;}
		MyVec3 reference_to_physical_t3(const MyVec3& gloablCoords_0, const MyVec3& gloablCoords_1, const MyVec3& gloablCoords_2, const vrInt nGaussPtIdx/*[0-GaussPointSize]*/);
	private:
		const vrInt m_order;
		vrInt m_nGaussPointSize;
		std::vector< MyVec2ParamSpace,Eigen::aligned_allocator<MyVec2ParamSpace> > m_vecGaussPointPosInParamSpace;
		std::vector< MyFloat > m_vecWeightOfGaussPoint;
		vrMat m_matShapeFunction;/* m_nGaussPointSize by MyDim */
	private:
		/*
		gaussPoint - a (n by 3) matrix:
		n - the order of the Gaussian quadrature (n<=12)
		1st column gives the x-coordinates of points
		2nd column gives the y-coordinates of points
		3rd column gives the weights
		*/
		static MyFloat gaussPoint_order_1[1][3];
		static MyFloat gaussPoint_order_2[3][3];
		static MyFloat gaussPoint_order_3[4][3];
		static MyFloat gaussPoint_order_4[6][3];
		static MyFloat gaussPoint_order_5[7][3];
		static MyFloat gaussPoint_order_6[12][3];
		static MyFloat gaussPoint_order_7[13][3];
		static MyFloat gaussPoint_order_8[16][3];
	};

	//maps reference points to physical points.
	void reference_to_physical_t3(const MyVec3& gloablCoords_0, const MyVec3& gloablCoords_1, const MyVec3& gloablCoords_2,
		const MyVec2ParamSpace& referenceCoords, MyVec3& phyCoords);
	//quad_order : 
	// 0 : (¦Î1, ¦Ç1) = (1/6, 1/6) , (¦Î2, ¦Ç2) = (2/3 , 1/6) , (¦Î3, ¦Ç3) = (1/6 , 2/3) , w1 = w2 = w3 = 1/3
	// 1 : (¦Î1, ¦Ç1) = (0, 1/2)   , (¦Î2, ¦Ç2) = (1/2 , 0)   , (¦Î3, ¦Ç3) = (1/2 , 1/2) , w1 = w2 = w3 = 1/3
	void quad_rule(const MyInt quad_order, MyDenseVector& quad_w, MyCoords_3X2& quad_xy, MyCoords_3X2& local_Coords);
}
#endif//_quad_rule_h_