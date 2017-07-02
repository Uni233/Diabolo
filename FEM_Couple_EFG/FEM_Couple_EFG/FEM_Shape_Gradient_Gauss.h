#ifndef _FEM_SHAPE_GRADIENT_GAUSS_H_
#define _FEM_SHAPE_GRADIENT_GAUSS_H_

#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include "VR_Global_Define.h"

namespace VR_FEM
{
	

	class FEM_Shape_Gradient_Gauss {
	public:
		FEM_Shape_Gradient_Gauss(const std::vector< MyCell >& vecCell );
		virtual ~FEM_Shape_Gradient_Gauss();
		void computeGaussPoint();
		void computeShapeFunction();
		void computeJxW();
		void computeShapeGrad();

		void test_compute();
		void print();
	private:
		void compute(const MyPoint &p, std::vector< MyFloat> &values, std::vector< MyVector3 > &grads, std::vector< MyMatrix_3X3 > &grad_grads);
		void compute_index (const unsigned int i, unsigned int  (&indices)[3]) const;
		MyFloat determinant (const MyMatrix_3X3 &t);
		MyMatrix_3X3 invert (const MyMatrix_3X3 &t);
		void contract (MyVector3 &dest, const MyVector3 &src1, const MyMatrix_3X3 &src2);
	private:

		std::vector< MyCell > m_vecCell;
		std::vector< Polynomial > Polynomials;

	};
}

#endif//_FEM_SHAPE_GRADIENT_GAUSS_H_