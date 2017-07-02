#ifndef _VR_Global_Define_H
#define _VR_Global_Define_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <vector>
#include <assert.h>

namespace YC
{
	using namespace Eigen;
	typedef float MyFloat;
	typedef int   MyInt;
	typedef Eigen::MatrixXf MyDenseMatrix;
	typedef Eigen::Vector3f MyDenseVector;
	typedef MyDenseVector MyPoint;
	typedef MyDenseVector MyVec3;
	typedef Eigen::Vector3i MyVectorI;
	typedef Eigen::MatrixXf MyMatrix;
	typedef Eigen::VectorXf MyVector;
	typedef Eigen::Matrix3f MyMatrix_3X3;
	typedef Eigen::SparseMatrix<MyFloat,1> MySpMat;

	
}
#endif//_VR_Global_Define_H