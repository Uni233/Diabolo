#ifndef _VR_GLOBAL_DEFINE_H
#define _VR_GLOBAL_DEFINE_H
#include "MyMacro.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <vector>
#include <assert.h>
#include <boost/smart_ptr.hpp>

namespace YC
{
	//using namespace Eigen;
	


#if DoublePrecision
	typedef double MyFloat;
	typedef int   MyInt;
	typedef Eigen::MatrixXd MyDenseMatrix;
	typedef Eigen::Vector3d MyDenseVector;

	typedef MyDenseVector MyPoint;
	typedef Eigen::MatrixXi MyIntMatrix;
	typedef Eigen::VectorXi MyIntVector;
	typedef Eigen::Vector3i MyVectorI;
	typedef Eigen::MatrixXd MyMatrix;
	typedef Eigen::VectorXd MyVector;
	typedef Eigen::Matrix3d MyMatrix_3X3;
	typedef Eigen::SparseMatrix<MyFloat,1> MySpMat;
#else
	typedef float MyFloat;
	typedef int   MyInt;
	typedef Eigen::MatrixXf MyDenseMatrix;
	typedef Eigen::Vector3f MyDenseVector;

	typedef MyDenseVector MyPoint;
	typedef Eigen::MatrixXf MyMatrix;
	typedef Eigen::MatrixXi MyIntMatrix;
	typedef Eigen::VectorXf MyVector;
	typedef Eigen::VectorXi MyIntVector;
	typedef Eigen::Vector3i MyVectorI;
	typedef Eigen::Matrix3f MyMatrix_3X3;
	typedef Eigen::SparseMatrix<MyFloat,1> MySpMat;
#endif
	

	typedef MySpMat cgMatrix;
	typedef MyVector cgVector;

	MyFloat NormInf(const cgVector& v);

	typedef enum{FEM=0,EFG=1,COUPLE=2,INVALIDTYPE=3} CellType;
	typedef enum{vex0 = 0,vex1=1,vex2=2,vex3=3,vex4=4,vex5=5,vex6=6,vex7=7} VertexOrder;



	namespace Geometry
	{
		const int shape_Function_Count_In_FEM = 8;
		const int gauss_Sample_Point = 8;
		const int n_tensor_pols = 8;
		 const int dofs_per_cell = 24;
		 const int dofs_per_cell_8 = 8;
		 const int vertexs_per_cell = 8;
		 const int dimensions_per_vertex = 3;
		 const int first_dof_idx = 0;
		 const int max_dofs_per_face = 12;
		 const int faces_per_cell = 6;
		 const int vertexs_per_face = 4;
		 const int sons_per_cell = 8;
		 const int lines_per_quad = 4;
		 const int lines_per_cell = 12;
		 const int vertexes_per_line = 2;
		 const int subQuads_per_quad = 4;
		 const int subLines_per_line = 2;

		 //FEM Ë³Ðò
		 static MyDenseVector step[Geometry::vertexs_per_cell] = {MyDenseVector(-1,-1,-1), MyDenseVector(1,-1,-1),
			 MyDenseVector(-1,1,-1)	, MyDenseVector(1,1,-1),
			 MyDenseVector(-1,-1,1)	, MyDenseVector(1,-1,1),
			 MyDenseVector(-1,1,1)	, MyDenseVector(1,1,1)};

		 static unsigned int index[8][3] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
	}

	namespace Material
	{
		const MyFloat GravityFactor = -9.8f;
		const MyFloat YoungModulus = 300000000;
		const MyFloat PossionRatio = 0.3333333f;
		const MyFloat Density = 7700.f;
		const MyFloat damping_alpha = 0.183f;
		const MyFloat damping_beta = 0.00128f ;
	}

	namespace Colors
	{
		typedef enum{black=0,blue=1,green=2,indigoBlue=3,red=4,pink=5,yellow=6,white=7} MyColors;
		static MyFloat colorTemplage[8][3] = {{0.0f,0.0f,0.0f},//black
		{0.0f,0.0f,1.0f},//blue
		{0.0f,1.0f,0.0f},//green
		{0.0f,1.0f,1.0f},//dian blue
		{1.0f,0.0f,0.0f},//red
		{1.0f,0.0f,1.0f},//pink
		{1.0f,1.0f,0.0f},//yellow
		{1.0f,1.0f,1.0f}};//white

		static YC::MyDenseVector purple(80.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0);
		static YC::MyDenseVector gold(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0);
	}

	namespace Order
	{
		static unsigned int indexQuads[Geometry::faces_per_cell][Geometry::vertexs_per_face] = {{0,2,4,6},{1,3,5,7},{0,1,4,5},{2,3,6,7},{0,1,2,3},{4,5,6,7}};
#if USE_MAKE_CELL_SURFACE
		static unsigned int indexQuads_Tri[Geometry::faces_per_cell][2][3] = 
		{
			{ {0,4,6}, {0,6,2} },
			{ {1,3,7}, {5,1,7} },
			{ {0,1,5}, {4,0,5} },
			{ {7,3,2}, {2,6,7} },
			{ {1,2,3}, {1,0,2} },
			{ {5,7,6}, {5,6,4} }
		};
#endif
	}
}
#endif//_VR_GLOBAL_DEFINE_H