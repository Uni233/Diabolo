#ifndef _VR_GLOBAL_DEFINE_H
#define _VR_GLOBAL_DEFINE_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <vector>
#include <assert.h>

namespace VR_FEM
{
	using namespace Eigen;
	typedef float MyFloat;
	typedef int   MyInt;
	typedef Eigen::MatrixXf MyDenseMatrix;
	typedef Eigen::Vector3f MyDenseVector;
	typedef MyDenseVector MyPoint;
	typedef Eigen::Vector3i MyVectorI;
	typedef Eigen::MatrixXf MyMatrix;
	typedef Eigen::VectorXf MyVector;
	typedef Eigen::Matrix3f MyMatrix_3X3;
	typedef Eigen::SparseMatrix<MyFloat,1> MySpMat;

	/*typedef float MyFloat;
	typedef Eigen::MatrixXf MyDenseMatrix;
	typedef Eigen::Vector3f MyDenseVector;
	typedef MyDenseVector MyPoint;
	typedef Eigen::Vector3i MyVectorI;
	typedef Eigen::MatrixXf MyMatrix;
	typedef Eigen::VectorXf MyVector;
	typedef Eigen::Matrix3f MyMatrix_3X3;
	typedef Eigen::SparseMatrix<MyFloat,1> MySpMat;*/

	typedef enum{FEM=0,EFG=1,COUPLE=2,INVALIDTYPE=3} CellType;
	typedef enum{vex0 = 0,vex1=1,vex2=2,vex3=3,vex4=4,vex5=5,vex6=6,vex7=7} VertexOrder;

#define  window_width  (1024)
#define  window_height (768)
#define Invalid_Id (-1)
#define MyZero (0)
#define MyNull (0)
#define X_AXIS (0)
#define Y_AXIS (1)
#define Z_AXIS (2)
#define MyPause system("pause")
#define MyExit exit(66)
#define Q_ASSERT(X) do{if (!(X)){printf("%s\n",#X);system("pause");}}while(false)
#define dim (3)
#define MaterialMatrixSize (6)
#define MaxTimeStep (10000)
#define EFGQuadPtsx_ (2)
#define EFGQuadPtsy_ (2)
#define EFGQuadPtsz_ (2)

#define CellRaidus   (0.015625)
//#define CellRaidus   (0.129077f)
#define SupportSize  (3.001*CellRaidus)
#define EFG_BasisNb_ (4) 

#define ShowTriangle 1
#define ShowLines 1
#define BufferSize (256)

#define Steak_MaxDiameter (1.0)
#define Steak_Translation (0.0f)
#define Steak_Translation_X (0.0f)
#define Steak_Translation_Y (0.0f)
#define Steak_Translation_Z (0.0f)
#define ValidEFGDomainId (2)
#define ValueScaleFactor (10000000000)

#define BLADE_A  0.417f, 0.4517f, (-0.1f+0.31f)
#define BLADE_B  0.417f, 0.4517f, (.2f+0.51f)
#define BLADE_C  0.417f, -0.1f,  (0.f+0.51f)

#define BLADE_E  0.6617f, 0.4517f, (-0.1f+0.3f)
#define BLADE_F  0.6617f, 0.4517f, (.2f+0.5f)
#define BLADE_G  0.6617f, -0.1f,  (0.f+0.5f)
#define USE_CO_RATION (1)

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
	}
}
#endif//_VR_GLOBAL_DEFINE_H