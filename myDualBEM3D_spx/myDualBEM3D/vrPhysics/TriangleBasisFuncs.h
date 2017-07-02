#ifndef _TriangleBasisFuncs_h_
#define _TriangleBasisFuncs_h_

#include "bemDefines.h"
namespace VR
{
	MyFloat compute_BEM_Triangle_Jacobian(const MyMatrix& contravariant)
	{
		MyFloat retV = 0.0;
		MyMatrix_2X2 m11,m12,m13;

		m11 << contravariant.coeff(1, 0) , contravariant.coeff(2, 0) , contravariant.coeff(1, 1) , contravariant.coeff(2, 1);
		m12 << contravariant.coeff(0, 0), contravariant.coeff(2, 0), contravariant.coeff(0, 1), contravariant.coeff(2, 1);
		m13 << contravariant.coeff(0, 0), contravariant.coeff(1, 0), contravariant.coeff(0, 1), contravariant.coeff(1, 1);

		const MyFloat m11_det = m11.determinant();
		const MyFloat m12_det = m12.determinant();
		const MyFloat m13_det = m13.determinant();

		retV = std::sqrtf((m11_det*m11_det) + (m12_det*m12_det) + (m13_det*m13_det));
		return retV;
	}

	MyDenseVector computeJxW(const MyCoords_3X3& global_Coords_Tri, const MyVec2 shapeDeri_3X3X2_gauss_support_localCoord[3][3], const MyDenseVector& GaussVertexWeight)
	{
		MyDenseVector JxW_values;
		MyMatrix contravariant(MyDim, MyParaDim);
		contravariant.setZero();
		for (unsigned int point = 0; point<Geometry::vertexs_per_tri; ++point)
		{
			for (unsigned int k = 0; k< Geometry::shape_Function_Count_In_BEM; ++k)
			{
				const MyVec2  &data_derv = shapeDeri_3X3X2_gauss_support_localCoord[point][k];
				const MyPoint &supp_pts = global_Coords_Tri.row(k);


				/*std::cout << data_derv << std::endl;
				std::cout << supp_pts << std::endl;*/

				for (unsigned int i = 0; i<MyDim; ++i)
				{
					for (unsigned int j = 0; j<MyParaDim; ++j)
					{
						contravariant.coeffRef(i, j) += data_derv[j] * supp_pts[i];
					}
				}
			}


			JxW_values[point] = compute_BEM_Triangle_Jacobian(contravariant)*GaussVertexWeight[point];
		}

		return JxW_values;
	}

	void TriangleBasisAndDeris(const MyCoords_3X2& localCoords, const MyCoords_3X2& localGaussPts, 
		MyMatrix& shapeFunc_3X3_gauss_support, MyVec2 shapeDeri_3X3X2_gauss_support_localCoord[3][3])
	{
		//1. compute shape function
		
		shapeFunc_3X3_gauss_support.resize(Geometry::vertexs_per_tri, Geometry::vertexs_per_tri);
		shapeFunc_3X3_gauss_support.setZero();

		for (int gaussIdx = 0; gaussIdx < Geometry::vertexs_per_tri;++gaussIdx)
		{
			/*
			N0(x,y) = 1 - x - y, // 0 is support idx
			N1(x,y) = x,           // 1 is support idx
			N2(x,y) = y.		   // 2 is support idx
			*/
			shapeFunc_3X3_gauss_support.coeffRef(gaussIdx, 0) = 1 - localGaussPts.coeff(gaussIdx, 0) - localGaussPts.coeff(gaussIdx, 1);
			shapeFunc_3X3_gauss_support.coeffRef(gaussIdx, 1) = localGaussPts.coeff(gaussIdx, 0);
			shapeFunc_3X3_gauss_support.coeffRef(gaussIdx, 2) = localGaussPts.coeff(gaussIdx, 1); 
		}

		//2. compute shape derivation
		/*shapeDeri_3X3X2_gauss_support_localCoord.resize(MyParaDim);
		for (int i = 0; i < MyParaDim; ++i)
		{
			shapeDeri_3X3X2_gauss_support_localCoord[i].resize(Geometry::vertexs_per_tri, Geometry::vertexs_per_tri);
			shapeDeri_3X3X2_gauss_support_localCoord[i].setZero();
		}*/

		for (int gaussIdx = 0; gaussIdx < Geometry::vertexs_per_tri; ++gaussIdx)
		{
			/*
			DN0(x,y)/x = -1 , // 0 is support idx
			DN1(x,y)/x = 1,           // 1 is support idx
			DN2(x,y)/x = 0.		   // 2 is support idx
			*/
			int derisIdx = 0;
			shapeDeri_3X3X2_gauss_support_localCoord[gaussIdx][0][derisIdx] = -1.0;
			shapeDeri_3X3X2_gauss_support_localCoord[gaussIdx][1][derisIdx] =  1.0;
			shapeDeri_3X3X2_gauss_support_localCoord[gaussIdx][2][derisIdx] =  0.0;

			/*shapeDeri_3X3X2_gauss_support_localCoord[derisIdx].coeffRef(gaussIdx, 0) = -1.0;
			shapeDeri_3X3X2_gauss_support_localCoord[derisIdx].coeffRef(gaussIdx, 1) =  1.0;
			shapeDeri_3X3X2_gauss_support_localCoord[derisIdx].coeffRef(gaussIdx, 2) =  0.0;*/
			/*
			DN0(x,y)/y = -1 , // 0 is support idx
			DN1(x,y)/y = 0,           // 1 is support idx
			DN2(x,y)/y = 1.		   // 2 is support idx
			*/
			derisIdx = 1;
			shapeDeri_3X3X2_gauss_support_localCoord[gaussIdx][0][derisIdx] = -1.0;
			shapeDeri_3X3X2_gauss_support_localCoord[gaussIdx][1][derisIdx] = 0.0;
			shapeDeri_3X3X2_gauss_support_localCoord[gaussIdx][2][derisIdx] = 1.0;

			/*shapeDeri_3X3X2_gauss_support_localCoord[derisIdx].coeffRef(gaussIdx, 0) = -1.0;
			shapeDeri_3X3X2_gauss_support_localCoord[derisIdx].coeffRef(gaussIdx, 1) = 0.0;
			shapeDeri_3X3X2_gauss_support_localCoord[derisIdx].coeffRef(gaussIdx, 2) = 1.0;*/

		}
	}
}
#endif//_TriangleBasisFuncs_h_