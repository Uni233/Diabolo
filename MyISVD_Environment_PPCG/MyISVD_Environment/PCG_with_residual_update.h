#ifndef _PCG_with_residual_update_H_
#define _PCG_with_residual_update_H_

#include <stdio.h>
#include "VR_Global_Define.h"
#include "PCG_with_residual_update_Monitor.h"

namespace YC
{
	namespace PCG_RU
	{
		MyVector PCG_Residual_Update_2(const MyDenseMatrix& H, const MyDenseMatrix& B, const MyDenseMatrix& C, const MyVector& c, const MyVector& d, const MyDenseMatrix& P, MyMonitor<MyFloat>& monitor);
		MyVector PCG_Residual_Update(const MyDenseMatrix& A, const MyDenseMatrix& B, const MyDenseMatrix& C, const MyVector& c, const MyVector& d, const MyDenseMatrix& P, MyMonitor<MyFloat>& monitor);
		MyVector PCG_Residual_Update(const MySpMat& A, const MySpMat& B, const MyVector& c, const MyVector& d,const MySpMat& P, MyMonitor<MyFloat>& monitor);
		MyVector PCG_Residual_Update_Sparse(const MySpMat& A, const MySpMat& B, const MySpMat& C, const MyVector& c, const MyVector& d, const MySpMat& P, MyMonitor<MyFloat>& monitor);
		MyVector PCG_Residual_Update_Sparse_Schilders(const MySpMat& A, const MySpMat& B, const MySpMat& B1, const MySpMat& B2, const MySpMat& C, const MyVector& c, const MyVector& d, MyMonitor<MyFloat>& monitor);
		MyVector PCG_Residual_Update_Dense_Schilders(
			const MyMatrix& A, const MyMatrix& B, const MyMatrix& B1, const MyMatrix& B2, const MyMatrix& C, const MyMatrix& mat_G22_Inv, const MyMatrix& mat_B1_Inv, const MyVector& c, const MyVector& d, MyMonitor<MyFloat>& monitor);
	}
}
#endif//_PCG_with_residual_update_H_