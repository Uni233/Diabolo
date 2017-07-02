#include "stdafx.h"

#include "BlasOperator.h"
#include "VR_Global_Define.h"

namespace VR_FEM
{
	
	void blasOperator::Blas_R1_Update(MyMatrix &C, MyVector &A, MyFloat alpha , MyFloat beta )
	{
		//C := alpha*A*A**T + beta*C.
		//SymmMatrix tmp(A);
		C = alpha * (A * A.transpose()) + beta * C;
	}

	void blasOperator::Blas_Mat_Trans_Vec_Mult(MyMatrix &A, MyVector &dx,	MyVector &dy, MyFloat alpha, MyFloat beta)
	{
		dy = (alpha * dx.transpose() * A ).transpose() + beta * dy;
		//dy = (alpha * dx.transpose() * A ).transpose();
	}

	void blasOperator::Blas_Mat_Vec_Mult(MyMatrix &A, MyVector &dx, MyMatrix &dy, MyFloat alpha , MyFloat beta )
	{
		dy = alpha * A * dx + beta * dy;
	}

	void blasOperator::Blas_Mat_Trans_Mat_Mult(const MyMatrix &A, const MyMatrix &B, MyMatrix &C, MyFloat alpha, MyFloat beta )
	{
		C = alpha * A.transpose() * B + beta * C;
	}

	void blasOperator::Blas_Mat_Trans_Vec_Mult(MyMatrix &A, MyVector &dx,	MyMatrix &dy, MyFloat alpha, MyFloat beta)
	{
		dy = (alpha * dx.transpose() * A ).transpose() + beta * dy;
		//dy = (alpha * dx.transpose() * A ).transpose();
	}

	void blasOperator::Blas_Mat_Mat_Mult(const MyMatrix &A, const MyMatrix &B, MyMatrix &C, MyFloat alpha , MyFloat beta )
	{
		C = alpha * A * B + beta * C;
	}

	void blasOperator::Blas_Scale(MyFloat da, MyMatrix &dx)
	{
		dx *= da;
	}

	void blasOperator::Blas_R1_Trans_Update(MyMatrix &C, MyMatrix &A,	MyFloat alpha , MyFloat beta )
	{
		C = alpha * A.transpose() * A + beta * C;
	}
}