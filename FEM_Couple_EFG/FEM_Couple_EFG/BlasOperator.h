#ifndef _BLASOPERATOR_H
#define _BLASOPERATOR_H

#include "VR_Global_Define.h"

namespace VR_FEM
{
	class blasOperator
	{
	public:
		static void Blas_R1_Update(MyMatrix &C, MyVector &A, MyFloat alpha , MyFloat beta );

		static void Blas_Mat_Trans_Vec_Mult(MyMatrix &A, MyVector &dx,	MyVector &dy, MyFloat alpha, MyFloat beta);

		static void Blas_Mat_Vec_Mult(MyMatrix &A, MyVector &dx, MyMatrix &dy, MyFloat alpha , MyFloat beta );

		static void Blas_Mat_Trans_Mat_Mult(const MyMatrix &A, const MyMatrix &B, MyMatrix &C, MyFloat alpha, MyFloat beta );

		static void Blas_Mat_Trans_Vec_Mult(MyMatrix &A, MyVector &dx,	MyMatrix &dy, MyFloat alpha, MyFloat beta);

		static void Blas_Mat_Mat_Mult(const MyMatrix &A, const MyMatrix &B, MyMatrix &C, MyFloat alpha , MyFloat beta );

		static void Blas_Scale(MyFloat da, MyMatrix &dx);

		static void Blas_R1_Trans_Update(MyMatrix &C, MyMatrix &A,	MyFloat alpha , MyFloat beta );
	};

}

#endif//_BLASOPERATOR_H