#ifndef __GaussElimination__H__
#define __GaussElimination__H__

#include <stdio.h>
#include "VR_Global_Define.h"

namespace YC
{
	namespace MyGaussElimination
	{
		MyVector GaussElimination(const MySpMat& A,MyVector& b);
		void GaussElimination(const MySpMat& A,const int n ,const int m ,MyVector& g,MyVector& v,const MyVector& r,const MyVector& w);
		void GaussElimination(const MyDenseMatrix& A,const int n ,const int m ,MyVector& g,MyVector& v,const MyVector& r,const MyVector& w);
	}
}



#endif /* defined(__GaussElimination__H__) */