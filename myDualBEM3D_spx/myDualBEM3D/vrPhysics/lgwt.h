#ifndef _lgwt_h_
#define _lgwt_h_
#include "bemDefines.h"
namespace VR
{
	//% This script is for computing definite integrals using Legendre - Gauss
	//% Quadrature.Computes the Legendre - Gauss nodes and weights  on an interval
	//%[a, b] with truncation order N
	//%
	//% Suppose you have a continuous function f(x) which is defined on[a, b]
	//% which you can evaluate at any x in[a, b].Simply evaluate it at all of
	//% the values contained in the x vector to obtain a vector f.Then compute
	//% the definite integral using sum(f.*w);
	void lgwt(MyInt N, const MyFloat a, const MyFloat b, MyVector& x, MyVector& w);
}//namespace YC
#endif//_lgwt_h_