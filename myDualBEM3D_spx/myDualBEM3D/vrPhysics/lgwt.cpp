#include "lgwt.h"
#include "bemMatlabFunc.h"
#include <iostream>
#include "constant_numbers.h"
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
	void lgwt(MyInt N, const MyFloat a, const MyFloat b, MyVector& x, MyVector& w)
	{
		N = N - 1;
		const MyInt N1 = N + 1; 
		const MyInt N2 = N + 2;

		const MyVector xu = MatlabFunc::linspace(-1, 1, N1);// linspace(-1, 1, N1)';
		//SHOWVECTOR(xu);

		//% Initial guess
		//y = cos((2 * (0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);
		MyVector y(N1);
		MyVector tmpOnes(y); tmpOnes.setOnes();
		for (int i = 0; i < N1;++i)
		{
			y[i] = cos((2 * i + 1) * numbers::MyPI / (2 * N + 2)) + (0.27 / N1)*sin(numbers::MyPI*xu[i]*N/N2);
		}
		//SHOWVECTOR(y);
		//% Legendre - Gauss Vandermonde Matrix
		MyMatrix L(N1, N2); L.setZero();// = zeros(N1, N2);

		//% Derivative of LGVM
		MyVector Lp(N1); Lp.setZero();// = zeros(N1, N2);

		//% Compute the zeros of the N + 1 Legendre Polynomial
		//% using the recursion relation and the Newton - Raphson method

		const MyFloat y0 = 2.0;
		MyVector y0_vec(N1); y0_vec.setOnes(); y0_vec = y0_vec * y0;

		
		//% Iterate until new points are uniformly within epsilon of old points
		while ((y - y0_vec).cwiseAbs().maxCoeff() > MyEPS)//while max(abs(y-y0))>eps
		{
			L.col(1 - MyArrayBase).setOnes();// L(:, 1) = 1;
			//Lp.col(1 - MyArrayBase).setZero();// Lp(:, 1) = 0;

			L.col(2 - MyArrayBase) = y;// L(:, 2) = y;
			//Lp.col(2 - MyArrayBase).setOnes();// Lp(:, 2) = 1;

			for (int k = 2 - MyArrayBase; k < N1;++k)//for k = 2:N1
			{
				
				L.col(k+1) = (((2 * (k + MyArrayBase) - 1) * y).cwiseProduct(L.col(k)) - ((k+MyArrayBase)-1)*L.col(k-1)) / (k+MyArrayBase);
				//L(:, k + 1) = ((2 * k - 1)*y.*L(:, k) - (k - 1)*L(:, k - 1)) / k;
				
			}

			//Lp = (N2)*(L(:, N1) - y.*L(:, N2)). / (1 - y. ^ 2);
			

			const MyVector& tmp0 = (N2)*(L.col(N1 - MyArrayBase) - y.cwiseProduct(L.col(N2 - MyArrayBase)));
			const MyVector& tmp1 = (tmpOnes - y.cwiseProduct(y));
			for (int i = 0; i < N1;++i)
			{
				Lp[i] = tmp0[i] / tmp1[i];
			}
			

			y0_vec = y;// y0 = y;
			//y = y0 - L(:, N2). / Lp;
			for (int i = 0; i < N1;++i)
			{
				y[i] = y0_vec[i] - L.coeff(i, N2 - MyArrayBase) / Lp[i];
			}
			
		}//while

		
		//% Linear map from[-1, 1] to[a, b]
		//x = (a*(1 - y) + b*(1 + y)) / 2;
		x = (a*(tmpOnes - y) + b*(tmpOnes + y)) / 2.0;

		//% Compute the weights
		//w = (b - a). / ((1 - y. ^ 2).*Lp. ^ 2)*(N2 / N1) ^ 2;
		
		const MyVector& tmp2 = (tmpOnes - y.cwiseProduct(y)).cwiseProduct(Lp.cwiseProduct(Lp));
		const MyFloat   tmp3 = (MyFloat(N2) / MyFloat(N1))*(MyFloat(N2) / MyFloat(N1));
		w.resize(N1);
		for (int i = 0; i < N1;++i)
		{
			w[i] = ((b - a) / tmp2[i])*tmp3;
		}
		
	}
}//namespace YC