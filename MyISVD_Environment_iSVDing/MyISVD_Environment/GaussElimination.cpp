#include "GaussElimination.h"

namespace YC
{
	namespace MyGaussElimination
	{
		void GaussElimination(const MyDenseMatrix& A,const int n ,const int m ,MyVector& g,MyVector& v,const MyVector& r,const MyVector& w)
		{
			printf("GaussElimination start \n");
			Q_ASSERT(A.rows() == (n+m));
			g.resize(n);
			v.resize(m);
			Q_ASSERT(n == r.size());
			Q_ASSERT(m == w.size());
			MyVector b(n+m);
			b.block(0,0,n,1) = r;
			b.block(n,0,m,1) = w;

			MySpMat refA(A.rows(),A.cols());
			for (int r=0;r<A.rows();++r)
			{
				for (int c=0;c<A.cols();++c)
				{
					refA.coeffRef(r,c) = A.coeff(r,c);
				}				
			}
			
			MyVector ret = GaussElimination(refA,b);
			g = ret.block(0,0,n,1);
			v = ret.block(n,0,m,1);

			printf("GaussElimination end \n");
		}

		void GaussElimination(const MySpMat& A,const int n ,const int m ,MyVector& g,MyVector& v,const MyVector& r,const MyVector& w)
		{
			printf("GaussElimination start \n");
			Q_ASSERT(A.rows() == (n+m));
			g.resize(n);
			v.resize(m);
			Q_ASSERT(n == r.size());
			Q_ASSERT(m == w.size());
			MyVector b(n+m);
			b.block(0,0,n,1) = r;
			b.block(n,0,m,1) = w;

			MySpMat refA = A;
			MyVector ret = GaussElimination(refA,b);
			g = ret.block(0,0,n,1);
			v = ret.block(n,0,m,1);

			printf("GaussElimination end \n");
		}

		MyVector GaussElimination(const MySpMat& K,MyVector& b)
		{			
			const int n = K.rows();

			MyMatrix AK(n,n+1);AK.setZero();
			AK.block(0,0,n,n) = K;
			AK.block(0,n,n,1) = b;
			for (int i=0; i<n; i++) {
				// Search for maximum in this column
				double maxEl = abs(AK.coeff(i,i)/*[i][i]*/);
				int maxRow = i;
				for (int k=i+1; k<n; k++) {
					if (abs(AK.coeff(k,i)/*[k][i]*/) > maxEl) {
						maxEl = abs(AK.coeff(k,i)/*[k][i]*/);
						maxRow = k;
					}
				}

				// Swap maximum row with current row (column by column)
				for (int k=i; k<n+1;k++) {
					double tmp = AK.coeff(maxRow,k);//A[maxRow][k];
					/*A[maxRow][k]*/AK.coeffRef(maxRow,k) = AK.coeff(i,k);//A[i][k];
					/*A[i][k]*/AK.coeffRef(i,k) = tmp;
				}

				// Make all rows below this one 0 in current column
				for (int k=i+1; k<n; k++) {
					double c = - AK.coeff(k,i) / AK.coeff(i,i);/*A[k][i]/A[i][i]*/;
					for (int j=i; j<n+1; j++) {
						if (i==j) {
							/*A[k][j]*/ AK.coeffRef(k,j) = 0;
						} else {
							/*A[k][j]*/AK.coeffRef(k,j) += c * AK.coeff(i,j) /*A[i][j]*/;
						}
					}
				}
			}

			// Solve equation Ax=b for an upper triangular matrix A
			MyVector x(n);
			for (int i=n-1; i>=0; i--) {
				x[i] = AK.coeff(i,n)/AK.coeff(i,i) ;//A[i][n]/A[i][i];
				for (int k=i-1;k>=0; k--) {
					/*A[k][n]*/AK.coeffRef(k,n) -= /*A[k][i]*/AK.coeff(k,i) * x[i];
				}
			}
			return x;
		}
	}
}