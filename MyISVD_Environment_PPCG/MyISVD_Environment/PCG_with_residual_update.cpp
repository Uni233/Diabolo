#include "PCG_with_residual_update.h"
#include "GaussElimination.h"
#include "constant_numbers.h"
#include <iostream>
#include <fstream>
#include "VR_Global_Define.h"
namespace YC
{
	namespace PCG_RU
	{
		MyFloat my_dot(const MyVector& a, const MyVector& g)
		{
			return a.dot(g);
		}
		void residual_update(MyVector& a, MyVector& g, MyVector& r, MyVector& v, MyVector& w, const MySpMat& B, const MySpMat& P)
		{
			r = r - B.transpose() * v;
			a = a + v;
			w.setZero();
			MyGaussElimination::GaussElimination(P,g.size(),v.size(),g,v,r,w);
		}

		void printfMTX(const char* lpszFileName, const MySpMat& sparseMat)
		{
			std::ofstream outfile(lpszFileName);
			const int nDofs = sparseMat.rows();
			const int nValCount = sparseMat.nonZeros();
			int nValCount_0 = 0;

			outfile << nDofs << " " << nDofs << " " << nValCount << std::endl;
			for (int k=0; k<sparseMat.outerSize(); ++k)
			{
				for (MySpMat::InnerIterator it(sparseMat,k); it; ++it)
				{
					//if ( !numbers::isEqual( it.value() , sparseMat.coeff(it.col(),it.row()),0.00000001 ) )
					{
						//printf("(%d,%d) (%f)----(%f)\n",it.row(),it.col(),it.value(),obj.coeff(it.col(),it.row()));
						++nValCount_0;
						outfile << it.row() << " " << it.col() << " " << it.value() << std::endl;
					}
				}
			}
			printf("nValCount_0(%d) == nValCount(%d) \n",nValCount_0 , nValCount);
			Q_ASSERT(nValCount_0 == nValCount);
			outfile.close();
		}

		void ProjectedPrecondition_Schilders_Dense(const MyMatrix& G22_Inv, const MyMatrix& A1_Inv, const MyMatrix& A2, const int n, const int m, MyVector& g, MyVector& v, const MyVector& r, const MyVector& w)
		{
			const int n_m = n - m;
#if 0
			static bool bFirst = true;
			static MyMatrix A1_Inv;
			static MyMatrix/*ColPivHouseholderQR< MyMatrix >*/ H22_Inv;
			if (bFirst)
			{
				bFirst = false;
				{
					MyMatrix tmp_A1(m, m);
					tmp_A1.setZero();

					for (int r = 0; r<m; ++r)
					{
						for (int c = 0; c<m; ++c)
						{
							tmp_A1.coeffRef(r, c) = A1.coeff(r, c);
						}
					}
					A1_Inv = tmp_A1.inverse();
					/*std::ofstream outfileA1("d:\\A1.txt");
					std::ofstream outfileA1_Inv("d:\\A1_Inv.txt");



					outfileA1 << tmp_A1 << std::endl;outfileA1.close();
					outfileA1_Inv << A1_Inv*tmp_A1  << std::endl;outfileA1_Inv.close();
					MyExit;*/
				}
				{
					MyMatrix tmp_H22(n_m, n_m);
					tmp_H22.setZero();
					H22_Inv.resize(n_m, n_m); H22_Inv.setZero();
					for (int r = 0; r<n_m; ++r)
					{

						H22_Inv.coeffRef(r, r) = 1.f / H.coeff(m + r, m + r);
						/*for (int c=0;c<n_m;++c)
						{
						tmp_H22.coeffRef(r,c) = H.coeff(m+r,m+c);
						}*/
					}
					//H22_Inv = tmp_H22.colPivHouseholderQr();
				}
			}
#endif
			MyVector g1(m), g2(n_m);
			MyVector r1(m), r2(n_m);
			r1 = r.block(0, 0, m, 1);
			r2 = r.block(m, 0, n_m, 1);

			v = A1_Inv.transpose() * r1;
			MyVector x2 = r2 - A2.transpose()*v;
			g2 = G22_Inv * (x2);//g2 = H22_Inv.solve(x2);
			g1 = A1_Inv * (w - A2 * g2);

			g.block(0, 0, m, 1) = g1;
			g.block(m, 0, n_m, 1) = g2;
		}

		void ProjectedPrecondition_Schilders(const MySpMat& H,const MySpMat& A1,const MySpMat& A2,const int n ,const int m ,MyVector& g,MyVector& v,const MyVector& r,const MyVector& w)
		{
			const int n_m = n-m;
			static bool bFirst = true;
			static MyMatrix A1_Inv;
			static MyMatrix/*ColPivHouseholderQR< MyMatrix >*/ H22_Inv;
			if (bFirst)
			{
				bFirst = false;
				{
					MyMatrix tmp_A1(m,m);
					tmp_A1.setZero();

					for (int r=0;r<m;++r)
					{
						for (int c=0;c<m;++c)
						{
							tmp_A1.coeffRef(r,c) = A1.coeff(r,c);
						}
					}
					A1_Inv = tmp_A1.inverse();
					/*std::ofstream outfileA1("d:\\A1.txt");
					std::ofstream outfileA1_Inv("d:\\A1_Inv.txt");

					
					
					outfileA1 << tmp_A1 << std::endl;outfileA1.close();
					outfileA1_Inv << A1_Inv*tmp_A1  << std::endl;outfileA1_Inv.close();
					MyExit;*/
				}
				{
					MyMatrix tmp_H22(n_m,n_m);
					tmp_H22.setZero();
					H22_Inv.resize(n_m,n_m);H22_Inv.setZero();
					for (int r=0;r<n_m;++r)
					{
						
						H22_Inv.coeffRef(r,r) = 1.f/H.coeff(m+r,m+r);
						/*for (int c=0;c<n_m;++c)
						{
							tmp_H22.coeffRef(r,c) = H.coeff(m+r,m+c);
						}*/
					}
					//H22_Inv = tmp_H22.colPivHouseholderQr();
				}
			}
			MyVector g1(m),g2(n_m);
			MyVector r1(m),r2(n_m);
			r1 = r.block(0,0,m,1);
			r2 = r.block(m,0,n_m,1);

			v = A1_Inv.transpose() * r1;
			MyVector x2 = r2 - A2.transpose()*v;
			g2 = H22_Inv * (x2);//g2 = H22_Inv.solve(x2);
			g1 = A1_Inv * (w - A2 * g2);

			g.block(0,0,m,1) = g1;
			g.block(m,0,n_m,1) = g2;
		}

		void residual_update_Schilders_Dense(MyVector& a, MyVector& g, MyVector& r, MyVector& v, MyVector& w, 
			const MyMatrix& B, const MyMatrix& mat_G22_Inv, const MyMatrix& mat_B1_Inv, const MyMatrix& B2)
		{
			r = r - B.transpose() * v;
			a = a + v;
			w.setZero();
			ProjectedPrecondition_Schilders_Dense(mat_G22_Inv, mat_B1_Inv, B2, g.size(), v.size(), g, v, r, w);
			//MyGaussElimination::GaussElimination(P,g.size(),v.size(),g,v,r,w);
		}

		void residual_update_Schilders(MyVector& a, MyVector& g, MyVector& r, MyVector& v, MyVector& w, const MySpMat& B, const MySpMat& H,const MySpMat& B1,const MySpMat& B2)
		{
			r = r - B.transpose() * v;
			a = a + v;
			w.setZero();
			ProjectedPrecondition_Schilders(H,B1,B2,g.size(),v.size(),g,v,r,w);
			//MyGaussElimination::GaussElimination(P,g.size(),v.size(),g,v,r,w);
		}

		void ProjectedPrecondition(const MyDenseMatrix& P,const int n ,const int m ,MyVector& g,MyVector& v,const MyVector& r,const MyVector& w)
		{
			MyVector b(n+m);
			b.block(0,0,n,1) = r;
			b.block(n,0,m,1) = w;

			MyVector ret = P.inverse() * b;
			g = ret.block(0,0,n,1);
			v = ret.block(n,0,m,1);

			//std::cout << "ProjectedPrecondition :" << (P*ret - b).transpose() << std::endl;
		}

		void ProjectedPrecondition(const MySpMat& P,const int n ,const int m ,MyVector& g,MyVector& v,const MyVector& r,const MyVector& w)
		{
			MyVector b(n+m);
			b.block(0,0,n,1) = r;
			b.block(n,0,m,1) = w;

			static bool bFirst = true;
			static Eigen::ColPivHouseholderQR< MyMatrix > ALM_QR;
			if (bFirst)
			{
				bFirst = false;
				MyMatrix tmp(P.rows(),P.cols());
				tmp.setZero();

				for (int r=0;r<P.rows();++r)
				{
					for (int c=0;c<P.cols();++c)
					{
						tmp.coeffRef(r,c) = P.coeff(r,c);
					}
				}
				ALM_QR = tmp.colPivHouseholderQr();
			}

			MyVector ret = ALM_QR.solve(b);
			g = ret.block(0,0,n,1);
			v = ret.block(0,0,m,1);

			//MyGaussElimination::GaussElimination(P,g.size(),v.size(),g,v,r,w);

			/*MyVector ret = P.inverse() * b;
			g = ret.block(0,0,n,1);
			v = ret.block(n,0,m,1);*/

			//std::cout << "ProjectedPrecondition :" << (P*ret - b).transpose() << std::endl;
		}

		void residual_update(MyVector& a, MyVector& g, MyVector& r, MyVector& v, MyVector& w, const MyDenseMatrix& B, const MyDenseMatrix& C, const MyDenseMatrix& P)
		{
			r = r - B.transpose() * v;
			a = a + v;
			w = C*a;
			//w.setZero();
			//MyGaussElimination::GaussElimination(P,g.size(),v.size(),g,v,r,w);
			ProjectedPrecondition(P,g.size(),v.size(),g,v,r,w);
		}

		bool need_residual_update(const MyVector& g, const MyVector& v)
		{
			const MyFloat update_tol = 1.e-5;
			return (g.norm() <= (update_tol * v.norm()));
		}

		

		MyVector Recovering_y_from_x(const MyDenseMatrix& A,const MyDenseMatrix& B,const MyDenseMatrix& P, const MyVector& x, const MyVector& c, const MyVector& d)
		{
			const int n = A.rows();
			const int m = B.rows();

			MyVector rhs(n+m);
			rhs.block(0,0,n,1) = c - A * x;
			rhs.block(n,0,m,1) = d - B * x;

			return (P.inverse() * rhs).block(n,0,m,1);
		}

		MyVector Recovering_y_from_x_Sparse(const MySpMat& A,const MySpMat& B,const MySpMat& P, const MyVector& x, const MyVector& c, const MyVector& d)
		{
			const int n = A.rows();
			const int m = B.rows();

			MyVector rhs(n+m);
			rhs.block(0,0,n,1) = c - A * x;
			rhs.block(n,0,m,1) = d - B * x;
			return MyGaussElimination::GaussElimination(P,rhs).block(n,0,m,1);
			//return (P.inverse() * rhs).block(n,0,m,1);
		}

		MyVector PCG_Residual_Update_Dense_Schilders(
			const MyMatrix& A, const MyMatrix& B, const MyMatrix& B1, const MyMatrix& B2, const MyMatrix& C, const MyMatrix& mat_G22_Inv, const MyMatrix& mat_B1_Inv, const MyVector& c, const MyVector& d, MyMonitor<MyFloat>& monitor)
		{
			static int largestLoop = 0;
			printf("PCG_Residual_Update start\n");
			const MyFloat curvature_tol = 1.e-5;

			const int n = A.rows();
			const int m = B.rows();
			Q_ASSERT(m<n);
			MyVector x(n), x_hat(n), r(n), g(n), vec_p(n), q(n);
			MyVector a(m), w(m), y_hat(m), v(m), t(m), h(m), l(m);
			a.setZero();
			w.setZero();

			x.setZero();//initial assumption x = 0;

			r.setZero(); w = d - B*x;
			//printfMTX("d:\\B.mtx",B);printfMTX("d:\\B1.mtx",B1);printfMTX("d:\\B2.mtx",B2);MyExit;
			ProjectedPrecondition_Schilders_Dense(mat_G22_Inv, mat_B1_Inv, B2, n, m, x_hat, y_hat, r, w);
			//ProjectedPrecondition_Schilders(A, B1, B2, n, m, x_hat, y_hat, r, w);//(P,n,m,x_hat,y_hat,r,w);
			//std::cout << "x_hat " << x_hat.transpose() << std::endl;
			//std::cout << "y_hat " << y_hat.transpose() << std::endl;

			x = x + x_hat;

			r = A*x + B.transpose()*y_hat - c;
			//r = -1.f * c;
			//std::cout << "r " << r.transpose() << std::endl;
			a.setZero();
			w.setZero();

			ProjectedPrecondition_Schilders_Dense(mat_G22_Inv, mat_B1_Inv, B2, n, m, g, v, r, w);
			//ProjectedPrecondition_Schilders(A, B1, B2, n, m, g, v, r, w);

			if (need_residual_update(g, v))
			{
				residual_update_Schilders_Dense(a, g, r, v, w, B, mat_G22_Inv, mat_B1_Inv, B2);
				//residual_update_Schilders(a, g, r, v, w, B, A, B1, B2);
				//sidual_update(a,g,r,v,w,B,C,P);
			}
			t = v + a;
			vec_p = -1.f*g;
			h = -1.f * t;
			q = A * vec_p;
			l = C*h;
			//std::cout << "t " << t.transpose() << std::endl;
			//std::cout << "h " << h.transpose() << std::endl;

			MyFloat sigma = my_dot(r, g) + my_dot(w, t);
			MyFloat old_sigma;
			MyFloat gamma = my_dot(vec_p, q) + my_dot(h, l);
			MyFloat alpha, beta;
			MyFloat r_nrm2 = my_dot(r, g);

			if (monitor.finished(r_nrm2))
			{
				return x;
			}

			if (sigma < 0.f || gamma < curvature_tol)
			{
				printf("Curvature too small and method has broken down. 1\n"); MyPause;
			}

			while (!monitor.finished(r_nrm2))
			{
				alpha = sigma / gamma;
				x = x + alpha * vec_p;
				//std::cout << "x " << x.transpose() << std::endl;
				r = r + alpha * q;
				a = a + alpha * h;
				w = w + alpha * l;

				ProjectedPrecondition_Schilders_Dense(mat_G22_Inv, mat_B1_Inv, B2, n, m, g, v, r, w);
				//ProjectedPrecondition_Schilders(A, B1, B2, n, m, g, v, r, w);//ProjectedPrecondition(P,n,m,g,v,r,w);
				//MyGaussElimination::GaussElimination(P,n,m,g,v,r,w);

#if USE_MY_PPCG_OPTIMIZE
				//if (need_residual_update(g,v))
#else
				if (need_residual_update(g, v))
#endif				
				{
					residual_update_Schilders_Dense(a, g, r, v, w, B, mat_G22_Inv, mat_B1_Inv, B2);
					//residual_update_Schilders(a, g, r, v, w, B, A, B1, B2);
					//residual_update(a,g,r,v,w,B,C,P);
				}

				t = a + v;
				old_sigma = sigma;

				sigma = my_dot(r, g) + my_dot(w, t);
				beta = sigma / old_sigma;

				vec_p = -1.f * g + beta * vec_p;
				h = -1.f * t + beta * h;
				q = A * vec_p;
				l = C * h;

				gamma = my_dot(vec_p, q) + my_dot(h, l);
				++monitor;

				if (sigma < 0.f || gamma < curvature_tol)
				{
					printf("Curvature too small and method has broken down. 2 [%f]<[%f] [%f]<[%f] sigma = %f + %f.\n", sigma, 0.f, gamma, curvature_tol, my_dot(r, g), my_dot(w, t));//MyPause;
				}
				//printf("PCG_Residual_Update end iter (%d)\n",monitor.iteration_count());

				r_nrm2 = my_dot(r, g);
			}

			if (monitor.iteration_count() > largestLoop)
			{
				largestLoop = monitor.iteration_count();
			}
			printf("PCG_Residual_Update largestLoop (%d)\n", largestLoop);
			/*MyVector y = Recovering_y_from_x_Sparse(A,B,P,x,c,d);
			std::cout << "x " << x.transpose() << std::endl;
			std::cout << "y " << y.transpose() << std::endl;*/
			return x;
		}

		MyVector PCG_Residual_Update_Sparse_Schilders(const MySpMat& A, const MySpMat& B, const MySpMat& B1, const MySpMat& B2, const MySpMat& C, const MyVector& c, const MyVector& d, MyMonitor<MyFloat>& monitor)
		{
			static int largestLoop = 0;
			printf("PCG_Residual_Update start\n");
			const MyFloat curvature_tol = 1.e-5;

			const int n = A.rows();
			const int m = B.rows();
			Q_ASSERT(m<n);
			MyVector x(n),x_hat(n),r(n),g(n),vec_p(n),q(n);
			MyVector a(m),w(m),y_hat(m),v(m),t(m),h(m),l(m);
			a.setZero();
			w.setZero();

			x.setZero();//initial assumption x = 0;

			r.setZero(); w = d - B*x;
			//printfMTX("d:\\B.mtx",B);printfMTX("d:\\B1.mtx",B1);printfMTX("d:\\B2.mtx",B2);MyExit;
			ProjectedPrecondition_Schilders(A,B1,B2,n,m,x_hat,y_hat,r,w);//(P,n,m,x_hat,y_hat,r,w);
			//std::cout << "x_hat " << x_hat.transpose() << std::endl;
			//std::cout << "y_hat " << y_hat.transpose() << std::endl;

			x = x + x_hat;

			r = A*x + B.transpose()*y_hat-c;
			//r = -1.f * c;
			//std::cout << "r " << r.transpose() << std::endl;
			a.setZero();
			w.setZero();

			ProjectedPrecondition_Schilders(A,B1,B2,n,m,g,v,r,w);

			if (need_residual_update(g,v))
			{
				residual_update_Schilders(a,g,r,v,w,B,A,B1,B2);
				//sidual_update(a,g,r,v,w,B,C,P);
			}
			t = v+a;
			vec_p = -1.f*g;
			h = -1.f * t;
			q = A * vec_p;
			l = C*h;
			//std::cout << "t " << t.transpose() << std::endl;
			//std::cout << "h " << h.transpose() << std::endl;

			MyFloat sigma = my_dot(r,g) + my_dot(w,t);
			MyFloat old_sigma;
			MyFloat gamma = my_dot(vec_p,q) + my_dot(h,l);
			MyFloat alpha,beta;
			MyFloat r_nrm2 = my_dot(r,g);

			if (monitor.finished(r_nrm2))
			{
				return x;
			}

			if (sigma < 0.f || gamma < curvature_tol)
			{
				printf("Curvature too small and method has broken down. 1\n");MyPause;
			}

			while (!monitor.finished(r_nrm2))
			{
				alpha = sigma / gamma;
				x = x + alpha * vec_p;
				//std::cout << "x " << x.transpose() << std::endl;
				r = r + alpha * q;
				a = a + alpha * h;
				w = w + alpha * l;

				ProjectedPrecondition_Schilders(A,B1,B2,n,m,g,v,r,w);//ProjectedPrecondition(P,n,m,g,v,r,w);
				//MyGaussElimination::GaussElimination(P,n,m,g,v,r,w);

#if USE_MY_PPCG_OPTIMIZE
				//if (need_residual_update(g,v))
#else
				if (need_residual_update(g,v))
#endif				
				{
					residual_update_Schilders(a,g,r,v,w,B,A,B1,B2);
					//residual_update(a,g,r,v,w,B,C,P);
				}

				t = a + v;
				old_sigma = sigma;

				sigma = my_dot(r,g) + my_dot(w,t);
				beta = sigma / old_sigma;

				vec_p = -1.f * g + beta * vec_p;
				h = -1.f * t + beta * h;
				q = A * vec_p;
				l = C * h;

				gamma = my_dot(vec_p,q) + my_dot(h,l);
				++monitor;

				if (sigma < 0.f || gamma < curvature_tol)
				{
					printf("Curvature too small and method has broken down. 2 [%f]<[%f] [%f]<[%f] sigma = %f + %f.\n",sigma,0.f,gamma,curvature_tol,my_dot(r,g),my_dot(w,t));//MyPause;
				}
				//printf("PCG_Residual_Update end iter (%d)\n",monitor.iteration_count());

				r_nrm2 = my_dot(r,g);
			}

			if (monitor.iteration_count() > largestLoop)
			{
				largestLoop = monitor.iteration_count();
			}
			printf("PCG_Residual_Update largestLoop (%d)\n",largestLoop);
			/*MyVector y = Recovering_y_from_x_Sparse(A,B,P,x,c,d);
			std::cout << "x " << x.transpose() << std::endl;
			std::cout << "y " << y.transpose() << std::endl;*/
			return x;
		}

		MyVector PCG_Residual_Update_Sparse(const MySpMat& A, const MySpMat& B, const MySpMat& C, const MyVector& c, const MyVector& d, const MySpMat& P, MyMonitor<MyFloat>& monitor)
		{
			printf("PCG_Residual_Update start\n");
			const MyFloat curvature_tol = 1.e-5;

			const int n = A.rows();
			const int m = B.rows();
			Q_ASSERT(m<n);
			MyVector x(n),x_hat(n),r(n),g(n),vec_p(n),q(n);
			MyVector a(m),w(m),y_hat(m),v(m),t(m),h(m),l(m);
			a.setZero();
			w.setZero();

			x.setZero();//initial assumption x = 0;

			r.setZero(); w = d - B*x;
			ProjectedPrecondition(P,n,m,x_hat,y_hat,r,w);
			//std::cout << "x_hat " << x_hat.transpose() << std::endl;
			//std::cout << "y_hat " << y_hat.transpose() << std::endl;

			x = x + x_hat;

			r = A*x + B.transpose()*y_hat-c;
			//r = -1.f * c;
			//std::cout << "r " << r.transpose() << std::endl;
			a.setZero();
			w.setZero();

			ProjectedPrecondition(P,n,m,g,v,r,w);

			while (need_residual_update(g,v))
			{
				residual_update(a,g,r,v,w,B,C,P);
			}
			t = v+a;
			vec_p = -1.f*g;
			h = -1.f * t;
			q = A * vec_p;
			l = C*h;
			//std::cout << "t " << t.transpose() << std::endl;
			//std::cout << "h " << h.transpose() << std::endl;

			MyFloat sigma = my_dot(r,g) + my_dot(w,t);
			MyFloat old_sigma;
			MyFloat gamma = my_dot(vec_p,q) + my_dot(h,l);
			MyFloat alpha,beta;
			MyFloat r_nrm2 = my_dot(r,g);

			if (monitor.finished(r_nrm2))
			{
				return x;
			}

			if (sigma < 0.f || gamma < curvature_tol)
			{
				printf("Curvature too small and method has broken down. 1\n");MyPause;
			}

			while (!monitor.finished(r_nrm2))
			{
				alpha = sigma / gamma;
				x = x + alpha * vec_p;
				//std::cout << "x " << x.transpose() << std::endl;
				r = r + alpha * q;
				a = a + alpha * h;
				w = w + alpha * l;

				ProjectedPrecondition(P,n,m,g,v,r,w);
				//MyGaussElimination::GaussElimination(P,n,m,g,v,r,w);

				while (need_residual_update(g,v))
				{
					residual_update(a,g,r,v,w,B,C,P);
				}

				t = a + v;
				old_sigma = sigma;

				sigma = my_dot(r,g) + my_dot(w,t);
				beta = sigma / old_sigma;

				vec_p = -1.f * g + beta * vec_p;
				h = -1.f * t + beta * h;
				q = A * vec_p;
				l = C * h;

				gamma = my_dot(vec_p,q) + my_dot(h,l);
				++monitor;

				if (sigma < 0.f || gamma < curvature_tol)
				{
					printf("Curvature too small and method has broken down. 2 [%f]<[%f] [%f]<[%f] sigma = %f + %f.\n",sigma,0.f,gamma,curvature_tol,my_dot(r,g),my_dot(w,t));//MyPause;
				}
				printf("PCG_Residual_Update end iter (%d)\n",monitor.iteration_count());

				r_nrm2 = my_dot(r,g);
			}
			/*MyVector y = Recovering_y_from_x_Sparse(A,B,P,x,c,d);
			std::cout << "x " << x.transpose() << std::endl;
			std::cout << "y " << y.transpose() << std::endl;*/
			return x;
		}
		
		MyVector PCG_Residual_Update(const MyDenseMatrix& A, const MyDenseMatrix& B, const MyDenseMatrix& C, const MyVector& c, const MyVector& d, const MyDenseMatrix& P, MyMonitor<MyFloat>& monitor)
		{
			printf("PCG_Residual_Update start\n");
			const MyFloat curvature_tol = 1.e-5;

			const int n = A.rows();
			const int m = B.rows();
			Q_ASSERT(m<n);
			MyVector x(n),x_hat(n),r(n),g(n),vec_p(n),q(n);
			MyVector a(m),w(m),y_hat(m),v(m),t(m),h(m),l(m);
			a.setZero();
			w.setZero();

			x.setZero();//initial assumption x = 0;

			r.setZero(); w = d - B*x;
			ProjectedPrecondition(P,n,m,x_hat,y_hat,r,w);
			std::cout << "x_hat " << x_hat.transpose() << std::endl;
			std::cout << "y_hat " << y_hat.transpose() << std::endl;

			x = x + x_hat;

			r = A*x + B.transpose()*y_hat-c;
			//r = -1.f * c;
			std::cout << "r " << r.transpose() << std::endl;
			a.setZero();
			w.setZero();

			ProjectedPrecondition(P,n,m,g,v,r,w);

			if (need_residual_update(g,v))
			{
				residual_update(a,g,r,v,w,B,C,P);
			}
			t = v+a;
			vec_p = -1.f*g;
			h = -1.f * t;
			q = A * vec_p;
			l = C*h;
			std::cout << "t " << t.transpose() << std::endl;
			std::cout << "h " << h.transpose() << std::endl;

			MyFloat sigma = my_dot(r,g) + my_dot(w,t);
			MyFloat old_sigma;
			MyFloat gamma = my_dot(vec_p,q) + my_dot(h,l);
			MyFloat alpha,beta;
			MyFloat r_nrm2 = my_dot(r,g);

			if (monitor.finished(r_nrm2))
			{
				return x;
			}

			if (sigma < 0.f || gamma < curvature_tol)
			{
				printf("Curvature too small and method has broken down. 1\n");MyPause;
			}

			while (!monitor.finished(r_nrm2))
			{
				alpha = sigma / gamma;
				x = x + alpha * vec_p;
				std::cout << "x " << x.transpose() << std::endl;
				r = r + alpha * q;
				a = a + alpha * h;
				w = w + alpha * l;

				ProjectedPrecondition(P,n,m,g,v,r,w);
				//MyGaussElimination::GaussElimination(P,n,m,g,v,r,w);

				if (need_residual_update(g,v))
				{
					residual_update(a,g,r,v,w,B,C,P);
				}

				t = a + v;
				old_sigma = sigma;

				/*std::cout << "r " << r.transpose() << std::endl;
				std::cout << "g " << g.transpose() << std::endl;
				std::cout << "w " << w.transpose() << std::endl;
				std::cout << "l " << l.transpose() << std::endl;
				std::cout << "t " << t.transpose() << std::endl;*/

				sigma = my_dot(r,g) + my_dot(w,t);
				beta = sigma / old_sigma;

				vec_p = -1.f * g + beta * vec_p;
				h = -1.f * t + beta * h;
				q = A * vec_p;
				l = C * h;

				gamma = my_dot(vec_p,q) + my_dot(h,l);
				++monitor;

				if (sigma < 0.f || gamma < curvature_tol)
				{
					printf("Curvature too small and method has broken down. 2 [%f]<[%f] [%f]<[%f] sigma = %f + %f.\n",sigma,0.f,gamma,curvature_tol,my_dot(r,g),my_dot(w,t));//MyPause;
				}
				printf("PCG_Residual_Update end iter (%d)\n",monitor.iteration_count());

				r_nrm2 = my_dot(r,g);
			}
			MyVector y = Recovering_y_from_x(A,B,P,x,c,d);
			std::cout << "x " << x.transpose() << std::endl;
			std::cout << "y " << y.transpose() << std::endl;
			return x;
		}

		MyVector PCG_Residual_Update(const MySpMat& A, const MySpMat& B, const MyVector& c, const MyVector& d,const MySpMat& P, MyMonitor<MyFloat>& monitor)
		{
			printf("PCG_Residual_Update start\n");
			const MyFloat curvature_tol = 10.e-5;

			const int n = A.rows();
			const int m = B.rows();
			Q_ASSERT(m<n);
			MyVector x(n),x_hat(n),r(n),g(n),vec_p(n),q(n);
			MyVector a(m),w(m),y_hat(m),v(m),t(m),h(m),l(m);
			a.setZero();
			w.setZero();

			x.setZero();//initial assumption x = 0;

			r.setZero(); w = d - B*x;
			MyGaussElimination::GaussElimination(P,n,m,x_hat,y_hat,r,w);

			x = x + x_hat;
			r = A*x + B.transpose()*y_hat-c;
			a.setZero();
			w.setZero();

			MyGaussElimination::GaussElimination(P,n,m,g,v,r,w);

			if (need_residual_update(g,v))
			{
				residual_update(a,g,r,v,w,B,P);
			}
			t = v+a;
			vec_p = -1.f*g;
			h = -1.f * t;
			q = A * vec_p;
			l.setZero();

			MyFloat sigma = my_dot(r,g) + my_dot(w,t);
			MyFloat old_sigma;
			MyFloat gamma = my_dot(vec_p,q) + my_dot(h,l);
			MyFloat alpha,beta;
			MyFloat r_nrm2 = my_dot(r,g);

			if (monitor.finished(r_nrm2))
			{
				return x;
			}
			
			if (sigma < 0.f || gamma < curvature_tol)
			{
				printf("Curvature too small and method has broken down.\n");MyPause;
			}

			while (!monitor.finished(r_nrm2))
			{
				alpha = sigma / gamma;
				x = x + alpha * vec_p;
				r = r + alpha * q;
				a = a + alpha * h;
				w = w + alpha * l;

				MyGaussElimination::GaussElimination(P,n,m,g,v,r,w);

				if (need_residual_update(g,v))
				{
					residual_update(a,g,r,v,w,B,P);
				}

				t = a + v;
				old_sigma = sigma;
				sigma = my_dot(r,g) + my_dot(w,t);
				beta = sigma / old_sigma;

				vec_p = -1.f * g + beta * vec_p;
				h = -1.f * t + beta * h;
				q = A * vec_p;
				l.setZero();

				gamma = my_dot(vec_p,q) + my_dot(h,l);
				++monitor;

				if (sigma < 0.f || gamma < curvature_tol)
				{
					printf("Curvature too small and method has broken down.\n");MyPause;
				}
				printf("PCG_Residual_Update end iter (%d)\n",monitor.iteration_count());

				r_nrm2 = my_dot(r,g);
			}
			
			return x;
#if 0

			typedef MyFloat  ValueType;
			assert(A.rows() == A.cols());        // sanity check
			assert(A.rows() == B.cols());
			const size_t N = A.rows();
			MyVector r(N),g(N),p(N);
			MyFloat alpha,beta;

			//1.start 
			x.setZero();
			r = A*x-c;
			g = P*r;
			p = -1.f*g;

			//2.loop

			PhysicsContext& currentCtx = FEM_State_Ctx;
			MyCuspVecView& y = currentCtx.cg_y; 
			MyCuspVecView& z = currentCtx.cg_z;
			MyCuspVecView& r = currentCtx.cg_r;
			MyCuspVecView& p = currentCtx.cg_p;
			y.resize(N);
			z.resize(N);
			r.resize(N);
			p.resize(N);	

			// y <- Ax
			cusp::multiply(A, x, y);

			// r <- b - A*x
			axpby(b, y, r, ValueType(1), ValueType(-1));

			// z <- M*r
			cusp::multiply(M, r, z);

			// p <- z
			copyAsync(thrust::raw_pointer_cast(&z[0]),thrust::raw_pointer_cast(&p[0]),N);
			//blas::copy(z, p);

			// rz = <r^H, z>
			ValueType rz = mydotc(r, z);

			ValueType r_nrm2 = nrm2(r);

			while (!monitor.finished(r_nrm2))
			{
				// y <- Ap
				cusp::multiply(A, p, y);

				// alpha <- <r,z>/<y,p>
				ValueType alpha =  rz / mydotc(y, p);

				// x <- x + alpha * p
				//blas::axpy(p, x, alpha);

				// r <- r - alpha * y		
				//blas::axpy(y, r, -alpha);

				axpy_axpy(p,x,alpha,y,r,-alpha);

				// z <- M*r
				cusp::multiply(M, r, z);

				ValueType rz_old = rz;

				// rz = <r^H, z>
				rz = mydotc(r, z);

				// beta <- <r_{i+1},r_{i+1}>/<r,r> 
				ValueType beta = rz / rz_old;

				// p <- r + beta*p
				axpby(z, p, p, ValueType(1), beta);

				++monitor;

				r_nrm2 = nrm2(r);
			}
#endif
		}
	}
}