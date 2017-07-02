#include "Polynomial.h"

namespace YC
{
	Polynomial::Polynomial(unsigned int degree, unsigned int support)
	{
		m_coefficient.resize (degree+1);
		compute_coefficients(degree,support);
	}

	std::vector< Polynomial > Polynomial::generate_complete_basis (const unsigned int degree/*1*/)
	{
		std::vector<Polynomial> v;
		for (unsigned int i=0; i<=degree; ++i)
		{
			v.push_back(Polynomial(degree,i));
		}
		return v;
	}

	void Polynomial::value (const MyFloat x, std::vector<MyFloat> &values) const
	{
		const unsigned int values_size=values.size();

		//printf("YANGCHENDEBUG 3-12 Polynomial<number>::value x(%f) values_size(%d)\n",x,values_size);
		//	YANGCHENDEBUG 3-12 Polynomial<number>::value x(0.211325) values_size(2)


		const unsigned int m=m_coefficient.size();
		std::vector<MyFloat> a(m_coefficient);

		for (unsigned int v=0;v<a.size();++v)
		{
			//printf("3-12 coefficients[%d]=%f\n",v,a[v]);
			/*support == 0
			3-12 coefficients[0]=1.000000
			3-12 coefficients[1]=-1.000000
			*/
			/*support == 1
			3-12 coefficients[0]=0.000000
			3-12 coefficients[1]=-1.000000
			*/
		}

		unsigned int j_faculty=1;

		// loop over all requested
		// derivatives. note that
		// derivatives @p{j>m} are
		// necessarily zero, as they
		// differentiate the polynomial
		// more often than the highest
		// power is
		const unsigned int min_valuessize_m/*2*/=(values_size < m ? values_size : m);//std::min(values_size, m);
		for (unsigned int j=0; j<min_valuessize_m; ++j)
		{
			for (int k=m-2; k>=static_cast<int>(j); --k)
			{
				a[k]+=x*a[k+1];
			}
			values[j]=static_cast<MyFloat>(j_faculty)*a[j];

			j_faculty*=j+1;
		}
		/*
		YANGCHENDEBUG 3-12 Polynomial<number>::value x(0.211325) values_size(2)
		3-12 coefficients[0]=1.000000
		3-12 coefficients[1]=-1.000000
		3-12 min_valuessize_m is 2
		3-12 a[0]=0.788675 a[1]=-1.000000
		3-12 j_faculty is 1
		3-12 j_faculty is 2
		3-12 values[0]=0.788675
		3-12 values[1]=-1.000000
		YANGCHENDEBUG 3-12 Polynomial<number>::value x(0.211325) values_size(2)
		3-12 coefficients[0]=0.000000
		3-12 coefficients[1]=1.000000
		3-12 min_valuessize_m is 2
		3-12 a[0]=0.211325 a[1]=1.000000
		3-12 j_faculty is 1
		3-12 j_faculty is 2
		3-12 values[0]=0.211325
		3-12 values[1]=1.000000
		*/
		/*
		YANGCHENDEBUG 3-12 Polynomial<number>::value x(0.788675) values_size(2)
		3-12 coefficients[0]=1.000000
		3-12 coefficients[1]=-1.000000
		3-12 min_valuessize_m is 2
		3-12 a[0]=0.211325 a[1]=-1.000000
		3-12 j_faculty is 1
		3-12 j_faculty is 2
		3-12 values[0]=0.211325
		3-12 values[1]=-1.000000
		YANGCHENDEBUG 3-12 Polynomial<number>::value x(0.788675) values_size(2)
		3-12 coefficients[0]=0.000000
		3-12 coefficients[1]=1.000000
		3-12 min_valuessize_m is 2
		3-12 a[0]=0.788675 a[1]=1.000000
		3-12 j_faculty is 1
		3-12 j_faculty is 2
		3-12 values[0]=0.788675
		3-12 values[1]=1.000000
		*/

		// fill higher derivatives by zero
		for (unsigned int j=min_valuessize_m; j<values_size; ++j)
			values[j] = 0;

		for (unsigned int j=0;j<values_size;++j)
		{
			//printf("3-12 values[%d]=%f\n",j,values[j]);
			/*support == 0
			3-12 values[0]=0.788675
			3-12 values[1]=-1.000000
			*/
			/*support == 1
			3-12 values[0]=0.211325
			3-12 values[1]=1.000000
			*/
		}
	}

	void Polynomial::compute_coefficients (const unsigned int n/*1*/, const unsigned int support_point/*0,1*/)
	{
		unsigned int n_functions=n+1;/*2*/
		MyFloat const *x=0;

		static const MyFloat x1[4]={1.0, -1.0, 0.0, 1.0};
		x=&x1[0];

		for (unsigned int i=0; i<n_functions; ++i)
		{
			m_coefficient[i]=x[support_point*n_functions+i];
		}
	}
}