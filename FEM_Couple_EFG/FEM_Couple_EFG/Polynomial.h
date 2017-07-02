#ifndef _Polynomial_h
#define _Polynomial_h
#include "VR_Global_Define.h"

namespace VR_FEM
{
	class Polynomial
	{
	public:
		Polynomial(unsigned int degree, unsigned int support);
		static std::vector<Polynomial > generate_complete_basis (const unsigned int degree/*1*/);
		void value (const MyFloat x, std::vector<MyFloat> &values) const;
	private:
		void compute_coefficients (const unsigned int n/*1*/, const unsigned int support_point/*0,1*/);

		std::vector<MyFloat> m_coefficient;
	};
}
#endif//_Polynomial_h