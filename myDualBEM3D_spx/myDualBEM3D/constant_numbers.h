#ifndef NUMBERS_H
#define NUMBERS_H

#include "bemDefines.h"
#include <math.h>
#include <iostream>

namespace VR
{
	struct numbers
	{
		static const unsigned int invalid_unsigned_int = static_cast<unsigned int> (-1);
		static const MyFloat 	E;// = 2.7182818284590452354;
		static const MyFloat 	LOG2E;// = 1.4426950408889634074;
		static const MyFloat 	LOG10E;// = 0.43429448190325182765;
		static const MyFloat 	LN2;// = 0.69314718055994530942;
		static const MyFloat 	LN10;// = 2.30258509299404568402;
		static const MyFloat 	MyPI;// = 3.14159265358979323846;
		static const MyFloat 	PI_2;// = 1.57079632679489661923;
		static const MyFloat 	PI_4;// = 0.78539816339744830962;
		static const MyFloat 	SQRT2;// = 1.41421356237309504880;
		static const MyFloat 	SQRT1_2;// = 0.70710678118654752440;
		static const MyFloat    EPSINON;
		static bool isZero(MyFloat var)
		{
			if(var < EPSINON  && var > -EPSINON)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		static bool isEqual(const MyDenseVector& b,const MyDenseVector& e,MyFloat t=EPSINON)
		{
			return isEqual(b(0),e(0)) && isEqual(b(1),e(1)) && isEqual(b(2),e(2));
		}

		static bool isEqual(MyFloat var1,MyFloat var2,MyFloat t = EPSINON)
		{
			const MyFloat positive = fabs(t);
			const MyFloat negative = -1.0f * positive;

			if ( (var1 - var2) < positive && (var1 - var2) > negative )
			{
				return true;
			}
			else
			{
				return false;
			}
		}
#if 0
		static bool IsEqual(const MyVec3& b, const MyVec3& e, int absDelta = 1)
		{
			return IsEqual(b(0), e(0), absDelta) && IsEqual(b(1), e(1), absDelta) && IsEqual(b(2), e(2), absDelta);
		}

		static bool IsEqual(float f1, float f2, int absDelta = 1)
		{
			int i1, i2;

			i1 = ( f1>0)  ? ((int&)f1)  : ( (int&) f1 - 0x80000000 );

			i2 = (f2>0)  ? ((int&)f2)  : ( (int&) f2 - 0x80000000 );

			return   ((abs(i1-i2))<absDelta) ? true : false;
		}
#endif

		static bool isSymmetric(const MySpMat& obj)
		{

			for (int k=0; k<obj.outerSize(); ++k)
			{
				for (MySpMat::InnerIterator it(obj,k); it; ++it)
				{
					if (   !numbers::isEqual( it.value() , obj.coeff(it.col(),it.row()),0.00000001 )   )
					{
						printf("(%d,%d) (%f)----(%f)\n",it.row(),it.col(),it.value(),obj.coeff(it.col(),it.row()));
						return false;
					}
				}
			}
			return true;
		}

		static bool isSymmetric(const MyMatrix& obj)
		{
			for (unsigned r=0;r<obj.rows();++r)
			{
				for (unsigned c = r+1;c<obj.cols();++c)
				{
					if (   !numbers::isEqual( obj.coeff(r,c) , obj.coeff(c,r),0.00000001 )   )
					{
						printf("(%d,%d) (%f)----(%f)\n",r,c,obj.coeff(r,c),obj.coeff(c,r));
						std::cout << obj << std::endl;
						return false;
					}
				}
				
			}
			return true;
		}

		static MyFloat norm_sqr(const MyVector& vec)
		{
			MyFloat ret = 0.;
			for (int v=0;v < vec.rows();++v)
			{
				ret += vec(v) * vec(v);
			}
			return ret;
		}

		static MyFloat l2_norm(const MyVector& vec)
		{
			return std::sqrt(norm_sqr(vec));
		}

		template <class T>
		static short sign(T a)
		{
			return (a >= 0 ? 1 : -1);
		}

		static float logN(const int base, float value)
		{
			return (float)(log((double)value) / log ((double)base));
			//,log2 x=(log10 x)/(log10 2)
		}

		static float max3(	float a,float b,float c)
		{
			float d = (b>c)?b:c;
			return ((a>d)?a:d);
		}
	};
}//namespace VR_FEM

#endif // NUMBERS_H
