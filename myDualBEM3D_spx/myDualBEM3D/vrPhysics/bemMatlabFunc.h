#ifndef _bemMatlabFunc_h_
#define _bemMatlabFunc_h_
#include "bemDefines.h"
#include "bemMarco.h"
namespace VR
{
	namespace MatlabFunc
	{
		const MyMatrix& zeros(MyInt nRow, MyInt nCol);

		const MyIntMatrix& zerosInt(MyInt nRow, MyInt nCol);

		/*MyFloat max(const MyVector& vec);		
		MyInt max(const MyIntVector& vec);*/
		
		template< class T , class T2>
		T2 max(T vec, T2 )
		{
			return vec.colwise().maxCoeff()[0];
		}
		

		MyInt size(const MyMatrix& mat, const MyInt idx);

		const MyMatrix& ones(const MyInt nRow, const MyInt nCol);

		void MatrixAttach(MyMatrix& dstMat, const MyMatrix& attachMat);

		const MyVector unique_float(const MyVector& vec);
		const MyIntVector unique_int(const MyIntVector& vec);

		//const MyVector& reshape(const MyVector& vec, const MyInt nRows, const MyInt nCols);
		template< class T >
		const T reshape(const T& srcObj, const MyInt nRows, const MyInt nCols)
		{
			const MyInt nSize = nRows * nCols;
			Q_ASSERT(srcObj.size() == (nSize));
			T retMat(nRows, nCols);
			retMat.setZero();
			for (int i = 0; i < nSize;++i)
			{
				retMat.data()[i] = srcObj.data()[i];
			}
			return retMat;
		}

		template< class T >
		const MyIntVector reshapeIntVector(const T& srcObj, const MyInt nSize)
		{
			Q_ASSERT(srcObj.size() == (nSize));
			MyIntVector retIntVec(nSize);
			retIntVec.setZero();
			for (int i = 0; i < nSize; ++i)
			{
				retIntVec[i] = srcObj.data()[i];
			}
			return retIntVec;
		}

		const MyInt length(const MyIntVector& vec);
		const MyInt length(const MyVector& vec);
		const MyInt length(const MyMatrix& mat);
		const MyInt length(const MyIntMatrix& mat);


		const MyMatrix MergeVector(const MyVector& col0, const MyVector& col1, const MyVector& col2);

		const MyVector leftDotMult(const MyVector& leftVec, const MyVector& rightVec);

		const MyVector subVector(const MyVector& srcVec, const MyVec2I& Range);

		template<class T>
		const T subMatrixCols(const T& srcMat, const MyIntVector& Range)
		{
			T retMat;
			retMat.resize(srcMat.rows(), Range.size());
			retMat.setZero();
			for (int i = 0; i < Range.size();++i)
			{
				retMat.col(i) = srcMat.col(Range[i]);
			}
			return retMat;
		}

		template<class T>
		const T subMatrixRows(const T& srcMat, const MyIntVector& Range)
		{
			T retMat;
			retMat.resize(Range.size(), srcMat.cols());
			retMat.setZero();
			for (int i = 0; i < Range.size(); ++i)
			{
				retMat.row(i) = srcMat.row(Range[i]);
			}
			return retMat;
		}

		bool isequal(const MyMatrix& leftMat, const MyMatrix& rightMat);

		const MyVector nonzeros(const MyVector& vec);

		const MyIntVector makeRange(const MyInt b, const MyInt e);

		const MyVector linspace(const MyFloat b, const MyFloat e, const MyInt nSize);

		template<class T>
		MyInt numel(T obj)
		{
			return obj.size();
		}

		/*template< class T >
		const T& joinVector(const T& vec1, const T& vec2)
		{
		static T sRetVec;
		sRetVec.resize(vec1.size() + vec2.size());
		sRetVec.block(0, 0, vec1.size(), 1) = vec1;
		sRetVec.block(vec1.size(), 0, vec2.size(), 1) = vec2;
		return sRetVec;
		}*/

		
		const MyIntVector& joinVector(const MyIntVector& vec1, const MyIntVector& vec2);

		const MyIntVector& setxorIntVector(const MyIntVector& vec1, const MyIntVector& vec2);

		template< class T >
		T sign(T val)
		{
			if (val > 0)
			{
				return T(1);
			}
			else if (val < 0)
			{
				return T(-1);
			}
			else
			{
				return T(0);
			}
		}

		MyFloat NthRoot(MyFloat value, MyInt degree);

	}
}

#endif//_bemMatlabFunc_h_