#include "bemMatlabFunc.h"

#include "bemDefines.h"
#include "bemMarco.h"
#include <set>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>     // std::cout
#include <algorithm>    // std::set_symmetric_difference, std::sort
#include <vector>       // std::vector
#include "vrBase/vrIterator.h"
#include "constant_numbers.h"
#include <cmath>
namespace VR
{
	namespace MatlabFunc
	{
		const MyMatrix& zeros(MyInt nRow, MyInt nCol)
		{
			static MyMatrix mat;
			mat.setZero(nRow, nCol);
			return mat;
		}

		const MyIntMatrix& zerosInt(MyInt nRow, MyInt nCol)
		{
			static MyIntMatrix mat;
			mat.setZero(nRow, nCol);
			return mat;
		}

		/*MyFloat max(const MyVector& vec)
		{
			return vec.colwise().maxCoeff()[0];
		}

		MyInt max(const MyIntVector& vec)
		{
			return vec.colwise().maxCoeff()[0];
		}*/

		MyInt size(const MyMatrix& mat, const MyInt idx)
		{
			if (1 == idx)
			{
				return mat.rows();
			}
			else if (2 == idx)
			{
				return mat.cols();
			}
			else
			{
				MyError("MyInt size(const MyMatrix& mat, const MyInt idx)");
			}
		}

		const MyMatrix& ones(const MyInt nRow, const MyInt nCol)
		{
			static MyMatrix mat;
			mat.setOnes(nRow, nCol);
			return mat;
		}

		void MatrixAttach(MyMatrix& dstMat, const MyMatrix& attachMat)
		{
			Q_ASSERT(dstMat.rows() == attachMat.rows());

			static MyMatrix tmp;
			tmp.resize(dstMat.rows(), dstMat.cols() + attachMat.cols());
			tmp.block(0, 0, dstMat.rows(), dstMat.cols()) = dstMat;
			tmp.block(0, dstMat.cols(), attachMat.rows(), attachMat.cols()) = attachMat;

			dstMat = tmp;
		}

		const MyVector unique_float(const MyVector& vec)
		{
			std::set< MyFloat > tmpSet;
			for (int i = 0; i < vec.size();++i)
			{
				tmpSet.insert(vec[i]);
			}

			std::vector< MyFloat > tmpVec;
			MyVector retVec(tmpSet.size());

			for (iterAllOf(itr,tmpSet))
			{
				tmpVec.push_back(*itr);
			}

			std::sort(tmpVec.begin(),tmpVec.end());

			for (int i = 0; i < tmpVec.size();++i)
			{
				retVec[i] = tmpVec[i];
			}

			return retVec;
		}

		const MyIntVector unique_int(const MyIntVector& vec)
		{
			std::set< MyInt > tmpSet;
			for (int i = 0; i < vec.size(); ++i)
			{
				tmpSet.insert(vec[i]);
			}

			std::vector< MyInt > tmpVec;
			MyIntVector retVec(tmpSet.size());

			for (iterAllOf(itr, tmpSet))
			{
				tmpVec.push_back(*itr); 
			}

			std::sort(tmpVec.begin(), tmpVec.end());

			for (int i = 0; i < tmpVec.size(); ++i)
			{
				retVec[i] = tmpVec[i];
			}

			return retVec;
		}

		const MyInt length(const MyIntVector& vec)
		{
			return vec.size();
		}

		const MyInt length(const MyVector& vec)
		{
			return vec.size();
		}

		const MyInt length(const MyMatrix& mat)
		{
			return mat.rows();
		}

		const MyInt length(const MyIntMatrix& mat)
		{
			return mat.rows();
		}

		const MyMatrix MergeVector(const MyVector& col0, const MyVector& col1, const MyVector& col2)
		{
			Q_ASSERT(col0.size() == col1.size());
			Q_ASSERT(col2.size() == col1.size());
			static MyMatrix retMat;
			retMat.resize(col0.size(), 3);
			retMat.col(0) = col0;
			retMat.col(1) = col1;
			retMat.col(2) = col2;
			return retMat;
		}

		const MyVector leftDotMult(const MyVector& leftVec, const MyVector& rightVec)
		{
			return leftVec.array() * rightVec.array();
		}

		const MyVector subVector(const MyVector& srcVec, const MyVec2I& Range)
		{
			const MyInt nSubSize = Range[1] - Range[0] + 1;
			MyVector retVec(nSubSize);

			for (int i = 0; i < nSubSize;++i)
			{
				retVec[i] = srcVec[i + Range[0]];
			}

			return retVec;
		}

		bool isequal(const MyMatrix& leftMat, const MyMatrix& rightMat)
		{
			Q_ASSERT(leftMat.rows() == rightMat.rows());
			Q_ASSERT(leftMat.cols() == rightMat.cols());

			bool retVal = true;
			const MyInt nRow = leftMat.rows();
			const MyInt nCol = leftMat.cols();
			for (int r = 0; r < nRow && retVal; ++r)
			{
				for (int c = 0; c < nCol && retVal; ++c)
				{
					retVal = numbers::isEqual(leftMat.coeff(r, c), rightMat.coeff(r, c));
				}
			}
			return retVal;
		}

		const MyVector nonzeros(const MyVector& vec)
		{
			static MyVector retVec;
			retVec.setZero(vec.size());
			int nonzeroIdx = 0;
			for (int i = 0; i < vec.size();++i)
			{
				if (!numbers::isZero(vec[i]))
				{
					retVec[nonzeroIdx++] = vec[i];
				}
			}
			retVec.conservativeResize(nonzeroIdx);
			return retVec;
		}

		const MyIntVector makeRange(const MyInt b, const MyInt e)
		{
			MyIntVector retVec(e - b + 1);
			for (MyInt i = b, retIdx = 0; i <= e; ++i, ++retIdx)
			{
				retVec[retIdx] = i;
			}
			return retVec;
		}

		/*const MyVector linspace(const MyFloat b, const MyFloat e, const MyInt nSize)
		{
			MyVector retVec(nSize);
			const MyFloat step = (e - b) / nSize;
			for (int i = 0; i < nSize; ++i )
			{
				retVec[i] = b + i * step;
			}
			return retVec;
		}*/

		const MyVector linspace(const MyFloat min, const MyFloat max, const MyInt n)//vector<double> linspace(double min, double max, int n)
		{
			MyVector result(n); 
			// vector iterator
			int iterator = 0;

			for (int i = 0; i <= (n - 2); i++)
			{
				double temp = min + i*(max - min) / (floor((double)n) - 1);
				result[iterator] = temp;
				//result.insert(result.begin() + iterator, temp);
				iterator += 1;
			}

			//iterator += 1;
			result[iterator] = max;
			//result.insert(result.begin() + iterator, max);
			return result;
		}

		const MyIntVector& joinVector(const MyIntVector& vec1, const MyIntVector& vec2)
		{
			static MyIntVector sRetVec;
			sRetVec.resize(vec1.size() + vec2.size());
			sRetVec.block(0, 0, vec1.size(), 1) = vec1;
			sRetVec.block(vec1.size(), 0, vec2.size(), 1) = vec2;
			return sRetVec;
		}

		const MyIntVector& setxorIntVector(const MyIntVector& vec1, const MyIntVector& vec2)
		{
			/*static MyIntVector sRetIntVec;
			return sRetIntVec;*/
			std::vector< MyInt > retVec(vec1.size() + vec2.size());
			std::vector<MyInt>::iterator it = std::set_symmetric_difference(
				vec1.data(), vec1.data() + vec1.size(), 
				vec2.data(), vec2.data() + vec2.size(),
				retVec.begin());

			retVec.resize(it-retVec.begin());

			static MyIntVector sRetIntVec;
			sRetIntVec.resize(retVec.size());
			//MyIntVector tmpVec(retVec.size());
			for (int i = 0; i < retVec.size();++i)
			{
				sRetIntVec[i] = retVec[i];
			}
			return sRetIntVec;
		}

		MyFloat NthRoot(MyFloat value, MyInt degree)
		{
			MyError("NthRoot");
			/*if (3 == degree)
			{
				return std::cbrt(value);
			}
			else
			{
				Q_ASSERT(value > 0);
				return pow(value, (MyFloat)(1.0 / degree));
			}*/
			
		};

		
	}
}