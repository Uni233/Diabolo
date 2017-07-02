#ifndef _tbb_parallel_set_color_h_
#define _tbb_parallel_set_color_h_

#include "VR_Global_Define.h"
#if USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/tbb_allocator.h>
#endif

namespace YC
{
	class ApplyColor
	{
		YC::MyMatrix * matPtr;
		YC::MyInt colorId;
	public:
		void operator()(const tbb::blocked_range<size_t> &r) const
		{
			YC::MyMatrix& refMat = *matPtr;
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				refMat.row(i) = MyDenseVector(Colors::colorTemplage[colorId][0], Colors::colorTemplage[colorId][1], Colors::colorTemplage[colorId][2]);
			}
		}

		ApplyColor(YC::MyMatrix * matPoint, const int colorId_)
			:matPtr(matPoint), colorId(colorId_)
		{}

	};

	class ApplyColorMultiDomain
	{
		YC::MyMatrix * matPtr;
		YC::MyIntVector * vecPtr;

	public:
		void operator()(const tbb::blocked_range<size_t> &r) const
		{
			YC::MyMatrix& refMat = *matPtr;
			YC::MyIntVector& refVec = *vecPtr;
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				int colorId = refVec[i];
				refMat.row(i) = MyDenseVector(Colors::colorTemplage[colorId][0], Colors::colorTemplage[colorId][1], Colors::colorTemplage[colorId][2]);
			}
		}

		ApplyColorMultiDomain(YC::MyMatrix * matPoint, YC::MyIntVector * colorId_)
			:matPtr(matPoint), vecPtr(colorId_)
		{}

	};
}

#endif//_tbb_parallel_set_color_h_