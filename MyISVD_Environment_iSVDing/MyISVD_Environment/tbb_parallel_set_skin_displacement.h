#ifndef _tbb_parallel_set_skin_displacement_h_
#define _tbb_parallel_set_skin_displacement_h_

#include "VR_Global_Define.h"
#if USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/tbb_allocator.h>
#endif

namespace YC
{
	class ApplySkinDisplacement
	{
		YC::MyMatrix * matU_ptr;
		YC::MyMatrix * matV_ptr;
		YC::MyVector *  vecDisp_ptr;
		YC::MyIntMatrix *  matDofs_ptr;
	public:
		void operator()(const tbb::blocked_range<size_t> &r) const
		{
			YC::MyMatrix& matV = *matV_ptr;
			YC::MyMatrix& matU = *matU_ptr;
			YC::MyVector& incremental_displacement = *vecDisp_ptr;
			YC::MyIntMatrix& matV_dofs = *matDofs_ptr;
			/*for (size_t i = r.begin(); i != r.end(); ++i)
			{
				refMat.row(i) = Colors::gold;
			}*/

			for (size_t v = r.begin(); v != r.end(); ++v) //for (int v = 0; v < nRows; v++)
			{
				const MyVectorI& dofs = matV_dofs.row(v);
				const MyVector& pos = matV.row(v);
				matU.row(v) = pos + MyDenseVector(incremental_displacement[dofs.x()],
					incremental_displacement[dofs.y()],
					incremental_displacement[dofs.z()]
					);
			}
		}

		ApplySkinDisplacement(YC::MyMatrix * matV, YC::MyMatrix * matU, YC::MyVector *  vecDisp, YC::MyIntMatrix *  matDofs)
			:matV_ptr(matV), matU_ptr(matU), vecDisp_ptr(vecDisp), matDofs_ptr(matDofs)
		{}

	};
}

#endif//_tbb_parallel_set_skin_displacement_h_