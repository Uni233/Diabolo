#ifndef _tbb_parallel_assemble_source_point_h_
#define _tbb_parallel_assemble_source_point_h_

#include "bemMarco.h"
#if USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/tbb_allocator.h>
#endif

#include <boost/atomic.hpp>
#include <iostream>
extern boost::atomic<int> g_atomix_count;

namespace VR
{
	class vrBEM3D;
	namespace TBB
	{
		

#if USE_DUAL

		class AssembleSystem_DualEquation
		{
		public:
			void operator()(const tbb::blocked_range<size_t> &r) const
			{

				for (size_t i = r.begin(); i != r.end(); ++i)
				{
					const int v = i / (MyDim*MyDim);
					const int idx_i = (i % (MyDim*MyDim)) / MyDim;
					const int idx_j = (i % (MyDim*MyDim)) % MyDim;
					//m_vrBEM3D_ptr->AssembleSystem_DisContinuous_DualEquation_Aliabadi(v,idx_i,idx_j);
					m_vrBEM3D_ptr->AssembleSystem_DisContinuous_DualEquation_Aliabadi_Nouse(v,idx_i,idx_j);
					//m_vrBEM3D_ptr->AssembleSystem_DisContinuous_DualEquation_Aliabadi_Peng(v,idx_i,idx_j);
					
					++g_atomix_count;
					std::cout << "Processing " << g_atomix_count << " of " << nSize << std::endl;
				}
			};

			AssembleSystem_DualEquation(vrBEM3D* vrBEM3D_ptr, const vrInt _size):m_vrBEM3D_ptr(vrBEM3D_ptr), nSize(_size)
			{}
			vrBEM3D* m_vrBEM3D_ptr;
			int nSize;
		};
#endif//USE_DUAL
	}
}
#endif//_tbb_parallel_assemble_source_point_h_