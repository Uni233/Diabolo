#ifndef _PCG_with_residual_update_Monitor_H_
#define _PCG_with_residual_update_Monitor_H_

#include <boost/math/tools/precision.hpp>

namespace YC
{
	namespace PCG_RU
	{

		template <typename Real>
		class MyMonitor
		{
		public:
			/*! Construct a \p default_monitor for a given right-hand-side \p b
			*
			*  The \p default_monitor terminates iteration when the residual norm
			*  satisfies the condition
			*       ||b - A x|| <= absolute_tolerance + relative_tolerance * ||b||
			*  or when the iteration limit is reached.
			*
			*  \param b right-hand-side of the linear system A x = b
			*  \param iteration_limit maximum number of solver iterations to allow
			*  \param relative_tolerance determines convergence criteria
			*  \param absolute_tolerance determines convergence criteria
			*
			*  \tparam VectorType vector
			*/
			//template <typename Vector>
			MyMonitor(const Real& bNorm, size_t iteration_limit = 500, Real relative_tolerance = 1e-5, Real absolute_tolerance = 0)
				: b_norm(bNorm),
				r_norm(boost::math::tools::max_value<Real>()),
				iteration_limit_(iteration_limit),
				iteration_count_(0),
				relative_tolerance_(relative_tolerance),
				absolute_tolerance_(absolute_tolerance)
			{}

			/*! increment the iteration count
			*/
			void operator++(void) {  ++iteration_count_; } // prefix increment

			/*! applies convergence criteria to determine whether iteration is finished
			*
			*  \param r residual vector of the linear system (r = b - A x)
			*  \tparam Vector vector
			*/
			//template <typename Vector>
			bool finished(const Real& r)
			{
				r_norm = r;//cusp::blas::nrm2(r);

				if (iteration_count() >= iteration_limit())
				{
					printf("[#####]iteration_count() >= iteration_limit();\n");
					return true;
				}
				return converged() || iteration_count() >= iteration_limit();
			}

			/*! whether the last tested residual satifies the convergence tolerance
			*/
			bool converged() const
			{
				printf("[%f] <= [%f]\n",residual_norm(),tolerance());
				return residual_norm() <= tolerance();
			}

			/*! Euclidean norm of last residual
			*/
			Real residual_norm() const { return r_norm; }

			/*! number of iterations
			*/
			size_t iteration_count() const { return iteration_count_; }

			/*! maximum number of iterations
			*/
			size_t iteration_limit() const { return iteration_limit_; }

			/*! relative tolerance
			*/
			Real relative_tolerance() const { return relative_tolerance_; }

			/*! absolute tolerance
			*/
			Real absolute_tolerance() const { return absolute_tolerance_; }

			/*! tolerance
			*
			*  Equal to absolute_tolerance() + relative_tolerance() * ||b||
			*
			*/ 
			Real tolerance() const { return absolute_tolerance() + relative_tolerance() * b_norm; }

		protected:

			Real r_norm;
			Real b_norm;
			Real relative_tolerance_;
			Real absolute_tolerance_;

			size_t iteration_limit_;
			size_t iteration_count_;
		};
	}
}
#endif//_PCG_with_residual_update_Monitor_H_