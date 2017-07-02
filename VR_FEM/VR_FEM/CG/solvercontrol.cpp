
#include "solvercontrol.h"
#include <cmath>
#include <iostream>
#include "../VR_MACRO.h"
namespace YC
{
    inline unsigned int
    SolverControl::max_steps () const
    {
      return maxsteps;
    }



    inline unsigned int
    SolverControl::set_max_steps (const unsigned int newval)
    {
      unsigned int old = maxsteps;
      maxsteps = newval;
      return old;
    }



    inline void
    SolverControl::set_failure_criterion (const MyFloat rel_failure_residual)
    {
      relative_failure_residual=rel_failure_residual;
      check_failure=true;
    }



    inline void
    SolverControl::clear_failure_criterion ()
    {
      relative_failure_residual=0;
      failure_residual=0;
      check_failure=false;
    }



    inline MyFloat
    SolverControl::tolerance () const
    {
      return tol;
    }



    inline MyFloat
    SolverControl::set_tolerance (const MyFloat t)
    {
      MyFloat old = tol;
      tol = t;
      return old;
    }


    inline void
    SolverControl::log_history (const bool newval)
    {
      m_log_history = newval;
    }



    inline bool
    SolverControl::log_history () const
    {
      return m_log_history;
    }



    inline void
    SolverControl::log_result (const bool newval)
    {
      m_log_result = newval;
    }



    inline bool
    SolverControl::log_result () const
    {
      return m_log_result;
    }

    SolverControl::NoConvergence::NoConvergence (const unsigned int last_step,
                             const MyFloat       last_residual)
            :
            last_step (last_step),
            last_residual (last_residual)
    {}


    const char *
    SolverControl::NoConvergence::what () const throw ()
    {
                       // have a place where to store the
                       // description of the exception as a char *
                       //
                       // this thing obviously is not multi-threading
                       // safe, but we don't care about that for now
                       //
                       // we need to make this object static, since
                       // we want to return the data stored in it
                       // and therefore need a liftime which is
                       // longer than the execution time of this
                       // function
      static std::string description;
                       // convert the messages printed by the
                       // exceptions into a std::string
      std::ostringstream out;
      out << "Iterative method reported convergence failure in step "
          << last_step << " with residual " << last_residual;

      description = out.str();
      return description.c_str();
    }



    SolverControl::SolverControl (const unsigned int maxiter,
                      const MyFloat tolerance,
                      const bool m_log_history,
                      const bool m_log_result)
            :
            maxsteps(maxiter),
            tol(tolerance),
            lvalue(1.e300),
            lstep(0),
            check_failure(false),
            relative_failure_residual(0),
            failure_residual(0),
            m_log_history(m_log_history),
            m_log_frequency(1),
            m_log_result(m_log_result),
            history_data_enabled(false)
    {}



    SolverControl::~SolverControl()
    {}



    SolverControl::State
    SolverControl::check (const unsigned int step,
                  const MyFloat check_value)
    {
                       // if this is the first time we
                       // come here, then store the
                       // residual for later comparisons
      if (step==0)
        {
          initial_val = check_value;
          if (history_data_enabled)
        history_data.resize(maxsteps);
        }

      if (true ||m_log_history && ((step % m_log_frequency) == 0))
		  std::cout << "Check " << step << "\t" << check_value << std::endl;

      lstep  = step;
      lvalue = check_value;

      if (step==0)
        {
          if (check_failure)
        failure_residual=relative_failure_residual*check_value;

          if (m_log_result)
			  std::cout << "Starting value " << check_value << std::endl;
        }

      if (history_data_enabled)
        history_data[step] = check_value;

      if (check_value <= tol)
        {
          if (m_log_result)
			  std::cout << "Convergence step " << step << " value " << check_value << std::endl;
          lcheck = success;
          return success;
        }
	  else
	  {
		  printf("check_value(%f) tol(%f)\n",check_value,tol);
	  }

      if ((step >= maxsteps) ||
    #ifdef HAVE_ISNAN
          isnan(check_value) ||
    #else
    #  if HAVE_UNDERSCORE_ISNAN
                           // on Microsoft Windows, the
                           // function is called _isnan
          _isnan(check_value) ||
    #  endif
    #endif
          (check_failure && (check_value > failure_residual))
      )
        {
          if (m_log_result)
			  std::cout << "Failure step " << step << " value " << check_value << std::endl;
          lcheck = failure;
          return failure;
        }

      lcheck = iterate;
      return iterate;
    }



    SolverControl::State
    SolverControl::last_check() const
    {
      return lcheck;
    }


    MyFloat
    SolverControl::initial_value() const
    {
      return initial_val;
    }


    MyFloat
    SolverControl::last_value() const
    {
      return lvalue;
    }


    unsigned int
    SolverControl::last_step() const
    {
      return lstep;
    }


    unsigned int
    SolverControl::log_frequency (unsigned int f)
    {
      if (f==0)
        f = 1;
      unsigned int old = m_log_frequency;
      m_log_frequency = f;
      return old;
    }


    void
    SolverControl::enable_history_data ()
    {
      history_data_enabled = true;
    }


    MyFloat
    SolverControl::average_reduction() const
    {
      if (lstep == 0)
        return 0.;

      Q_ASSERT (history_data_enabled);
      Q_ASSERT (history_data.size() > lstep);
      Q_ASSERT (history_data[0] > 0.);
      Q_ASSERT (history_data[lstep] > 0.);

      return std::pow((MyFloat)history_data[lstep]/history_data[0], (MyFloat)1./lstep);
    }



    MyFloat
    SolverControl::step_reduction(unsigned int step) const
    {
      Q_ASSERT (history_data_enabled);
      Q_ASSERT (history_data.size() > lstep);
      Q_ASSERT (step <=lstep);
      Q_ASSERT (step>0);

      return history_data[step]/history_data[step-1];
    }


    MyFloat
    SolverControl::final_reduction() const
    {
      return step_reduction(lstep);
    }
}//NAMESPACE
