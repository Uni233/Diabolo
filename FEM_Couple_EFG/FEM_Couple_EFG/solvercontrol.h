#ifndef SOLVERCONTROL_H
#define SOLVERCONTROL_H

#include <vector>
#include <string>
#include <sstream>
#include "VR_Global_Define.h"


namespace VR_FEM
{
    class SolverControl
    {
        public:

                             /**
                              * Enum denoting the different
                              * states a solver can be in. See
                              * the general documentation of
                              * this class for more
                              * information.
                              */
            enum State {
                               /// Continue iteration
              iterate = 0,
                               /// Stop iteration, goal reached
              success,
                               /// Stop iteration, goal not reached
              failure
            };

                             /**
                              * Class to be thrown upon
                              * failing convergence of an
                              * iterative solver, when either
                              * the number of iterations
                              * exceeds the limit or the
                              * residual fails to reach the
                              * desired limit, e.g. in the
                              * case of a break-down.
                              *
                              * The residual in the last
                              * iteration, as well as the
                              * iteration number of the last
                              * step are stored in this object
                              * and can be recovered upon
                              * catching an exception of this
                              * class.
                              */
            class NoConvergence : public std::exception
            {
              public:
                             /**
                              * Constructor.
                              */
            NoConvergence (const unsigned int last_step,
                       const MyFloat       last_residual);

                             /**
                              * Standardized output for
                              * catch handlers.
                              */
            virtual const char * what () const throw ();

                             /**
                              * Iteration number of the
                              * last step.
                              */
            const unsigned int last_step;

                             /**
                              * Residual in the last step.
                              */
            const MyFloat       last_residual;
            };


                             /**
                              * Constructor. The parameters
                              * @p n and @p tol are the
                              * maximum number of iteration
                              * steps before failure and the
                              * tolerance to determine success
                              * of the iteration.
                              *
                              * @p log_history specifies
                              * whether the history (i.e. the
                              * value to be checked and the
                              * number of the iteration step)
                              * shall be printed to
                              * @p deallog stream.  Default
                              * is: do not print. Similarly,
                              * @p log_result specifies the
                              * whether the final result is
                              * logged to @p deallog. Default
                              * is yes.
                              */
            SolverControl (const unsigned int n           = 100,
                   const MyFloat       tol         = 1.e-10,
                   const bool         log_history = false,
                   const bool         log_result  = true);

                             /**
                              * Virtual destructor is needed
                              * as there are virtual functions
                              * in this class.
                              */
            virtual ~SolverControl();

                             /**
                              * Interface to parameter file.
                              */
            //static void declare_parameters (ParameterHandler& param);

                             /**
                              * Read parameters from file.
                              */
            //void parse_parameters (ParameterHandler& param);

                             /**
                              * Decide about success or failure
                              * of an iteration.  This function
                              * gets the current iteration step
                              * to determine, whether the
                              * allowed number of steps has
                              * been exceeded and returns
                              * @p failure in this case. If
                              * @p check_value is below the
                              * prescribed tolerance, it
                              * returns @p success. In all
                              * other cases @p iterate is
                              * returned to suggest
                              * continuation of the iterative
                              * procedure.
                              *
                              * The iteration is also aborted
                              * if the residual becomes a
                              * denormalized value
                              * (@p NaN). Note, however, that
                              * this check is only performed
                              * if the @p isnan function is
                              * provided by the operating
                              * system, which is not always
                              * true. The @p configure
                              * scripts checks for this and
                              * sets the flag @p HAVE_ISNAN
                              * in the file
                              * <tt>Make.global_options</tt> if
                              * this function was found.
                              *
                              * <tt>check()</tt> additionally
                              * preserves @p step and
                              * @p check_value. These
                              * values are accessible by
                              * <tt>last_value()</tt> and
                              * <tt>last_step()</tt>.
                              *
                              * Derived classes may overload
                              * this function, e.g. to log the
                              * convergence indicators
                              * (@p check_value) or to do
                              * other computations.
                              */
            virtual State check (const unsigned int step,
                     const MyFloat   check_value);

                             /**
                              * Return the result of the last check operation.
                              */
            State last_check() const;

                             /**
                              * Return the initial convergence
                              * criterion.
                              */
            MyFloat initial_value() const;

                             /**
                              * Return the convergence value of last
                              * iteration step for which @p check was
                              * called by the solver.
                              */
            MyFloat last_value() const;

                             /**
                              * Number of last iteration step.
                              */
            unsigned int last_step() const;

                             /**
                              * Maximum number of steps.
                              */
            unsigned int max_steps () const;

                             /**
                              * Change maximum number of steps.
                              */
            unsigned int set_max_steps (const unsigned int);

                             /**
                              * Enables the failure
                              * check. Solving is stopped with
                              * @p ReturnState @p failure if
                              * <tt>residual>failure_residual</tt> with
                              * <tt>failure_residual:=rel_failure_residual*first_residual</tt>.
                              */
            void set_failure_criterion (const MyFloat rel_failure_residual);

                             /**
                              * Disables failure check and
                              * resets
                              * @p relative_failure_residual
                              * and @p failure_residual to
                              * zero.
                              */
            void clear_failure_criterion ();

                             /**
                              * Tolerance.
                              */
            MyFloat tolerance () const;

                             /**
                              * Change tolerance.
                              */
            MyFloat set_tolerance (const MyFloat);

                             /**
                              * Enables writing residuals of
                              * each step into a vector for
                              * later analysis.
                              */
            void enable_history_data();

                             /**
                              * Average error reduction over
                              * all steps.
                              *
                              * Requires
                              * enable_history_data()
                              */
            MyFloat average_reduction() const;
                             /**
                              * Error reduction of the last
                              * step; for stationary
                              * iterations, this approximates
                              * the norm of the iteration
                              * matrix.
                              *
                              * Requires
                              * enable_history_data()
                              */
            MyFloat final_reduction() const;

                             /**
                              * Error reduction of any
                              * iteration step.
                              *
                              * Requires
                              * enable_history_data()
                              */
            MyFloat step_reduction(unsigned int step) const;

                             /**
                              * Log each iteration step. Use
                              * @p log_frequency for skipping
                              * steps.
                              */
            void log_history (const bool);

                             /**
                              * Returns the @p log_history flag.
                              */
            bool log_history () const;

                             /**
                              * Set logging frequency.
                              */
            unsigned int log_frequency (unsigned int);

                             /**
                              * Log start and end step.
                              */
            void log_result (const bool);

                             /**
                              * Returns the @p log_result flag.
                              */
            bool log_result () const;

                             /**
                              * This exception is thrown if a
                              * function operating on the
                              * vector of history data of a
                              * SolverControl object id
                              * called, but storage of history
                              * data was not enabled by
                              * enable_history_data().
                              */
            //DeclException0(ExcHistoryDataRequired);

          protected:
                             /**
                              * Maximum number of steps.
                              */
            unsigned int maxsteps;

                             /**
                              * Prescribed tolerance to be achieved.
                              */
            MyFloat       tol;

                             /**
                              * Result of last check operation.
                              */
            State        lcheck;

                             /**
                              * Initial value.
                              */
            MyFloat initial_val;

                             /**
                              * Last value of the convergence criterion.
                              */
            MyFloat       lvalue;

                             /**
                              * Last step.
                              */
            unsigned int lstep;

                             /**
                              * Is set to @p true by
                              * @p set_failure_criterion and
                              * enables failure checking.
                              */
            bool         check_failure;

                             /*
                              * Stores the
                              * @p rel_failure_residual set by
                              * @p set_failure_criterion
                              */
            MyFloat       relative_failure_residual;

                             /**
                              * @p failure_residual equals the
                              * first residual multiplied by
                              * @p relative_crit set by
                              * @p set_failure_criterion (see there).
                              *
                              * Until the first residual is
                              * known it is 0.
                              */
            MyFloat       failure_residual;

                             /**
                              * Log convergence history to
                              * @p deallog.
                              */
            bool         m_log_history;
                             /**
                              * Log only every nth step.
                              */
            unsigned int m_log_frequency;

                             /**
                              * Log iteration result to
                              * @p deallog.  If true, after
                              * finishing the iteration, a
                              * statement about failure or
                              * success together with @p lstep
                              * and @p lvalue are logged.
                              */
            bool         m_log_result;

                             /**
                              * Control over the storage of
                              * history data. Set by
                              * enable_history_data().
                              */
            bool history_data_enabled;

                             /**
                              * Vector storing the result
                              * after each iteration step for
                              * later statistical analysis.
                              *
                              * Use of this vector is enabled
                              * by enable_history_data().
                              */
            std::vector<MyFloat> history_data;
    };
}//NAMESPACE

#endif // SOLVERCONTROL_H
