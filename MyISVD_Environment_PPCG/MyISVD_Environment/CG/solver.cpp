
#include "solver.h"

namespace YC
{


    Solver::Solver (YC::SolverControl        &solver_control)
            :
            cntrl(solver_control)
    {}



    SolverControl &
    Solver::control() const
    {
      return cntrl;
    }
}
