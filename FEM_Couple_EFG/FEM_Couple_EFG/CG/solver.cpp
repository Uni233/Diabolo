#include "stdafx.h"
#include "solver.h"

namespace VR_FEM
{


    Solver::Solver (VR_FEM::SolverControl        &solver_control)
            :
            cntrl(solver_control)
    {}



    SolverControl &
    Solver::control() const
    {
      return cntrl;
    }
}
