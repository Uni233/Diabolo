#ifndef SOLVER_H
#define SOLVER_H

#include "VR_Global_Define.h"
#include "solvercontrol.h"

namespace VR_FEM
{
    class Solver
    {
      public:
                         /**
                          * Constructor. Takes a control
                          * object which evaluates the
                          * conditions for convergence,
                          * and an object to provide
                          * memory.
                          *
                          * Of both objects, a reference is
                          * stored, so it is the user's
                          * responsibility to guarantee that the
                          * lifetime of the two arguments is at
                          * least as long as that of the solver
                          * object.
                          */
//        Solver (SolverControl        &solver_control,
//                VectorMemory<VECTOR> &vector_memory);

                         /**
                          * Constructor. Takes a control
                          * object which evaluates the
                          * conditions for convergence. In
                          * contrast to the other
                          * constructor, this constructor
                          * denotes an internal object of
                          * type GrowingVectorMemory to
                          * allocate memory.
                          *
                          * A reference to the control
                          * object is stored, so it is the
                          * user's responsibility to
                          * guarantee that the lifetime of
                          * the two arguments is at least
                          * as long as that of the solver
                          * object.
                          */
        Solver (SolverControl        &solver_control);

                         /**
                          * Access to object that controls
                          * convergence.
                          */
        SolverControl & control() const;

      protected:
                         /**
                          * A static vector memory object
                          * to be used whenever no such
                          * object has been given to the
                          * constructor.
                          */
//        mutable GrowingVectorMemory<VECTOR> static_vector_memory;

                         /**
                          * Control structure.
                          */
        SolverControl &cntrl;

                         /**
                          * Memory for auxilliary vectors.
                          */
//        VectorMemory<VECTOR> &memory;
    };
}//namespace

#endif // SOLVER_H
