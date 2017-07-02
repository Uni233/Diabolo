#ifndef SOLVERCG_H
#define SOLVERCG_H

#include "solver.h"
#include "preconditionssor.h"

namespace VR_FEM
{

    class SolverCG : public Solver
    {
    public:
                             /**
                          * Standardized data struct to pipe
                          * additional data to the solver.
                          */
        struct AdditionalData
        {
                                             /**
                                              * Write coefficients alpha and beta
                                              * to the log file for later use in
                                              * eigenvalue estimates.
                                              */
            bool log_coefficients;

                                             /**
                          * Compute the condition
                          * number of the projected
                          * matrix.
                          *
                          * @note Requires LAPACK support.
                          */
        bool compute_condition_number;

                                             /**
                          * Compute the condition
                          * number of the projected
                          * matrix in each step.
                          *
                          * @note Requires LAPACK support.
                          */
        bool compute_all_condition_numbers;

                         /**
                          * Compute all eigenvalues of
                          * the projected matrix.
                          *
                          * @note Requires LAPACK support.
                          */
        bool compute_eigenvalues;

                                             /**
                                              * Constructor. Initialize data
                                              * fields.  Confer the description of
                                              * those.
                                              */
            AdditionalData (const bool log_coefficients = false,
                const bool compute_condition_number = false,
                const bool compute_all_condition_numbers = false,
                const bool compute_eigenvalues = false);
        };


                         /**
                          * Constructor. Use an object of
                          * type GrowingVectorMemory as
                          * a default to allocate memory.
                          */
        SolverCG (SolverControl        &cn,
              const AdditionalData &data=AdditionalData());

                         /**
                          * Virtual destructor.
                          */
        virtual ~SolverCG ();

                         /**
                          * Solve the linear system $Ax=b$
                          * for x.
                          */
        void      solve (const MySpMat         &A,
                         MyVector               &x,
                         const MyVector         &b,
                         PreconditionRelaxation &precondition);

      protected:
                         /**
                          * Implementation of the computation of
                          * the norm of the residual. This can be
                          * replaced by a more problem oriented
                          * functional in a derived class.
                          */
        virtual MyFloat criterion();

                         /**
                          * Interface for derived class.
                          * This function gets the current
                          * iteration vector, the residual
                          * and the update vector in each
                          * step. It can be used for a
                          * graphical output of the
                          * convergence history.
                          */
        virtual void print_vectors(const unsigned int step,
                       const MyVector& x,
                       const MyVector& r,
                       const MyVector& d) const;

                         /**
                          * Temporary vectors, allocated through
                          * the @p VectorMemory object at the start
                          * of the actual solution process and
                          * deallocated at the end.
                          */
//        RhsVec *Vr;
//        RhsVec *Vp;
//        RhsVec *Vz;
//        RhsVec *VAp;

                         /**
                          * Within the iteration loop, the
                          * square of the residual vector is
                          * stored in this variable. The
                          * function @p criterion uses this
                          * variable to compute the convergence
                          * value, which in this class is the
                          * norm of the residual vector and thus
                          * the square root of the @p res2 value.
                          */
        MyFloat res2;

                         /**
                          * Additional parameters.
                          */
        AdditionalData additional_data;

      private:
        void cleanup();
    };
}

#endif // SOLVERCG_H
