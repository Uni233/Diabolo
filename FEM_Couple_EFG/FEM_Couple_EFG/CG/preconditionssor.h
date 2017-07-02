#ifndef PRECONDITIONSSOR_H
#define PRECONDITIONSSOR_H

#include "preconditionrelaxation.h"

namespace VR_FEM
{
    class PreconditionSSOR : public PreconditionRelaxation
    {
    public:

                         /**
                          * A typedef to the base class.
                          */
        typedef PreconditionRelaxation BaseClass;


                         /**
                          * Initialize matrix and
                          * relaxation parameter. The
                          * matrix is just stored in the
                          * preconditioner object. The
                          * relaxation parameter should be
                          * larger than zero and smaller
                          * than 2 for numerical
                          * reasons. It defaults to 1.
                          */
        void initialize ( MySpMat &A,
                 const  BaseClass::AdditionalData &parameters =  BaseClass::AdditionalData());

        void initialize ( MySpMat &A, const MyFloat relax);

                         /**
                          * Apply preconditioner.
                          */

        void vmult (MyVector&, const MyVector&) ;

                         /**
                          * Apply transpose
                          * preconditioner. Since this is
                          * a symmetric preconditioner,
                          * this function is the same as
                          * vmult().
                          */

        void Tvmult (MyVector&, const MyVector&) const;


                         /**
                          * Perform one step of the
                          * preconditioned Richardson
                          * iteration
                          */

        void step (MyVector& x, const MyVector& rhs) const;

                         /**
                          * Perform one transposed step of
                          * the preconditioned Richardson
                          * iteration.
                          */

        void Tstep (MyVector& x, const MyVector& rhs) const;

      private:
                         /**
                          * An array that stores for each matrix
                          * row where the first position after
                          * the diagonal is located.
                          */
        std::vector<unsigned int> pos_right_of_diagonal;

        static void  precondition_SSOR ( const MySpMat &sm,
                                         MyVector             &dst,
                                         const MyVector        &src,
                                         const MyFloat        om,
                                         const std::vector<unsigned int> &pos_right_of_diagonal);
        static void SSOR_step (const MySpMat &sm,MyVector &v,const MyVector &b, const MyFloat om);
        static void SOR_step (const MySpMat &sm,MyVector &v,const MyVector &b,const MyFloat        om);
        static void TSOR_step (const MySpMat &sm,MyVector &v,const MyVector &b,const MyFloat        om);
    };
}


#endif // PRECONDITIONSSOR_H
