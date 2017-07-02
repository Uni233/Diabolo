#ifndef PRECONDITIONRELAXATION_H
#define PRECONDITIONRELAXATION_H

#include <boost/smart_ptr.hpp>
#include "../VR_Global_Define.h"

namespace VR_FEM
{

    class PreconditionRelaxation
    {
    public:
                         /**
                          * Class for parameters.
                          */
        class AdditionalData
        {
          public:
                         /**
                          * Constructor.
                          */
        AdditionalData (const MyFloat relaxation = 1.):relaxation (relaxation){}

                         /**
                          * Relaxation parameter.
                          */
        MyFloat relaxation;
        };

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
        void initialize (MySpMat &A,
                 const AdditionalData & parameters = AdditionalData());

        void initialize (MySpMat &rA,const MyFloat relax);

                         /**
                          * Release the matrix and reset
                          * its pointer.
                          */
        void clear();

        virtual void vmult (MyVector&, const MyVector&) = 0;

      protected:
                         /**
                          * Pointer to the matrix object.
                          */
        MySpMat * A;
        //boost::shared_ptr< SpMat > A;
        //SmartPointer<const MATRIX, PreconditionRelaxation<MATRIX> > A;

                         /**
                          * Relaxation parameter.
                          */
        MyFloat relaxation;
    };
}

#endif // PRECONDITIONRELAXATION_H
