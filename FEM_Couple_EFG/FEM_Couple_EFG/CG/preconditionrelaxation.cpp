#include "stdafx.h"
#include "preconditionrelaxation.h"

namespace VR_FEM
{
    void
    PreconditionRelaxation::initialize (MySpMat &rA,
                            const AdditionalData &parameters)
    {
      A = &rA;//A.reset(&rA);
      relaxation = parameters.relaxation;
    }

    void PreconditionRelaxation::initialize (MySpMat &rA,const MyFloat relax)
    {
        A = &rA;//A.reset(&rA);
        relaxation = relax;
    }


    void
    PreconditionRelaxation::clear ()
    {
        A = 0;
      //A.reset();
    }


}
