
#include "preconditionssor.h"
#include "../VR_MACRO.h"

namespace YC
{
    void PreconditionSSOR::initialize ( MySpMat &rA, const MyFloat relax)
    {
        initialize(rA,BaseClass::AdditionalData(relax));
    }

    void
    PreconditionSSOR::initialize ( MySpMat &rA,
                          const  BaseClass::AdditionalData &parameters)
    {
      this->PreconditionRelaxation::initialize (rA, parameters);

                       // in case we have a SparseMatrix class,
                       // we can extract information about the
                       // diagonal.
//      const SparseMatrix<typename MATRIX::value_type> * mat =
//        dynamic_cast<const SparseMatrix<typename MATRIX::value_type> *>(&*this->A);

                       // calculate the positions first after
                       // the diagonal.
      if (true/*mat != 0*/)
      {
/*
const int * innerPtr = rowMatrix.innerIndexPtr();
//        const int * outerPtr = rowMatrix.outerIndexPtr();
//        const double * val = rowMatrix.valuePtr();
*/
//#ifdef _TEST
          const int  * rowstart_ptr = rA.outerIndexPtr();//mat->get_sparsity_pattern().get_rowstart_indices();
          const  int * const colnums = rA.innerIndexPtr();// mat->get_sparsity_pattern().get_column_numbers();
          const unsigned int n = rA.rows();//this->A->n();
          pos_right_of_diagonal.resize(n);
          for (unsigned int row=0; row<n; ++row, ++rowstart_ptr)
            {
                               // find the first element in this line
                               // which is on the right of the diagonal.
                               // we need to precondition with the
                               // elements on the left only.
                               // note: the first entry in each
                               // line denotes the diagonal element,
                               // which we need not check.
              pos_right_of_diagonal[row] =
                std::lower_bound (&colnums[*rowstart_ptr],
                                        &colnums[*(rowstart_ptr+1)],
                                        row) - colnums;
              //Q_ASSERT(colnums[ pos_right_of_diagonal[row] ] == row);
			  //printf("row(%d) pos_right_of_diagonal[%d](%d) colnums(%d)\n",row,row,pos_right_of_diagonal[row],colnums[ pos_right_of_diagonal[row] ]);
            }
//#endif
       }
    }




    void
    PreconditionSSOR::vmult (MyVector &dst, const MyVector &src)
    {
      Q_ASSERT (this->A!=0);
//#ifdef _TEST
      precondition_SSOR (*A,dst, src, this->relaxation, pos_right_of_diagonal);
//#endif
    }





    inline void
    PreconditionSSOR::Tvmult (MyVector &dst, const MyVector &src) const
    {
      Q_ASSERT (this->A!=0);
//#ifdef _TEST
      precondition_SSOR (*A,dst, src, this->relaxation, pos_right_of_diagonal);
//#endif
    }





     void
    PreconditionSSOR::step (MyVector &dst, const MyVector &src) const
    {
      Q_ASSERT (this->A!=0);
//#ifdef _TEST
      SSOR_step (*A,dst, src, this->relaxation);
//#endif
    }





     void
    PreconditionSSOR::Tstep (MyVector &dst, const MyVector &src) const
    {
      step (dst, src);
    }





    void PreconditionSSOR::SOR_step (const MySpMat &sm,MyVector &v,const MyVector &b,const MyFloat        om)
    {
//#ifdef _TEST

        Q_ASSERT (sm.rows() == v.rows());
        Q_ASSERT (sm.rows() == b.rows());
        const int  *rowstart_ptr = sm.outerIndexPtr();
        const int  *colnums_ptr  = sm.innerIndexPtr();
        const MyFloat *       val = sm.valuePtr();

      for (unsigned int row=0; row<sm.rows(); ++row)
        {
          MyFloat s = b(row);
          for (unsigned int j=rowstart_ptr[row]; j<rowstart_ptr[row+1]; ++j)
            {
              s -= val[j] * v(colnums_ptr[j]);
            }
          Q_ASSERT( sm.coeff(row,row)!= 0. );
          v(row) += s * om / sm.coeff(row,row);
        }
//#endif
    }



    void PreconditionSSOR::TSOR_step (const MySpMat &sm,MyVector &v,const MyVector &b,const MyFloat        om)
    {
//#ifdef _TEST

      Q_ASSERT (sm.rows() == v.rows());
      Q_ASSERT (sm.rows() == b.rows());

        const int  *rowstart_ptr = sm.outerIndexPtr();
        const int  *colnums_ptr  = sm.innerIndexPtr();
        const MyFloat *       val = sm.valuePtr();

      for (int row=sm.rows()-1; row>=0; --row)
        {
          MyFloat s = b(row);
          for (unsigned int j=rowstart_ptr[row]; j<rowstart_ptr[row+1]; ++j)
            {
              s -= val[j] * v(colnums_ptr[j]);
            }
          Q_ASSERT( sm.coeff(row,row)!= 0. );
          v(row) += s * om / sm.coeff(row,row);
        }
//#endif
    }

    void PreconditionSSOR::SSOR_step (const MySpMat& sm,MyVector &v,const MyVector &b,const MyFloat om)
    {
        SOR_step(sm,v,b,om);
        TSOR_step(sm,v,b,om);
    }


    void  PreconditionSSOR::precondition_SSOR ( const MySpMat &sm,
                                                MyVector             &dst,
                                                const MyVector        &src,
                                                const MyFloat        om,
                                                const std::vector<unsigned int> &pos_right_of_diagonal)
    {
                       // to understand how this function works
                       // you may want to take a look at the CVS
                       // archives to see the original version
                       // which is much clearer...
//#ifdef _TEST
      Q_ASSERT ( dst.rows() == sm.rows() );
      Q_ASSERT ( src.rows() == sm.rows() );

      const unsigned int  n            = src.rows();
      const int  *rowstart_ptr = sm.outerIndexPtr();//&cols->rowstart[0];
      const int  *colnums_ptr  = sm.innerIndexPtr();
      const MyFloat *       val = sm.valuePtr();
      int                 dst_ptr      = 0;

                       // case when we have stored the position
                       // just right of the diagonal (then we
                       // don't have to search for it).
      if (pos_right_of_diagonal.size() != 0)
        {
          Q_ASSERT (pos_right_of_diagonal.size() == dst.rows());

                       // forward sweep
          for (unsigned int row=0; row<n; ++row, ++dst_ptr, ++rowstart_ptr)
          {
              dst(dst_ptr) = src(row);
              const unsigned int first_right_of_diagonal_index = pos_right_of_diagonal[row];

              Q_ASSERT (first_right_of_diagonal_index <= *(rowstart_ptr+1));
              MyFloat s = 0;
              for (unsigned int j=(*rowstart_ptr); j<first_right_of_diagonal_index; ++j)
                s += val[j] * dst(colnums_ptr[j]);

                           // divide by diagonal element
              dst(dst_ptr) -= s * om;
              Q_ASSERT(val[first_right_of_diagonal_index]!= 0.);
              dst(dst_ptr) /= val[first_right_of_diagonal_index];
          };

          rowstart_ptr = sm.outerIndexPtr();//&cols->rowstart[0];
          dst_ptr      = 0;//&dst(0);
          for (int row=0 ; rowstart_ptr!=&(sm.outerIndexPtr()[n]); ++rowstart_ptr, ++dst_ptr,++row)
          {
              const unsigned int first_right_of_diagonal_index = pos_right_of_diagonal[row];
            dst(dst_ptr) *= om*(2.0f-om)*val[first_right_of_diagonal_index];
          }

                       // backward sweep
          rowstart_ptr = &(sm.outerIndexPtr()[n-1]);//&cols->rowstart[n-1];
          dst_ptr      = n-1;//&dst(n-1);
          for (int row=n-1; row>=0; --row, --rowstart_ptr, --dst_ptr)
          {
              const unsigned int end_row = *(rowstart_ptr+1);
              const unsigned int first_right_of_diagonal_index = pos_right_of_diagonal[row];
              MyFloat s = 0;
              for (unsigned int j=first_right_of_diagonal_index+1; j<end_row; ++j)
                s += val[j] * dst(colnums_ptr[j]);

              dst(dst_ptr) -= s * om;
              Q_ASSERT(val[first_right_of_diagonal_index]!= 0.);
              dst(dst_ptr) /= val[first_right_of_diagonal_index];
          };
          return;
        }
//#endif

    }
}

