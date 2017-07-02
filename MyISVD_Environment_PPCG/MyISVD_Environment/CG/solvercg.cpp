
#include "solvercg.h"
#include "../constant_numbers.h"

namespace YC
{

    SolverCG::AdditionalData::
    AdditionalData (const bool log_coefficients,
            const bool compute_condition_number,
            const bool compute_all_condition_numbers,
            const bool compute_eigenvalues)
                    :
                    log_coefficients (log_coefficients),
            compute_condition_number(compute_condition_number),
            compute_all_condition_numbers(compute_all_condition_numbers),
            compute_eigenvalues(compute_eigenvalues)
    {}



    SolverCG::SolverCG (YC::SolverControl        &cn,
                    const AdditionalData &data)
            :
                    Solver(cn),
                    additional_data(data)
    {}



    SolverCG::~SolverCG ()
    {}




    MyFloat
    SolverCG::criterion()
    {
      return std::sqrt(res2);
    }




    void
    SolverCG::cleanup()
    {
//      this->memory.free(Vr);
//      this->memory.free(Vp);
//      this->memory.free(Vz);
//      this->memory.free(VAp);
//      deallog.pop();
    }




    void
    SolverCG::print_vectors(const unsigned int,
                    const MyVector&,
                    const MyVector&,
                    const MyVector&) const
    {}





    void
    SolverCG::solve (const MySpMat         &A,
                     MyVector               &x,
                     const MyVector         &b,
                     PreconditionRelaxation &precondition)
    {
      SolverControl::State conv=SolverControl::iterate;
                       // Should we build the matrix for
                       // eigenvalue computations?
      bool do_eigenvalues = additional_data.compute_condition_number
                | additional_data.compute_all_condition_numbers
                | additional_data.compute_eigenvalues;
      Q_ASSERT(false == do_eigenvalues);
      MyFloat eigen_beta_alpha = 0;

                       // vectors used for eigenvalue
                       // computations
      std::vector<MyFloat> diagonal;
      std::vector<MyFloat> offdiagonal;

      try {
                         // define some aliases for simpler access
          MyVector g(x.rows());//  = *Vr;
          MyVector h(x.rows());//  = *Vp;
          MyVector d(x.rows());//  = *Vz;
          MyVector Ad(x.rows());// = *VAp;
                         // resize the vectors, but do not set
                         // the values since they'd be overwritten
                         // soon anyway.
        //g.resize(x.rows());//reinit(x, true);
        g.setZero();
        //h.resize(x.rows());//reinit(x, true);
        h.setZero();
        //d.resize(x.rows());//reinit(x, true);
        d.setZero();
        //x.rows()Ad.resize(x.rows());//reinit(x, true);
        Ad.setZero();
                         // Implementation taken from the DEAL
                         // library
        int  it=0;
        MyFloat res,gh,alpha,beta;

                         // compute residual. if vector is
                         // zero, then short-circuit the
                         // full computation
        if (x.nonZeros()/*all_zero()*/)
        {
            g = A*x;//A.vmult(g,x);//
            g += (-1.) * b;//g.add(-1.,b);//*this += a*V
        }
        else
        {
            g = (-1.) * b;//g.equ(-1.,b);//*this = a*u.
        }

        res = numbers::l2_norm(g);
        //res = g.l2_norm();//std::sqrt(norm_sqr());

        conv = this->control().check(0,res);
        if (conv)
          {
            cleanup();
            return;
          }

        precondition.vmult(h,g);

        d = (-1.) * h;//d.equ(-1.,h);

        gh = g.adjoint()*h;//   gh = g*h;//double dp = v.adjoint()*w;

        
        while ( conv == SolverControl::iterate)
          {
            
                it++;
                Ad = A * d;//A.vmult(Ad,d);//dst = M*src

                alpha = d.adjoint()*Ad;
                Q_ASSERT(alpha != 0.);
                alpha = gh/alpha;

                g += alpha * Ad;//g.add(alpha,Ad);//*this += a*V
                x += alpha * d;//x.add(alpha,d );
				
                res = numbers::l2_norm(g);//res = g.l2_norm();
				

                print_vectors(it, x, g, d);

                conv = this->control().check(it,res);
                if (conv != SolverControl::iterate)
                  break;

                precondition.vmult(h,g);

                beta = gh;
                Q_ASSERT(beta != 0.);
                gh   = g.adjoint()*h;
                beta = gh/beta;

                d = beta * d+ (-1.) * h;//d.sadd(beta,-1.,h);//*this = s*(*this)+a*V
          }
      }
      catch (...)
        {
          cleanup();
          throw;
        }

                       // Write eigenvalues or condition number
      if (do_eigenvalues)
        {
          Q_ASSERT(false);
        }

                       // Deallocate Memory
      cleanup();

                       // in case of failure: throw
                       // exception
      if (this->control().last_check() != SolverControl::success)
        throw SolverControl::NoConvergence (this->control().last_step(),
                        this->control().last_value());
                       // otherwise exit as normal
    }
}

