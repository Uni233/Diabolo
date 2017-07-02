#ifndef GLOBALROWSFROMLOCAL_H
#define GLOBALROWSFROMLOCAL_H

#include "VR_Global_Define.h"
#include <vector>
#include "distributing.h"
#include "datacache.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace VR_FEM
{
class GlobalRowsFromLocal
  {
    public:
      GlobalRowsFromLocal (const unsigned int n_local_rows)
                      :
                      total_row_indices (n_local_rows),
                      n_active_rows (n_local_rows),
                      n_inhomogeneous_rows (0)
        {}


                                       // implemented below
      void insert_index (const unsigned int global_row,
                         const unsigned int local_row,
                         const double       constraint_value);
      void sort ();

                                       // Print object for debugging
                                       // purpose
      void print(std::ostream& os)
        {
          os << "Active rows " << n_active_rows << std::endl
             << "Constr rows " << n_constraints() << std::endl
             << "Inhom  rows " << n_inhomogeneous_rows << std::endl
             << "Local: ";
          for (unsigned int i=0 ; i<total_row_indices.size() ; ++i)
            os << ' ' << std::setw(4) << total_row_indices[i].local_row;
          os << std::endl
             << "Global:";
          for (unsigned int i=0 ; i<total_row_indices.size() ; ++i)
            os << ' ' << std::setw(4) << total_row_indices[i].global_row;
          os << std::endl
             << "ConPos:";
          for (unsigned int i=0 ; i<total_row_indices.size() ; ++i)
            os << ' ' << std::setw(4) << total_row_indices[i].constraint_position;
          os << std::endl;
        }


                                       // return all kind of information on the
                                       // constraints

                                       // returns the number of global indices in the
                                       // struct
      unsigned int size () const
        {
          return n_active_rows;
        }

                                       // returns the global index of the
                                       // counter_index-th entry in the list
      unsigned int & global_row (const unsigned int counter_index)
        {
          return total_row_indices[counter_index].global_row;
        }

                                       // returns the number of constraints that are
                                       // associated to the counter_index-th entry in
                                       // the list
      unsigned int size (const unsigned int counter_index) const
        {
          return (total_row_indices[counter_index].constraint_position ==
                  numbers::invalid_unsigned_int ?
                  0 :
                  data_cache.get_size(total_row_indices[counter_index].
                                      constraint_position));
        }

                                       // returns the global row associated with the
                                       // counter_index-th entry in the list
      const unsigned int & global_row (const unsigned int counter_index) const
        {
          return total_row_indices[counter_index].global_row;
        }

                                       // returns the local row in the cell matrix
                                       // associated with the counter_index-th entry
                                       // in the list. Returns invalid_unsigned_int
                                       // for invalid unsigned ints
      const unsigned int & local_row (const unsigned int counter_index) const
        {
          return total_row_indices[counter_index].local_row;
        }

                                       // writable index
      unsigned int & local_row (const unsigned int counter_index)
        {
          return total_row_indices[counter_index].local_row;
        }

                                       // returns the local row in the cell matrix
                                       // associated with the counter_index-th entry
                                       // in the list in the index_in_constraint-th
                                       // position of constraints
      unsigned int local_row (const unsigned int counter_index,
                              const unsigned int index_in_constraint) const
        {
          return (data_cache.get_entry(total_row_indices[counter_index].constraint_position)
                  [index_in_constraint]).first;
        }

                                       // returns the value of the constraint in the
                                       // counter_index-th entry in the list in the
                                       // index_in_constraint-th position of
                                       // constraints
      double constraint_value (const unsigned int counter_index,
                               const unsigned int index_in_constraint) const
        {
          return (data_cache.get_entry(total_row_indices[counter_index].constraint_position)
                  [index_in_constraint]).second;
        }

                                       // returns whether there is one row with
                                       // indirect contributions (i.e., there has
                                       // been at least one constraint with
                                       // non-trivial ConstraintLine)
      bool have_indirect_rows () const
        {
          return data_cache.element_size;
        }

                                       // append an entry that is
                                       // constrained. This means that
                                       // there is one less nontrivial
                                       // row
      void insert_constraint (const unsigned int constrained_local_dof)
        {
          --n_active_rows;
          total_row_indices[n_active_rows].local_row = constrained_local_dof;
        }

                                       // returns the number of constrained
                                       // dofs in the structure. Constrained
                                       // dofs do not contribute directly to
                                       // the matrix, but are needed in order
                                       // to set matrix diagonals and resolve
                                       // inhomogeneities
      unsigned int n_constraints () const
        {
          return total_row_indices.size()-n_active_rows;
        }

                                       // returns the number of constrained
                                       // dofs in the structure that have an
                                       // inhomogeneity
      unsigned int n_inhomogeneities () const
        {
          return n_inhomogeneous_rows;
        }

                                       // tells the structure that the ith
                                       // constraint is
                                       // inhomogeneous. inhomogeneous
                                       // constraints contribute to right hand
                                       // sides, so to have fast access to
                                       // them, put them before homogeneous
                                       // constraints
      void set_ith_constraint_inhomogeneous (const unsigned int i)
        {
          Q_ASSERT (i >= n_inhomogeneous_rows);
          std::swap (total_row_indices[n_active_rows+i],
                     total_row_indices[n_active_rows+n_inhomogeneous_rows]);
          n_inhomogeneous_rows++;
        }

                                       // the local row where
                                       // constraint number i was
                                       // detected, to find that row
                                       // easily when the
                                       // GlobalRowsToLocal has been
                                       // set up
      unsigned int constraint_origin (unsigned int i) const
        {
          return total_row_indices[n_active_rows+i].local_row;
        }

                                       // a vector that contains all the
                                       // global ids and the corresponding
                                       // local ids as well as a pointer to
                                       // that data where we store how to
                                       // resolve constraints.
      std::vector<Distributing> total_row_indices;

    private:
                                       // holds the actual data from
                                       // the constraints
      DataCache                 data_cache;

                                       // how many rows there are,
                                       // constraints disregarded
      unsigned int              n_active_rows;

                                       // the number of rows with
                                       // inhomogeneous constraints
      unsigned int              n_inhomogeneous_rows;
};

                                   // a function that appends an additional
                                   // row to the list of values, or appends a
                                   // value to an already existing
                                   // row. Similar functionality as for
                                   // std::map<unsigned int,Distributing>, but
                                   // here done for a
                                   // std::vector<Distributing>, much faster
                                   // for short lists as we have them here
  inline
  void
  GlobalRowsFromLocal::insert_index (const unsigned int global_row,
                                     const unsigned int local_row,
                                     const double       constraint_value)
  {
    typedef std::vector<Distributing>::iterator index_iterator;
    index_iterator pos, pos1;
    Distributing row_value (global_row);
    std::pair<unsigned int,double> constraint (local_row, constraint_value);

                                     // check whether the list was really
                                     // sorted before entering here
    for (unsigned int i=1; i<n_active_rows; ++i)
      Q_ASSERT (total_row_indices[i-1] < total_row_indices[i]);

    pos = std::lower_bound (total_row_indices.begin(),
                            total_row_indices.begin()+n_active_rows,
                            row_value);
    if (pos->global_row == global_row)
      pos1 = pos;
    else
      {
        pos1 = total_row_indices.insert(pos, row_value);
        ++n_active_rows;
      }

    if (pos1->constraint_position == numbers::invalid_unsigned_int)
      pos1->constraint_position = data_cache.insert_new_index (constraint);
    else
      data_cache.append_index (pos1->constraint_position, constraint);
  }

                                   // this sort algorithm sorts
                                   // std::vector<Distributing>, but does not
                                   // take the constraints into account. this
                                   // means that in case that constraints are
                                   // already inserted, this function does not
                                   // work as expected. Use shellsort, which
                                   // is very fast in case the indices are
                                   // already sorted (which is the usual case
                                   // with DG elements), and not too slow in
                                   // other cases
  inline
  void
  GlobalRowsFromLocal::sort ()
  {
    unsigned int i, j, j2, temp, templ, istep;
    unsigned int step;

                                     // check whether the
                                     // constraints are really empty.
    const unsigned int length = size();


    step = length/2;
    while (step > 0)
      {
        for (i=step; i < length; i++)
          {
            istep = step;
            j = i;
            j2 = j-istep;
            temp = total_row_indices[i].global_row;
            templ = total_row_indices[i].local_row;
            if (total_row_indices[j2].global_row > temp)
              {
                while ((j >= istep) && (total_row_indices[j2].global_row > temp))
                  {
                    total_row_indices[j].global_row = total_row_indices[j2].global_row;
                    total_row_indices[j].local_row = total_row_indices[j2].local_row;
                    j = j2;
                    j2 -= istep;
                  }
                total_row_indices[j].global_row = temp;
                total_row_indices[j].local_row = templ;
              }
          }
        step = step>>1;
      }
  }
}

#endif // GLOBALROWSFROMLOCAL_H
