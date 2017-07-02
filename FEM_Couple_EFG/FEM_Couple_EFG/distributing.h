#ifndef DISTRIBUTING_H
#define DISTRIBUTING_H

#include "constant_numbers.h"

namespace VR_FEM
{
    struct Distributing
      {
          Distributing (const unsigned int global_row = numbers::invalid_unsigned_int,
                        const unsigned int local_row = numbers::invalid_unsigned_int);
          Distributing (const Distributing &in);
          Distributing & operator = (const Distributing &in);
          bool operator < (const Distributing &in) const {return global_row<in.global_row;};

          unsigned int global_row;
          unsigned int local_row;
          mutable unsigned int constraint_position;
      };

      inline
      Distributing::Distributing (const unsigned int global_row,
                                  const unsigned int local_row) :
                      global_row (global_row),
                      local_row (local_row),
                      constraint_position (numbers::invalid_unsigned_int) {}

      inline
      Distributing::Distributing (const Distributing &in)
                      :
                      constraint_position (numbers::invalid_unsigned_int)
      {
        *this = (in);
      }

      inline
      Distributing & Distributing::operator = (const Distributing &in)
      {
        global_row = in.global_row;
        local_row = in.local_row;
                                         // the constraints pointer should not
                                         // contain any data here.
        Q_ASSERT (constraint_position == numbers::invalid_unsigned_int);

        if (in.constraint_position != numbers::invalid_unsigned_int)
          {
            constraint_position = in.constraint_position;
            in.constraint_position = numbers::invalid_unsigned_int;
          }
        return *this;
      }
}

#endif // DISTRIBUTING_H
