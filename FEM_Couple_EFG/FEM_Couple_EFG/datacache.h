#ifndef DATACACHE_H
#define DATACACHE_H

#include <vector>
#include <string.h>
#include "VR_Global_Define.h"

namespace VR_FEM
{
  struct DataCache
  {
      DataCache ()
                      :
                      element_size (0),
                      data (0),
                      n_used_elements(0)
        {}

      ~DataCache()
        {
          delete [] data;
          data = 0;
        }

      void reinit ()
        {
          Q_ASSERT (element_size == 0);
          element_size = 6;
          data = new std::pair<unsigned int,double> [20*6];
          individual_size.resize(20);
          n_used_elements = 0;
        }

      unsigned int element_size;

      std::pair<unsigned int,double> * data;

      std::vector<unsigned int> individual_size;

      unsigned int n_used_elements;

      unsigned int insert_new_index (const std::pair<unsigned int,double> &pair)
        {
          if (element_size == 0)
            reinit();
          if (n_used_elements == individual_size.size())
            {
              std::pair<unsigned int,double> * new_data =
                new std::pair<unsigned int,double> [2*individual_size.size()*element_size];
              memcpy (new_data, data, individual_size.size()*element_size*
                      sizeof(std::pair<unsigned int,double>));
              delete [] data;
              data = new_data;
              individual_size.resize (2*individual_size.size(), 0);
            }
          unsigned int index = n_used_elements;
          data[index*element_size] = pair;
          individual_size[index] = 1;
          ++n_used_elements;
          return index;
        }

      void append_index (const unsigned int index,
                         const std::pair<unsigned int,double> &pair)
        {
          //ASSERT (index, n_used_elements);
          const unsigned int my_size = individual_size[index];
          if (my_size == element_size)
            {
              std::pair<unsigned int,double> * new_data =
                new std::pair<unsigned int,double> [2*individual_size.size()*element_size];
              for (unsigned int i=0; i<n_used_elements; ++i)
                memcpy (&new_data[i*element_size*2], &data[i*element_size],
                        element_size*sizeof(std::pair<unsigned int,double>));
              delete [] data;
              data = new_data;
              element_size *= 2;
            }
          data[index*element_size+my_size] = pair;
          individual_size[index]++;
        }

      unsigned int
      get_size (const unsigned int index) const
        {
          return individual_size[index];
        }

      const std::pair<unsigned int,double> *
      get_entry (const unsigned int index) const
        {
          return &data[index*element_size];
        }
  };
}

#endif // DATACACHE_H
