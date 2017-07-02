#ifndef _vrIterator_h_
#define _vrIterator_h_
#include <boost/typeof/typeof.hpp>
#include <map>
#include <set>
using std::set;
using std::map;
//#define iterof(i,s) typeof((s).begin()) i((s).begin())
#define iterof(i,s) BOOST_TYPEOF((s).begin()) i((s).begin())     
#define iterAllOf(i,s) BOOST_TYPEOF((s).begin()) i((s).begin());(i)!=(s).end();++(i)
namespace VR
{
	namespace Iterator
	{
		template<class T, class L>
		inline bool has_elt(set<T, L> const &s, T const &k)
		{
			return s.find(k) != s.end();
		}
		template<class T, class V, class L>
		inline bool has_key(map<T, V, L> const &s, T const &k)
		{
			return s.find(k) != s.end();
		}
	}

}//namespace VR

#endif//_vrIterator_h_