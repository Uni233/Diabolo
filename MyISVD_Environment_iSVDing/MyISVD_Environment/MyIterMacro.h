#ifndef _MYITERMACRO_H_
#define _MYITERMACRO_H_

#include <map>
#include <set>
using std::set;
using std::map;

#include <boost/typeof/typeof.hpp>

#define iterof(i,s) BOOST_TYPEOF((s).begin()) i((s).begin())	//#define iterof(i,s) typeof((s).begin()) i((s).begin())
#define iterAllOf(i,s) BOOST_TYPEOF((s).begin()) i((s).begin());(i)!=(s).end();++(i)
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

/*

const Triangle::TriangleSet& tris = triangles();
for (iterAllOf(i,tris))
for (iterof (i,tris); i !=tris.end (); i++)

*/


#endif//_MYITERMACRO_H_