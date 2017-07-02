#include "poly2tri.h"
#include <vector>
#include <list>
#include "boost/tuple/tuple.hpp"
#include "../VR_Global_Define.h"
using namespace p2t;
namespace YC
{
	namespace Geometry
	{
		namespace ConstrainedDT
		{
			using namespace std;

			std::vector< boost::tuple< MyPoint,MyPoint,MyPoint > > constrainedDelaunayTriangulation(vector<p2t::Point*>& polyline);
		}//namespace ConstrainedDT
	}//namespace Geometry
}//namespace YC