#include "constrainedDelaunayTriangulation.h"

namespace YC
{
	namespace Geometry
	{
		namespace ConstrainedDT
		{
			using namespace std;
			/// Constrained triangles
			template <class C> void FreeClear( C & cntr ) {
				for ( typename C::iterator it = cntr.begin(); 
					it != cntr.end(); ++it ) {
						delete * it;
				}
				cntr.clear();
			}

			std::vector< boost::tuple< MyPoint,MyPoint,MyPoint > > constrainedDelaunayTriangulation(vector<p2t::Point*>& polyline)
			{
				std::vector< boost::tuple< MyPoint,MyPoint,MyPoint > > retVal;
				vector<Triangle*> triangles;
				vector< vector<Point*> > polylines;

				polylines.push_back(polyline);

				CDT* cdt = new CDT(polyline);
				cdt->Triangulate();
				triangles = cdt->GetTriangles();

				for (int i = 0; i < triangles.size(); i++) {
					Triangle& t = *triangles[i];
					Point& a = *t.GetPoint(0);
					Point& b = *t.GetPoint(1);
					Point& c = *t.GetPoint(2);

					retVal.push_back(boost::make_tuple(MyPoint(a.x, a.y,0.f),MyPoint(b.x, b.y,0.f),MyPoint(c.x, c.y,0.f)));
					
				}

				delete cdt;

				// Free points
				for(int i = 0; i < polylines.size(); i++) {
					vector<Point*> poly = polylines[i];
					FreeClear(poly);
				}
				return retVal;
			}
		}//namespace CDT
	}//namespace Geometry
}//namespace YC