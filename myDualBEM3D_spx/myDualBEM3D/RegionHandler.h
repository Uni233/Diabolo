#ifndef _RegionHandler_h_
#define _RegionHandler_h_
#include "vrBase/vrBase.h"
#include "bemDefines.h"
#include <string>
#if USE_Fracture


namespace VR
{	
	class RegionHandler
	{
	public:
		/* Read region definitions from the given file into regionDefs
			* the file format is as follows:
			* comment lines start with a '#' (as the very first character), empty lines are ignored
			* region definitions start with a line "regions N", where N is the number of region definitions to follow
			* a region definition has the format n (a,b,c,d) ... where n is the region-ID and
			* the bracketed expression is a half-space condition for points (x,y,z) of the form ax + by + cz <= d
			*/
		int readRegionDefinitions(std::string filename); //node_map maps id to vector<double>

		/* Assign each element to the first region where at least one of its vertices matches all region-conditions
			*/
		int assignRegions(idMap& regions,  nodeMap& nodes,  elemMap& elems);
	protected:
		nodeMap regionDefs;
	};
}

#endif//USE_Fracture

#endif//_RegionHandler_h_