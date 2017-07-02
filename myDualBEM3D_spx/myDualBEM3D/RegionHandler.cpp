#include "RegionHandler.h"
#include <fstream>
#if USE_Fracture
using namespace std;
namespace VR
{
	int RegionHandler::assignRegions(idMap& regions,  nodeMap& nodes,  elemMap& elems)
	{
		regions.clear();
		Eigen::Vector3d n; // tmp storage for a region-constraint
		double d; // such that the constraint is evaluated on p as n.dot(p)<=d
		bool found, match;

		unsigned int i=0;
		printf("nodes.size = %d\n",nodes.size());
		for(elemMap::const_iterator it=elems.begin(); it!=elems.end(); ++it,++i)
		{
			// check which region fits this tri
			//printf("%d %d %d\n",it->second[0], it->second[1], it->second[2]);
			Eigen::Vector3d // a,b,c are node coordinates
				a (nodes[it->second[0]][0], nodes[it->second[0]][1], nodes[it->second[0]][2]),
				b (nodes[it->second[1]][0], nodes[it->second[1]][1], nodes[it->second[1]][2]),
				c (nodes[it->second[2]][0], nodes[it->second[2]][1], nodes[it->second[2]][2]);
			found=false;
			for(nodeMap::iterator rd=regionDefs.begin(); rd!=regionDefs.end() && !found; ++rd)
			{ //iterate region defs
				match=true;
				for(int j=0; j<rd->second.size() && match; j+=4)
				{
					n[0]=rd->second[j  ]; n[1]=rd->second[j+1]; n[2]=rd->second[j+2];
					d   =rd->second[j+3];
					if( n.dot(a) > d || // using && means tri is added to region if at least 1 vertex matches
						n.dot(b) > d || // using || means tri is added to region if all of its vertices match
						n.dot(c) > d ) 
					{
						match=false;
					}
				}
				if(match)
				{
					found=true;
					regions[it->first]=rd->first;
				}
			}
		}
		return 0;
	}

	int RegionHandler::readRegionDefinitions(std::string filename){ //node_map maps id to vector<double>
		int nRegions=0; // number of regionDefs read
		bool regKwdFound=false; // look for the keyword "regions" followed by the number of regions to read
		std::string line, token;
		int ret, done=0;
		char check;
		double a,b,c,d;
		unsigned int nextId;

		regionDefs.clear();
		/*

		regions 2
		1 (1,0,0,0.15)
		2 (-0.17,-0.99,0,-0.08986) (-1,0.6,0,0.08)

		*/
		std::istringstream strstream;
		ifstream in(filename.c_str());
		if(!in.is_open()) return -1;

		while(in.good()){
			getline(in, line);
			if(line.empty() || line.at(0)=='#') continue; // comments have # in the first column
			strstream.clear();
			strstream.str(line);

			getline(strstream, token, ' ');
			if(!regKwdFound){
				if(token.compare("regions")==0){
					regKwdFound=true;
					getline(strstream, token, ' '); //next token is number of regions to read
					ret=sscanf(token.c_str(), "%u", &nRegions);
					if(ret!=1){
						in.close();
						printf("invalid ret=%d should be 1 on token %s\n",ret, token.c_str());
						return -1;
					}
				}
			}else if(done<nRegions){ //reading next region until we have plenty
				//printf("processing '%s' done %d\n",line.c_str(),done);
				ret=sscanf(token.c_str(), "%u", &nextId);
				if(ret==1) while(getline(strstream, token, ' ')){ //token is now one condition of the region definition
					//printf("parsing token '%s'",token.c_str());
					ret=sscanf(token.c_str(), "(%lf,%lf,%lf,%lf%c",&a,&b,&c,&d,&check);
					if(ret==5 && check==')'){ // correct format
						regionDefs[nextId].push_back(a);
						regionDefs[nextId].push_back(b);
						regionDefs[nextId].push_back(c);
						regionDefs[nextId].push_back(d);
						//printf(" ... ok\n");
					}
					//else printf(" ... reject ret=%d, check='%c'\n",ret,check);
				}
				++done;
			}
		}
		printf("\nregionDefs:\n");
		for (iterAllOf(ci,regionDefs))
		{
			printf("%d [%f,%f,%f,%f]\n",ci->first, ci->second[0], ci->second[1], ci->second[2], ci->second[3]);
		}
		vrPause;
		//printMap(regionDefs);
		// finally add an empty region def which will be the default
		if(regionDefs.empty()) 
		{
			nextId=ELEM_BASE_INDEX;
		}
		else 
		{
			nextId = regionDefs.rbegin()->first +1;
		}
		regionDefs[nextId].assign(0,0.0);

		return nRegions;
	}
}

#endif//USE_Fracture
