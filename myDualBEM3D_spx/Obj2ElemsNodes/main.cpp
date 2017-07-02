#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <glm/glm.hpp>
#include <string>
#include<iomanip>
#include <set>



#define vrPause system("pause")

typedef double vrFloat;
typedef glm::vec2 vrGLMVec2;
typedef glm::vec3 vrGLMVec3;
typedef glm::vec4 vrGLMVec4;
typedef glm::mat3 vrGLMMat3;
typedef glm::mat4 vrGLMMat4;
typedef Eigen::Vector3i vrVec3I;
typedef int vrInt;
typedef char const * vrLpsz;
typedef std::string vrString;
typedef std::size_t vrSizt_t;

class vec3LessCompare
	{
	public:
		bool operator()(const vrGLMVec3& lhs,const vrGLMVec3& rhs)const
		{
			if( lhs.z < rhs.z )
			{   return true;
			}
			else if(lhs.z > rhs.z)
			{   return false;
			}
			// Otherwise z is equal
			if( lhs.y < rhs.y )
			{   return true;
			}
			else if( lhs.y > rhs.y )
			{   return false;
			}
			// Otherwise z and y are equal
			if( lhs.x < rhs.x )
			{   return true;
			}
			return false;
		}
	};

std::vector< vrGLMVec3 >  points;
std::vector< vrGLMVec3 >  normals;
std::vector< vrGLMVec2 > texCoords;
std::vector< vrGLMVec4 > tangents;
std::vector<vrVec3I> facesVec3I;
std::vector<vrInt> facesVec3I_group;
vrFloat m_maxDiameter,m_translation_x,m_translation_y,m_translation_z;
vrFloat xmin = FLT_MAX, xmax = -FLT_MAX, ymin = FLT_MAX, ymax = -FLT_MAX, zmin = FLT_MAX, zmax = -FLT_MAX;


const vrFloat    EPSINON = 0.000001f;
bool isZero(vrFloat var)
{
	if(var < EPSINON  && var > -EPSINON)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool isEqual(vrFloat var1,vrFloat var2,vrFloat t = EPSINON)
{
	const vrFloat positive = fabs(t);
	const vrFloat negative = -1.0f * positive;

	if ( (var1 - var2) < positive && (var1 - var2) > negative )
	{
		return true;
	}
	else
	{
		return false;
	}
}

void trimString( vrString & str ) 
{
	char const * whiteSpace = " \t\n\r";
	vrSizt_t location;
	location = str.find_first_not_of(whiteSpace);
	str.erase(0,location);
	location = str.find_last_not_of(whiteSpace);
	str.erase(location + 1);
}

vrFloat minIn3(vrFloat a, vrFloat b, vrFloat c)
{
	return (a < b) ? 
		( a < c ? a : c ) :
		( b < c ? b : c);
}

vrFloat maxIn3(vrFloat a, vrFloat b, vrFloat c)
{
	return (a > b) ? 
		(a > c ? a : c) : 
		(b > c ? b : c);
}

void unitMesh()
{
	using namespace std;
	//normalize
	vrFloat rx, ry, rz;
	

	std::vector< vrGLMVec3 > normalizePoints;
	vector<vrGLMVec3> &vertices = points;
	vector<vrGLMVec3> &normalizeVertices = normalizePoints;
	const long verticesNum = vertices.size();
	for (vrInt i = 0; i < verticesNum; ++i)
	{
		rx = vertices[i][0];
		if (xmin > rx)
		{
			xmin = rx;
		}
		if (xmax < rx)
		{
			xmax = rx;
		}

		ry = vertices[i][1];
		if (ymin > ry)
		{
			ymin = ry;
		}
		if (ymax < ry)
		{
			ymax = ry;
		}

		rz = vertices[i][2];
		if (zmin > rz)
		{
			zmin = rz;
		}
		if (zmax < rz)
		{
			zmax = rz;
		}

	}

	printf("x : %f,%f \t  y : %f,%f \t  z : %f,%f \t",xmax,xmin,ymax,ymin,zmax,zmin);vrPause;
	return;
	//MyPause;

	vrFloat maxDiameter, minDiameter, maxRadius, minRadius, X_diameter(xmax - xmin), Y_diameter(ymax - ymin), Z_diameter(zmax - zmin);
	vrFloat X_radius(X_diameter / 2.f), Y_radius(Y_diameter / 2.f), Z_radius(Z_diameter / 2.f);
	vrFloat massCenter_x(xmin + X_diameter / 2.f), massCenter_y(ymin + Y_diameter / 2.f), massCenter_z(zmin + Z_diameter / 2.f);

	maxDiameter = maxIn3(X_diameter	, Y_diameter, Z_diameter);
	minDiameter = minIn3(X_diameter	, Y_diameter, Z_diameter);
	maxRadius   = maxIn3(X_radius	, Y_radius	, Z_radius);
	minRadius   = minIn3(X_radius	, Y_radius	, Z_radius);

	vrFloat translation_x(maxRadius - massCenter_x), translation_y(maxRadius - massCenter_y), translation_z(maxRadius - massCenter_z);


	vrFloat x_scale = 1.0 / (maxDiameter);
	vrFloat y_scale = 1.0 / (maxDiameter);
	vrFloat z_scale = 1.0 / (maxDiameter);
	printf("maxDiameter is %f translation_x(%f,%f,%f) \n",maxDiameter,translation_x, translation_y, translation_z);
	m_maxDiameter = maxDiameter;
	m_translation_x = translation_x; 
	m_translation_y = translation_y;
	m_translation_z = translation_z;

	normalizeVertices.clear();
	for (vrInt j = 0; j< vertices.size(); ++j)
	{
		vrGLMVec3& refPoint = vertices[j];
		normalizeVertices.push_back(vrGLMVec3(x_scale * (refPoint[0] + translation_x) - 0.5f,
			y_scale * (refPoint[1] + translation_y) -0.5f,
			z_scale * (refPoint[2] + translation_z) -0.5f));
	}

	//Q_ASSERT(g_strMeshId == std::string("armadillo"));
	points = normalizePoints;
}

void loadOBJ(vrLpsz lpszFileName, bool loadTex)
{
	//reCenterMesh(center), loadTex(loadTc), genTang(genTangents)
	using namespace std;

	

	std::string fileName = vrString(lpszFileName);
	vrInt nFaces = 0;

	std::ifstream objStream( fileName, std::ios::in );

	if( !objStream ) {
		std::cout << "Unable to open OBJ file: " << fileName;
		exit(0);
	}

	vrString line, token;
	std::vector< vrInt > face;

	vrInt g = -1;

	getline( objStream, line );
	while( !objStream.eof() ) {
		trimString(line);
		if( line.length( ) > 0 && line.at(0) != '#' ) {
			istringstream lineStream( line );

			lineStream >> token;

			if (token == "v" ) {
				vrFloat x, y, z;
				lineStream >> x >> y >> z;
				points.push_back(vrGLMVec3(x, y, z));
			} else if (token == "vt" && loadTex) {
				// Process texture coordinate
				vrFloat s, t;
				lineStream >> s >> t;
				texCoords.push_back(vrGLMVec2(s, t));
			} else if (token == "vn" ) {
				vrFloat x, y, z;
				lineStream >> x >> y >> z;
				normals.push_back(vrGLMVec3(x, y, z));
			} else if (token == "g" ){
				g++;
			} else if (token == "f" ) {
				nFaces++;

				// Process face
				face.clear();
				size_t slash1, slash2;
				//int point, texCoord, normal;
				while( lineStream.good() ) {
					string vertString;
					lineStream >> vertString;
					vrInt pIndex = -1, nIndex = -1 , tcIndex = -1;

					slash1 = vertString.find("/");
					if( slash1 == string::npos ){
						pIndex = atoi( vertString.c_str() ) - 1;
					} else {
						slash2 = vertString.find("/", slash1 + 1 );
						pIndex = atoi( vertString.substr(0,slash1).c_str() )
							- 1;
						if( slash2 > slash1 + 1 ) {
							tcIndex =
								atoi( vertString.substr(slash1 + 1, slash2).c_str() )
								- 1;
						}
						nIndex =
							atoi( vertString.substr(slash2 + 1,vertString.length()).c_str() )
							- 1;
					}
					if( pIndex == -1 ) {
						printf("Missing point index!!!");
					} else {
						face.push_back(pIndex);
					}

					if( loadTex && tcIndex != -1 && pIndex != tcIndex ) {
						printf("Texture and point indices are not consistent.\n");
					}
					if ( nIndex != -1 && nIndex != pIndex ) {
						printf("Normal and point indices are not consistent.\n");
					}
				}
				// If number of edges in face is greater than 3,
				// decompose into triangles as a triangle fan.
				if( face.size() > 3 ) {
					printf("face.size() > 3 nFaces(%d)\n",nFaces);
				} else {
					vrVec3I node;
					node[0] = face[0];
					node[1] = face[1];
					node[2] = face[2];
					printf("face [%d]:[%d,%d,%d]\n",nFaces,node[0],node[1],node[2]);
					facesVec3I.push_back(node);
					facesVec3I_group.push_back(g);
				}
			}
		}
		getline( objStream, line );
	}

	objStream.close();

	unitMesh();
}

double middleTri_Vertex[24][3]={{-0.200000,0.066667,0.052632},{-0.200000,0.200000,0.052632},{-0.200000,-0.066667,0.052632},{-0.200000,-0.200000,0.052632},{-0.200000,0.066667,-0.052632},{-0.200000,0.200000,-0.052632},{-0.200000,-0.066667,-0.052632},{-0.200000,-0.200000,-0.052632},{0.200000,0.066667,0.052632},{0.200000,0.200000,0.052632},{0.200000,-0.066667,0.052632},{0.200000,-0.200000,0.052632},{0.200000,0.066667,-0.052632},{0.200000,0.200000,-0.052632},{0.200000,-0.066667,-0.052632},{0.200000,-0.200000,-0.052632},{0.066667,-0.200000,0.052632},{-0.066667,-0.200000,0.052632},{0.066667,-0.200000,-0.052632},{-0.066667,-0.200000,-0.052632},{0.066667,0.200000,0.052632},{-0.066667,0.200000,0.052632},{0.066667,0.200000,-0.052632},{-0.066667,0.200000,-0.052632}};
int middleTri[24][3]={{5,1,2},{5,2,6},{7,3,1},{7,1,5},{8,4,3},{8,3,7},{13,14,10},{13,10,9},{15,13,9},{15,9,11},{16,15,11},{16,11,12},{19,16,12},{19,12,17},{20,19,17},{20,17,18},{8,20,18},{8,18,4},{23,21,10},{23,10,14},{24,22,21},{24,21,23},{6,2,22},{6,22,24}};
int main()
{
	loadOBJ("D:/myDualBEM3D/Release/mesh/beam4BEM_test_for_discontinuous.obj",false);
	std::vector< vrVec3I > groupFace[4];
	for (int f=0; f < facesVec3I.size(); ++f)
	{
		const vrGLMVec3& v0 = points[facesVec3I[f][0]];
		const vrGLMVec3& v1 = points[facesVec3I[f][1]];
		const vrGLMVec3& v2 = points[facesVec3I[f][2]];

		if (isEqual(v0[0],0.2) && isEqual(v1[0],0.2) &&isEqual(v2[0],0.2)   )
		{
			groupFace[0].push_back( vrVec3I( facesVec3I[f][0], facesVec3I[f][1], facesVec3I[f][2]) );
		}

		if (isEqual(v0[0],-0.2) && isEqual(v1[0],-0.2) &&isEqual(v2[0],-0.2)   )
		{
			groupFace[1].push_back( vrVec3I( facesVec3I[f][0], facesVec3I[f][1], facesVec3I[f][2]) );
		}

		if (isEqual(v0[1], 0.2) && isEqual(v1[1], 0.2) &&isEqual(v2[1], 0.2)   )
		{
			groupFace[2].push_back( vrVec3I( facesVec3I[f][0], facesVec3I[f][1], facesVec3I[f][2]) );
		}

		if (isEqual(v0[1], -0.2) && isEqual(v1[1], -0.2) &&isEqual(v2[1], -0.2)   )
		{
			groupFace[3].push_back( vrVec3I( facesVec3I[f][0], facesVec3I[f][1], facesVec3I[f][2]) );
		}
	}

	std::set< vrGLMVec3 ,vec3LessCompare > endGroup[4];
	for (int g=0;g<4;++g)
	{
		for(int f=0; f< groupFace[g].size();++f)
		{			
			endGroup[g].insert(points[groupFace[g][f][0]]);
			endGroup[g].insert(points[groupFace[g][f][1]]);
			endGroup[g].insert(points[groupFace[g][f][2]]);
		}
	}

	points.clear();
	normals.clear();
	texCoords.clear();
	tangents.clear();
	facesVec3I.clear();
	facesVec3I_group.clear();

#if 0
	std::ofstream outfile("d:/disconti.obj");
	for (int v=0;v<points.size();++v)
	{
		outfile << "v " << points[v][0] << " " << points[v][1] <<  " " << points[v][2] << std::endl;
	}

	for (int g=0;g<4;++g)
	{
		outfile << "g sub_"<<g <<std::endl;

		for(int f=0; f< groupFace[g].size();++f)
		{
			outfile << "f " << groupFace[g][f][0]+1 << " "  << groupFace[g][f][1]+1 << " "   << groupFace[g][f][2]+1 << std::endl;
		}
	}
	outfile.close();
	return 0;
#endif

	std::set< vrGLMVec3 ,vec3LessCompare > setMiddleVertex;
	for (int v=0;v<24;++v)
	{
		setMiddleVertex.insert(vrGLMVec3(middleTri_Vertex[v][0],middleTri_Vertex[v][1],middleTri_Vertex[v][2]));
	}

	loadOBJ("D:/myDualBEM3D/Release/mesh/beam4BEM.obj",false);
	//return 0;
	/*

	std::vector< vrGLMVec3 >  points;
	std::vector< vrGLMVec3 >  normals;
	std::vector< vrGLMVec2 > texCoords;
	std::vector< vrGLMVec4 > tangents;
	std::vector<vrVec3I> facesVec3I;
	std::vector<vrInt> facesVec3I_group;

	*/
#if 1
	printf("facesVec3I.size(%d)\n",facesVec3I.size()); 

	facesVec3I_group.clear();
	facesVec3I_group.assign(facesVec3I.size(),2);

	for (int f=0;f<facesVec3I.size();++f)
	{
		const vrVec3I refFace = facesVec3I[f];

		bool match_boundary = true;
		bool match_tract = true;
		bool match_middle = true;

		bool match_end[4]={true,true,true,true};
		for (int v=0;v<3;++v)
		{
			if (!isEqual(zmin,points[refFace[v]][2]))
			{
				match_boundary = false;
			}

			if (!isEqual(zmax,points[refFace[v]][2]))
			{
				match_tract = false;
			}
			if (0 == setMiddleVertex.count(points[refFace[v]]))
			{
				match_middle = false;
			}

			for (int e=0;e<4;++e)
			{
				if (0 == endGroup[e].count(points[refFace[v]]))
				{
					match_end[e] = false;
				}
			}
		}
		if (match_boundary)
		{
			facesVec3I_group[f] = 0;
		}
		if (match_tract)
		{
			facesVec3I_group[f] = 1;
		}
		if (match_middle)
		{
			facesVec3I_group[f] = 3;
		}

		if (match_end[0])
		{
			facesVec3I_group[f] = 4;
		}
		if (match_end[1])
		{
			facesVec3I_group[f] = 5;
		}
		if (match_end[2])
		{
			facesVec3I_group[f] = 6;
		}
		if (match_end[3])
		{
			facesVec3I_group[f] = 7;
		}
	}
#else
	printf("facesVec3I.size(%d)\n",facesVec3I.size()); 
	vrInt nBndSize = 0;
	for (int f=0;f<facesVec3I.size();++f)
	{
		const vrVec3I& refFaces = facesVec3I[f];
		vrInt nCount = 0;
		for (int v=0;v<3;++v)
		{
			const vrGLMVec3& refVtx = points[refFaces[v]];

			if (isZero(refVtx[2]))
			{
				nCount++;
			}
		}
		if (3 == nCount)
		{
			facesVec3I_group[f] = 3;
			nBndSize++;
		}
	}
	printf("nBndSize %d\n",nBndSize);system("pause");
	nBndSize = 0;
	for (int f=0;f<facesVec3I.size();++f)
	{
		const vrVec3I& refFaces = facesVec3I[f];
		vrInt nCount = 0;
		for (int v=0;v<3;++v)
		{
			const vrGLMVec3& refVtx = points[refFaces[v]];

			if ( (isEqual(refVtx[1],0.125) ) && (refVtx[2] > (3.0*0.125/4.0)) )
			{
				nCount++;
			}
		}
		if (3 == nCount)
		{
			facesVec3I_group[f] = 4;
			nBndSize++;
		}
	}
	printf("nBndSize %d\n",nBndSize);system("pause");

	nBndSize = 0;
	for (int f=0;f<facesVec3I.size();++f)
	{
		const vrVec3I& refFaces = facesVec3I[f];
		vrInt nCount = 0;
		for (int v=0;v<3;++v)
		{
			const vrGLMVec3& refVtx = points[refFaces[v]];

			if ( (isEqual(refVtx[1],0.0)) && (refVtx[2] > (3.0*0.125/4.0)) )
			{
				nCount++;
			}
		}
		if (3 == nCount)
		{
			facesVec3I_group[f] = 5;
			nBndSize++;
		}
	}
	printf("nBndSize %d\n",nBndSize);system("pause");
#endif

	std::ofstream outfile_nodes("D:/myDualBEM3D/Release/mesh/beam4BEM_DisContinuous_Debug.obj.nodes");
	for (int v=0;v<points.size();++v)
	{
		outfile_nodes << (v+1) << " " << -1 << " " /*<< std::setprecision(10)*/ << points[v][0] << " " << points[v][1] << " " << points[v][2] << std::endl;
	}
	outfile_nodes.close();

	std::ofstream outfile_elems("D:/myDualBEM3D/Release/mesh/beam4BEM_DisContinuous_Debug.obj.elements");

	for (int f=0;f<facesVec3I.size();++f)
	{
		outfile_elems << (f+1)  << " " << facesVec3I_group[f]+1 << " 303 " << facesVec3I[f][0]+1 << " "  << facesVec3I[f][1]+1 << " " << facesVec3I[f][2]+1 << std::endl;
	}
	outfile_elems.close();
	return 0;
}