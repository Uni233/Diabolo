#include "VR_Geometry_MeshDataStruct.h"
#include <iostream>
#include <fstream>
#include "vrBase/vrLog.h"
#include "../bemMarco.h"
//#include <VR_MACRO.h>
namespace VR
{
	namespace Geometry
	{
		void trimString( vrString & str ) 
		{
			vrLpsz whiteSpace = " \t\n\r";
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

		void MeshDataStruct::clear()
		{
			points.clear();
			normals.clear();
			texCoords.clear();
			tangents.clear();
			/*faces.clear();
			faces_group.clear();*/
			displacedVertices.clear();
		}

		void MeshDataStruct::loadOBJ(vrLpsz lpszFileName, bool loadTex, bool unify)
		{
#if 0
			
			//reCenterMesh(center), loadTex(loadTc), genTang(genTangents)
			using namespace std;

			clear();

			fileName = vrString(lpszFileName);
			vrInt nFaces = 0;

			std::ifstream objStream( fileName, std::ios::in );

			if( !objStream ) {
				errorLog << "Unable to open OBJ file: " << fileName;
				vrExit;
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
							vrInt v0 = face[0];
							vrInt v1 = face[1];
							vrInt v2 = face[2];
							// First face
							faces.push_back(v0);faces_group.push_back(g);
							faces.push_back(v1);faces_group.push_back(g);
							faces.push_back(v2);faces_group.push_back(g);
							for( unsigned i = 3; i < face.size(); i++ ) {
								v1 = v2;
								v2 = face[i];
								faces.push_back(v0);faces_group.push_back(g);
								faces.push_back(v1);faces_group.push_back(g);
								faces.push_back(v2);faces_group.push_back(g);
							}
						} else {
							faces.push_back(face[0]);faces_group.push_back(g);
							faces.push_back(face[1]);faces_group.push_back(g);
							faces.push_back(face[2]);faces_group.push_back(g);
						}
					}
				}
				getline( objStream, line );
			}

			objStream.close();

			if (unify)
			{
				unitMesh();
			}

			/*if( normals.size() == 0 ) {
				generateAveragedNormals(points,normals,faces);
			}

			vector<vec4> tangents;
			if( genTang && texCoords.size() > 0 ) {
				generateTangents(points,normals,faces,texCoords,tangents);
			}

			if( reCenterMesh ) {
				center(points);
			}*/

			//storeVBO(points, normals, texCoords, tangents, faces);

			printMeshDataStructInfo(std::cout);
#endif
		}

		void MeshDataStruct::unitMesh()
		{
			using namespace std;
			//normalize
			vrFloat rx, ry, rz;
			vrFloat xmin = FLT_MAX, xmax = -FLT_MAX, ymin = FLT_MAX, ymax = -FLT_MAX, zmin = FLT_MAX, zmax = -FLT_MAX;

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

			printf("x : %f,%f \t  y : %f,%f \t  z : %f,%f \t",xmax,xmin,ymax,ymin,zmax,zmin);
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

		void MeshDataStruct::printMeshDataStructInfo(std::ostream& out)
		{
#if 0

			out << "Mesh Information :" << std::endl;
			out << "Loaded mesh from: " << fileName.c_str() << std::endl;
			out << " " << points.size() << " points" << std::endl;
			out << " " << faces.size() / 3 << " triangles." << std::endl;
			out << " " << normals.size() << " normals" << std::endl;
			out << " " << tangents.size() << " tangents " << std::endl;
			out << " " << texCoords.size() << " texture coordinates." << std::endl;

			std::map< int,int > groupInfo;
			for (iterAllOf(itr,faces_group))
			{
				vrInt gId = *itr;
				if (groupInfo.find(gId) == groupInfo.end())
				{
					groupInfo[gId] = 0;
				}

				groupInfo[gId]++;
			}

			for (iterAllOf(itr,groupInfo))
			{
				out << "group id : " << (*itr).first << ". face size : " << (*itr).second / 3 << std::endl;
			}

			Q_ASSERT(3 == groupInfo.size());
#endif
		}

		vrGLMVec3 MeshDataStruct::calculateNormal(const vrGLMVec3& vtx0, const vrGLMVec3& vtx1, const vrGLMVec3& vtx2)
		{
			/* calculate Vector1 and Vector2 */			
			vrFloat val;
			vrGLMVec3 va, vb, vr;
			va = vtx0 - vtx1;

			vb = vtx0 - vtx2;

			/* cross product */
			vr = glm::normalize(glm::cross(va,vb));
			return vr;
		}

	}
}