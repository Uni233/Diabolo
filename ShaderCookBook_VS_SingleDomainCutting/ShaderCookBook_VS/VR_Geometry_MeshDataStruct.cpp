#include "VR_Geometry_MeshDataStruct.h"
#include <iostream>
#include <fstream>
#include <VR_MACRO.h>
namespace YC
{
	namespace Geometry
	{
		void trimString( std::string & str ) 
		{
			const char * whiteSpace = " \t\n\r";
			size_t location;
			location = str.find_first_not_of(whiteSpace);
			str.erase(0,location);
			location = str.find_last_not_of(whiteSpace);
			str.erase(location + 1);
		}

		float minIn3(float a, float b, float c)
		{
			return (a < b) ? 
				( a < c ? a : c ) :
				( b < c ? b : c);
		}

		float maxIn3(float a, float b, float c)
		{
			return (a > b) ? 
				(a > c ? a : c) : 
				(b > c ? b : c);
		}

		void MeshDataStruct::loadOBJ(const char* lpszFileName, bool loadTex)
		{
			//reCenterMesh(center), loadTex(loadTc), genTang(genTangents)
			using namespace std;

			vector <vec3> normalizePoints;

			fileName = std::string(lpszFileName);
			int nFaces = 0;

			std::ifstream objStream( fileName, std::ios::in );

			if( !objStream ) {
				cerr << "Unable to open OBJ file: " << fileName << endl;
				exit(1);
			}

			string line, token;
			vector<int> face;

			getline( objStream, line );
			while( !objStream.eof() ) {
				trimString(line);
				if( line.length( ) > 0 && line.at(0) != '#' ) {
					istringstream lineStream( line );

					lineStream >> token;

					if (token == "v" ) {
						float x, y, z;
						lineStream >> x >> y >> z;
						points.push_back( vec3(x,y,z) );
					} else if (token == "vt" && loadTex) {
						// Process texture coordinate
						float s,t;
						lineStream >> s >> t;
						texCoords.push_back( vec2(s,t) );
					} else if (token == "vn" ) {
						float x, y, z;
						lineStream >> x >> y >> z;
						normals.push_back( vec3(x,y,z) );
					} else if (token == "f" ) {
						nFaces++;

						// Process face
						face.clear();
						size_t slash1, slash2;
						//int point, texCoord, normal;
						while( lineStream.good() ) {
							string vertString;
							lineStream >> vertString;
							int pIndex = -1, nIndex = -1 , tcIndex = -1;

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
							int v0 = face[0];
							int v1 = face[1];
							int v2 = face[2];
							// First face
							faces.push_back(v0);
							faces.push_back(v1);
							faces.push_back(v2);
							for( unsigned i = 3; i < face.size(); i++ ) {
								v1 = v2;
								v2 = face[i];
								faces.push_back(v0);
								faces.push_back(v1);
								faces.push_back(v2);
							}
						} else {
							faces.push_back(face[0]);
							faces.push_back(face[1]);
							faces.push_back(face[2]);
						}
					}
				}
				getline( objStream, line );
			}

			objStream.close();

			//if (unify)
			{
				//normalize
				float rx,ry,rz;
				float xmin = FLT_MAX ,xmax = -FLT_MAX ,ymin = FLT_MAX ,ymax = -FLT_MAX ,zmin = FLT_MAX ,zmax = -FLT_MAX ;

				vector<vec3> &vertices = points;
				vector<vec3> &normalizeVertices = normalizePoints;
				const long verticesNum = vertices.size();
				for (int i = 0; i < verticesNum; ++i)
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

				float maxDiameter,minDiameter,maxRadius,minRadius,X_diameter(xmax - xmin),Y_diameter(ymax - ymin),Z_diameter(zmax - zmin);
				float X_radius(X_diameter/2.f),Y_radius(Y_diameter/2.f),Z_radius(Z_diameter/2.f);
				float massCenter_x(xmin + X_diameter / 2.f),massCenter_y(ymin + Y_diameter/2.f),massCenter_z(zmin + Z_diameter/2.f);

				maxDiameter = maxIn3(X_diameter	, Y_diameter, Z_diameter);
				minDiameter = minIn3(X_diameter	, Y_diameter, Z_diameter);
				maxRadius   = maxIn3(X_radius	, Y_radius	, Z_radius);
				minRadius   = minIn3(X_radius	, Y_radius	, Z_radius);

				float translation_x(maxRadius-massCenter_x), translation_y(maxRadius-massCenter_y), translation_z(maxRadius-massCenter_z);


				float x_scale = 1.0 / (maxDiameter) ;
				float y_scale = 1.0 / (maxDiameter) ;
				float z_scale = 1.0 / (maxDiameter) ;
				printf("maxDiameter is %f translation_x(%f,%f,%f) \n",maxDiameter,translation_x, translation_y, translation_z);
				m_maxDiameter = maxDiameter;
				m_translation_x = translation_x; 
				m_translation_y = translation_y;
				m_translation_z = translation_z;

				normalizeVertices.clear();
				for (int j=0;j< vertices.size() ; ++j)
				{
					vec3& refPoint = vertices[j];
					normalizeVertices.push_back(vec3( x_scale * (refPoint[0]+ translation_x) -0.5f,
						y_scale * (refPoint[1] + translation_y) -0.5f,
						z_scale * (refPoint[2] + translation_z) -0.5f));
				}

				//Q_ASSERT(g_strMeshId == std::string("armadillo"));
				points = normalizePoints;
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
		}

		void MeshDataStruct::printMeshDataStructInfo(std::ostream& out)
		{
			out << "Mesh Information :" << std::endl;
			out << "Loaded mesh from: " << fileName.c_str() << std::endl;
			out << " " << points.size() << " points" << std::endl;
			out << " " << faces.size() / 3 << " triangles." << std::endl;
			out << " " << normals.size() << " normals" << std::endl;
			out << " " << tangents.size() << " tangents " << std::endl;
			out << " " << texCoords.size() << " texture coordinates." << std::endl;
		}

		vec3 MeshDataStruct::calculateNormal(const vec3& vtx0,const vec3& vtx1,const vec3& vtx2)
		{
			/* calculate Vector1 and Vector2 */			
			float val;
			vec3 va,vb,vr;
			va = vtx0 - vtx1;

			vb = vtx0 - vtx2;

			/* cross product */
			vr = glm::normalize(glm::cross(va,vb));
			return vr;
		}

		
		void MeshDataStruct::loadPLY(const char* lpszFileName)
		{
			using namespace std;

			vector <vec3> normalizePoints;

			fileName = std::string(lpszFileName);
			int nFaces = 0;

			std::ifstream objStream( fileName, std::ios::in );

			if( !objStream ) {
				cerr << "Unable to open PLY file: " << fileName << endl;
				exit(1);
			}

			string line, token;
			vector<int> face;
			char buffer[1024];
			int TotalConnectedPoints;
			int TotalFaces;
			bool loadTex = false;
			/*
			ply
			format ascii 1.0
			element vertex 35947
			property float x
			property float y
			property float z
			property float w
			element face 69451
			property list int int vertex_indices
			end_header
			*/

			do 
			{
				getline( objStream, line );
			} while (strncmp( "ply", line.c_str(),strlen("ply")) != 0);

			do 
			{
				getline( objStream, line );
			} while (strncmp( "element vertex", line.c_str(),strlen("element vertex")) != 0);

			strcpy(buffer, line.c_str()+strlen("element vertex"));
			sscanf(buffer,"%i", &TotalConnectedPoints);

			do 
			{
				getline( objStream, line );
			} while (strncmp( "element face", line.c_str(),strlen("element face")) != 0);
			
			strcpy(buffer, line.c_str()+strlen("element face"));
			sscanf(buffer,"%i", &TotalFaces);
			LogInfo("PLY Faces : %d\n",TotalFaces);
			MyPause;
			do 
			{
				getline( objStream, line );
			} while (strncmp( "end_header", line.c_str(),strlen("end_header")) != 0);
			
			float x, y, z;
			points.clear();
			normals.clear();
			std::vector< int > vecNormalCount;
			for (int iterator = 0; iterator < TotalConnectedPoints; iterator++)
			{
				getline( objStream, line );
				trimString(line);
				istringstream lineStream( line );
				lineStream >> x >> y >> z;
				points.push_back( vec3(x,y,z) );
				normals.push_back(vec3(0.f,0.f,0.f));
				vecNormalCount.push_back(0);
			}

			int v[4];
			vec3 curNormal;
			faces.clear();
			for (int iterator = 0; iterator < TotalFaces; iterator++)
			{
				getline( objStream, line );
				trimString(line);
				istringstream lineStream( line );
				lineStream >> v[0] >> v[1] >> v[2] >> v[3];
				Q_ASSERT(3 == v[0]);
				faces.push_back(v[1]);faces.push_back(v[2]);faces.push_back(v[3]);

				curNormal = calculateNormal(points[v[1]],points[v[2]],points[v[3]]);

				normals[v[1]] += curNormal;vecNormalCount[v[1]]++;
				normals[v[2]] += curNormal;vecNormalCount[v[2]]++;
				normals[v[3]] += curNormal;vecNormalCount[v[3]]++;
			}
			//printf("faces.size() %d, totalfaces %d\n",faces.size(),3*TotalFaces);
			Q_ASSERT(faces.size() == 3*TotalFaces);

			for (int iterator = 0; iterator < TotalConnectedPoints; iterator++)
			{
				normals[iterator] = glm::normalize(normals[iterator]);
			}

			objStream.close();

			//if (unify)
			{
				//normalize
				float rx,ry,rz;
				float xmin = FLT_MAX ,xmax = -FLT_MAX ,ymin = FLT_MAX ,ymax = -FLT_MAX ,zmin = FLT_MAX ,zmax = -FLT_MAX ;

				vector<vec3> &vertices = points;
				vector<vec3> &normalizeVertices = normalizePoints;
				const long verticesNum = vertices.size();
				for (int i = 0; i < verticesNum; ++i)
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

				float maxDiameter,minDiameter,maxRadius,minRadius,X_diameter(xmax - xmin),Y_diameter(ymax - ymin),Z_diameter(zmax - zmin);
				float X_radius(X_diameter/2.f),Y_radius(Y_diameter/2.f),Z_radius(Z_diameter/2.f);
				float massCenter_x(xmin + X_diameter / 2.f),massCenter_y(ymin + Y_diameter/2.f),massCenter_z(zmin + Z_diameter/2.f);

				maxDiameter = maxIn3(X_diameter	, Y_diameter, Z_diameter);
				minDiameter = minIn3(X_diameter	, Y_diameter, Z_diameter);
				maxRadius   = maxIn3(X_radius	, Y_radius	, Z_radius);
				minRadius   = minIn3(X_radius	, Y_radius	, Z_radius);

				float translation_x(maxRadius-massCenter_x), translation_y(maxRadius-massCenter_y), translation_z(maxRadius-massCenter_z);


				float x_scale = 1.0 / (maxDiameter) ;
				float y_scale = 1.0 / (maxDiameter) ;
				float z_scale = 1.0 / (maxDiameter) ;
				printf("maxDiameter is %f translation_x(%f,%f,%f) \n",maxDiameter,translation_x, translation_y, translation_z);
				m_maxDiameter = maxDiameter;
				m_translation_x = translation_x; 
				m_translation_y = translation_y;
				m_translation_z = translation_z;

				normalizeVertices.clear();
				for (int j=0;j< vertices.size() ; ++j)
				{
					vec3& refPoint = vertices[j];
					normalizeVertices.push_back(vec3( x_scale * (refPoint[0]+ translation_x) -0.5f,
						y_scale * (refPoint[1] + translation_y) -0.5f,
						z_scale * (refPoint[2] + translation_z) -0.5f));
				}

				//Q_ASSERT(g_strMeshId == std::string("armadillo"));
				points = normalizePoints;
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
		}
	}
}