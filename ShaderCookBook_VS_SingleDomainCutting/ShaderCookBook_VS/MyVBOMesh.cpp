#include "MyVBOMesh.h"
#include "glutils.h"
#include "cookbookogl.h"
#include "VR_MACRO.h"
#include <vector_types.h>
#include <helper_math.h>
#define uint unsigned int

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <fstream>
using std::ifstream;
#include <sstream>
using std::istringstream;

MyVBOMesh::MyVBOMesh(const char * fileName, bool reCenterMesh, bool loadTc, bool genTangents)
	:VBOMesh(fileName,reCenterMesh,loadTc,genTangents)
{
}


MyVBOMesh::~MyVBOMesh(void)
{
}

void MyVBOMesh::loadPLY( const char * fileName )
{
	m_objInfo.loadPLY(fileName);

	vector <vec3>& points = m_objInfo.points;
	vector <vec3>& normals = m_objInfo.normals;
	vector <vec2>& texCoords = m_objInfo.texCoords;
	vector <int>& faces = m_objInfo.faces;

	if( normals.size() == 0 ) {
		generateAveragedNormals(points,normals,faces);
	}

	vector<vec4> tangents;
	if( genTang && texCoords.size() > 0 ) {
		generateTangents(points,normals,faces,texCoords,tangents);
	}

	if( reCenterMesh ) {
		center(points);
	}

	storeVBO(points, normals, texCoords, tangents, faces);
}

void MyVBOMesh::loadOBJ( const char * fileName ) {

#if 0

	vector <vec3> points;
	vector <vec3> normalizePoints;
	vector <vec3> normals;
	vector <vec2> texCoords;
	vector <int> faces;

	int nFaces = 0;

	ifstream objStream( fileName, std::ios::in );

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
#else
	m_objInfo.loadOBJ(fileName,loadTex);

	vector <vec3>& points = m_objInfo.points;
	vector <vec3>& normals = m_objInfo.normals;
	vector <vec2>& texCoords = m_objInfo.texCoords;
	vector <int>& faces = m_objInfo.faces;
#endif

	if( normals.size() == 0 ) {
		generateAveragedNormals(points,normals,faces);
	}

	vector<vec4> tangents;
	if( genTang && texCoords.size() > 0 ) {
		generateTangents(points,normals,faces,texCoords,tangents);
	}

	if( reCenterMesh ) {
		center(points);
	}

	storeVBO(points, normals, texCoords, tangents, faces);
}

void MyVBOMesh::loadOBJ( YC::Geometry::MeshDataStruct& obj)
{
	m_objInfo = obj;

	vector <vec3>& points = m_objInfo.points;
	vector <vec3>& normals = m_objInfo.normals;
	vector <vec2>& texCoords = m_objInfo.texCoords;
	vector <int>& faces = m_objInfo.faces;

	if( normals.size() == 0 ) {
		generateAveragedNormals(points,normals,faces);
	}

	vector<vec4> tangents;
	if( genTang && texCoords.size() > 0 ) {
		generateTangents(points,normals,faces,texCoords,tangents);
	}

	if( reCenterMesh ) {
		center(points);
	}

	storeVBO(points, normals, texCoords, tangents, faces);
}

void MyVBOMesh::initialize(const int nPointSize, const int nNormalSize)
{
	nElementCount = 0;
	faces = 0;
	const int nVtxVnCount = nPointSize * _nExternalMemory;
	float3 * v = new float3[nVtxVnCount];
	memset(v,'\0',nVtxVnCount*sizeof(float3));

	float3 * n = new float3[nVtxVnCount];
	memset(n,'\0',nVtxVnCount*sizeof(float3));

	glGenVertexArrays( 1, &vaoHandle );
	glBindVertexArray(vaoHandle);

	int nBuffers = 2;
	uint bufIdx = 0;
	glGenBuffers(nBuffers, handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
	glBufferData(GL_ARRAY_BUFFER, nVtxVnCount * sizeof(float3), v, GL_STREAM_DRAW/*GL_STATIC_DRAW*/);
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(0);  // Vertex position

	glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
	glBufferData(GL_ARRAY_BUFFER, nVtxVnCount * sizeof(float3), n, GL_STREAM_DRAW/*GL_STATIC_DRAW*/);
	glVertexAttribPointer( (GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(1);  // Vertex normal

	glBindVertexArray(0);

	delete [] v;
	delete [] n;
}

void MyVBOMesh::storeVBO( const vector<vec3> & points,
	const vector<vec3> & normals,
	const vector<vec2> &texCoords,
	const vector<vec4> &tangents,
	const vector<int> &elements )
{
	int nVerts  = points.size();
	faces = elements.size() / 3;
	nElementCount = elements.size();
	LogInfo("faces (%d)  element (%d)\n",faces,elements.size());
#if 1
	const int nVtxVnCount = nElementCount * _nExternalMemory;
	float3 * v = new float3[nVtxVnCount];
	memset(v,'\0',nVtxVnCount*sizeof(float3));

	float3 * n = new float3[nVtxVnCount];
	memset(n,'\0',nVtxVnCount*sizeof(float3));

	float2 * tc = NULL;
	float4 * tang = NULL;

	for (int i=0;i<nElementCount;++i)
	{
		const int idx = elements[i];
		const vec3& refPoint = points[idx];
		const vec3& refNormal = normals[idx];
		v[i] = make_float3(refPoint.x,refPoint.y,refPoint.z);
		n[i] = make_float3(refNormal.x,refNormal.y,refNormal.z);
	}

	glGenVertexArrays( 1, &vaoHandle );
	glBindVertexArray(vaoHandle);

	int nBuffers = 2;
	uint bufIdx = 0;
	glGenBuffers(nBuffers, handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
	glBufferData(GL_ARRAY_BUFFER, nVtxVnCount * sizeof(float3), v, GL_STREAM_DRAW/*GL_STATIC_DRAW*/);
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(0);  // Vertex position

	glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
	glBufferData(GL_ARRAY_BUFFER, nVtxVnCount * sizeof(float3), n, GL_STREAM_DRAW/*GL_STATIC_DRAW*/);
	glVertexAttribPointer( (GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(1);  // Vertex normal

	glBindVertexArray(0);

	delete [] v;
	delete [] n;
#else

	float3 * v = new float3[ nVerts * _nExternalMemory];
	memset(v,'\0', nVerts * _nExternalMemory*sizeof(float3));LogInfo("v[%d]\n",(int)v);
	float3 * n = new float3 nVerts * _nExternalMemory];
	memset(n,'\0', nVerts * _nExternalMemory*sizeof(float3));LogInfo("n[%d]\n",(int)n);

	
	float2 * tc = NULL;
	float4 * tang = NULL;

	if(texCoords.size() > 0) {
		tc = new float2[  nVerts * _nExternalMemory];
		memset(tc,'\0', nVerts * _nExternalMemory*sizeof(float2));
		if( tangents.size() > 0 )
		{
			tang = new float4[nVerts * _nExternalMemory];
			memset(tang,'\0', nVerts * _nExternalMemory*sizeof(float4));
		}
	}

	/*unsigned int*/int3 *el = new int3[/*elements.size()*/faces * _nExternalMemory];
	memset(el,'\0',faces * _nExternalMemory*sizeof(/*unsigned int*/int3));LogInfo("el[%d]\n",(int)el);
	//MyPause;
	int idx = 0, tcIdx = 0, tangIdx = 0;
	for( int i = 0; i < nVerts; ++i )
	{
		v[idx].x = points[i].x;
		v[idx].y = points[i].y;
		v[idx].z = points[i].z;
		n[idx].x = normals[i].x;
		n[idx].y = normals[i].y;
		n[idx].z = normals[i].z;
		idx += 1;
		if( tc != NULL ) {
			tc[tcIdx].x = texCoords[i].x;
			tc[tcIdx].y = texCoords[i].y;
			tcIdx += 1;
		}
		if( tang != NULL ) {
			tang[tangIdx].x = tangents[i].x;
			tang[tangIdx].y = tangents[i].y;
			tang[tangIdx].z = tangents[i].z;
			tang[tangIdx].w = tangents[i].w;
			tangIdx += 1;
		}
	}
	for( unsigned int i = 0; i < faces/*elements.size()*/; ++i )
	{
		el[i].x = elements[i*3];
		el[i].y = elements[i*3+1];
		el[i].z = elements[i*3+2];
	}
	glGenVertexArrays( 1, &vaoHandle );
	glBindVertexArray(vaoHandle);

	int nBuffers = 3;
	if( tc != NULL ) nBuffers++;
	if( tang != NULL ) nBuffers++;
	GLuint elementBuffer = nBuffers - 1;


	uint bufIdx = 0;
	glGenBuffers(nBuffers, handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
	glBufferData(GL_ARRAY_BUFFER, (/*3 **/ nVerts * _nExternalMemory) * sizeof(float3), v, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(0);  // Vertex position

	glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
	glBufferData(GL_ARRAY_BUFFER, (/*3 **/ nVerts * _nExternalMemory) * sizeof(float3), n, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(1);  // Vertex normal

	if( tc != NULL ) {
		glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
		glBufferData(GL_ARRAY_BUFFER, (/*2 **/ nVerts * _nExternalMemory) * sizeof(float2), tc, GL_STATIC_DRAW);
		glVertexAttribPointer( (GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
		glEnableVertexAttribArray(2);  // Texture coords
	}
	if( tang != NULL ) {
		glBindBuffer(GL_ARRAY_BUFFER, handle[bufIdx++]);
		glBufferData(GL_ARRAY_BUFFER, (/*4 **/ nVerts * _nExternalMemory) * sizeof(float4), tang, GL_STATIC_DRAW);
		glVertexAttribPointer( (GLuint)3, 4, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
		glEnableVertexAttribArray(3);  // Tangent vector
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle[elementBuffer]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, /*3 * */faces * _nExternalMemory * sizeof(/*unsigned int*/int3), el, GL_STATIC_DRAW);
	glBindVertexArray(0);

	// Clean up
	delete [] v;
	delete [] n;
	if( tc != NULL ) delete [] tc;
	if( tang != NULL ) delete [] tang;
	delete [] el;
#endif
}

void MyVBOMesh::render() const {
	glBindVertexArray(vaoHandle);
	glDrawArrays(GL_TRIANGLES, 0, nElementCount);
	//glDrawElements(GL_TRIANGLES, 3 * faces, GL_UNSIGNED_INT, ((GLubyte *)NULL + (0)));
}