#pragma once
#include "vbomesh.h"
#include "VR_Geometry_MeshDataStruct.h"
class MyVBOMesh :
	public VBOMesh
{
public:
	MyVBOMesh(const char * fileName, bool reCenterMesh = false, bool loadTc = false, bool genTangents = false);
	~MyVBOMesh(void);

public:
	virtual void loadOBJ( const char * fileName );
	virtual void loadOBJ( YC::Geometry::MeshDataStruct& obj);
	virtual void loadPLY( const char * fileName );
	virtual const YC::Geometry::MeshDataStruct& getOBJInfo(){return m_objInfo;}
	virtual unsigned int getVBOHandle(int idx){return handle[idx];}
	virtual unsigned int getVAOHandle(){return vaoHandle;}
	virtual void setFaces(const int nFaces){faces = nFaces;}
	virtual void setElements(const int nElem){nElementCount = nElem;}
	virtual void render()const;
	virtual void storeVBO( const vector<vec3> & points,	const vector<vec3> & normals,	const vector<vec2> &texCoords,	const vector<vec4> &tangents,	const vector<int> &elements );
	virtual void initialize(const int nPointSize, const int nNormalSize);
private:
	YC::Geometry::MeshDataStruct m_objInfo;
	int nElementCount;//3*faces
	//float m_maxDiameter,m_translation_x,m_translation_y,m_translation_z;
};

