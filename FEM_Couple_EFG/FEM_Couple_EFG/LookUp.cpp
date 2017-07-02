#include "stdafx.h"
#include <sstream>
#include <time.h>
#include "VR_Global_Define.h"
#include "Configure/INIReader.h"
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
//#include "MeshGenerate.h"
#include "ElasticSimulation_Couple.h"
#include "plane.h"

#include "Maths/VECTOR3D.h"
#include "Maths/VECTOR4D.h"
#include "Maths/MATRIX4X4.h"
#include "Maths/COLOR.h"

std::string g_strCurrentPath;
std::string s_qstrCurrentTime;
std::string s_qstrCurrentTimeStamp;

/************************************************************************/
/* Configure data                                                       */
/************************************************************************/
bool g_hasCoords=false;
bool g_hasVerticeNormal=false;
bool g_normalizeMesh=true;
std::string g_strMeshName;
std::string g_strMeshId;
std::string g_strTextureName;
std::vector< VR_FEM::MyPoint > g_vecMaterialPoint;
std::vector< float > g_vecMaterialPointStiff;

////////////////////////////////////////////////////shadow map ////////////////////////////////////
GLuint shadowMapTexture;
const int shadowMapSize=512*2;
//Matrices
MATRIX4X4 lightProjectionMatrix, lightViewMatrix;
MATRIX4X4 cameraProjectionMatrix, cameraViewMatrix;
VECTOR3D cameraPosition(0.0f, .5f,5.0f);
VECTOR3D lightPosition(3.0f, 3.0f,3.0f);

////////////////////////////////////////////////////shadow map ////////////////////////////////////

//std::vector< VR_FEM::Plane >  vecPlane;
//VR_FEM::MeshGenerate mesh(CellRaidus,24,20,4,vecPlane);
VR_FEM::ElasticSimulation_Couple mesh;

extern std::string GetModuleDir();

extern float rotate_x;
extern float rotate_y;
extern float translate_z;
extern float scaled;

void do_something()
{

	std::string strObjFilePath = g_strCurrentPath + std::string("\\data\\") + g_strMeshName;
	std::string strTexturePath = g_strCurrentPath + std::string("\\data\\") + g_strTextureName;
	mesh.parserObj(g_hasCoords,g_hasVerticeNormal,strObjFilePath.c_str());
	mesh.LoadGLTextures(strTexturePath.c_str());
	mesh.loadOctreeNode_Beam(g_strMeshId.c_str());
	mesh.createOutBoundary();
	mesh.generateCoupleDomain();
	
	mesh.distributeDof_global();
	mesh.distributeDof_local();
	mesh.distributeDof_Couple();
	//mesh.createOutBoundary_Force();
	mesh.createInnerBoundary();
	//mesh.createInnerBoundary_Force();
	for (VR_FEM::MyInt v=0;v<VR_FEM::Cell::LocalDomainCount;++v)
	{
		mesh.createForceBoundaryCondition(v);
		mesh.createGlobalMassAndStiffnessAndDampingMatrixFEM(v);
		mesh.createNewMarkMatrix(v);
		mesh.createDCBoundaryCondition(v);
		
	}
	for (VR_FEM::MyInt v=0;v<VR_FEM::Cell::CoupleDomainCount;++v)
	{
		mesh.createGlobalMassAndStiffnessAndDampingMatrixEFG(v);
		mesh.createNewMarkMatrix_Couple_EFG(v);
	}
	mesh.setSimulate(true);
	//mesh.simulation();
	//MyPause;
}

void DrawScene()
{
	mesh.print();
	
}

void myidle()
{
	mesh.simulation_Couple_FEM_EFG();
	glutPostRedisplay();
	//MyPause;
}

void display()
{
	static GLuint baseList=0;
	if(!baseList)
	{
		baseList=glGenLists(1);
		glNewList(baseList, GL_COMPILE);
		{
			glColor3f(248.0f / 255.0f, 161.0f/255.0f, 82.0f/255.0f);
			glPushMatrix();

			glTranslatef(0.0f, -0.5f, 0.0f);
			//glScalef(2.0f, 0.01f, 2.0f);
			glScalef(3.0f, 0.01f, 3.0f);
			glutSolidCube(3.0f);

			glPopMatrix();
		}
		glEndList();
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	/*glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();*/
	
	
	static unsigned nTimeStep=0;
	//mesh.simulation();
	//mesh.simulation_Couple_FEM_EFG();
	//mesh.solve_timestep_steak(nTimeStep++);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	//
	{
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(lightProjectionMatrix);

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(lightViewMatrix);

		//Use viewport the same size as the shadow map
		glViewport(0, 0, shadowMapSize, shadowMapSize);

		//Draw back faces into the shadow map
		glCullFace(GL_FRONT);

		//Disable color writes, and use flat shading for speed
		glShadeModel(GL_FLAT);
		glColorMask(0, 0, 0, 0);

	}
	glCallList(baseList);
	DrawScene();

	{
		//Read the depth buffer into the shadow map texture
		glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, shadowMapSize, shadowMapSize);

		//restore states
		glCullFace(GL_BACK);
		glShadeModel(GL_SMOOTH);
		glColorMask(1, 1, 1, 1);




		//2nd pass - Draw from camera's point of view
		glClear(GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(cameraProjectionMatrix);

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(cameraViewMatrix);

		glViewport(0, 0, window_width, window_height);

		//Use dim light to represent shadowed areas
		glLightfv(GL_LIGHT1, GL_POSITION, VECTOR4D(lightPosition));
		glLightfv(GL_LIGHT1, GL_AMBIENT, white*0.2f);
		glLightfv(GL_LIGHT1, GL_DIFFUSE, white*0.2f);
		glLightfv(GL_LIGHT1, GL_SPECULAR, black);
		glEnable(GL_LIGHT1);
		glEnable(GL_LIGHTING);
	}
	glCallList(baseList);
	DrawScene();
	{
		//3rd pass
		//Draw with bright light
		glLightfv(GL_LIGHT1, GL_DIFFUSE, white);
		glLightfv(GL_LIGHT1, GL_SPECULAR, white);

		//Calculate texture matrix for projection
		//This matrix takes us from eye space to the light's clip space
		//It is postmultiplied by the inverse of the current view matrix when specifying texgen
		static MATRIX4X4 biasMatrix(0.5f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.5f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.5f, 0.0f,
			0.5f, 0.5f, 0.5f, 1.0f);	//bias from [-1, 1] to [0, 1]
		MATRIX4X4 textureMatrix=biasMatrix*lightProjectionMatrix*lightViewMatrix;

		//Set up texture coordinate generation.
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		glTexGenfv(GL_S, GL_EYE_PLANE, textureMatrix.GetRow(0));
		glEnable(GL_TEXTURE_GEN_S);

		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		glTexGenfv(GL_T, GL_EYE_PLANE, textureMatrix.GetRow(1));
		glEnable(GL_TEXTURE_GEN_T);

		glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		glTexGenfv(GL_R, GL_EYE_PLANE, textureMatrix.GetRow(2));
		glEnable(GL_TEXTURE_GEN_R);

		glTexGeni(GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		glTexGenfv(GL_Q, GL_EYE_PLANE, textureMatrix.GetRow(3));
		glEnable(GL_TEXTURE_GEN_Q);

		//Bind & enable shadow map texture
		glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
		glEnable(GL_TEXTURE_2D);

		//Enable shadow comparison
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE);

		//Shadow comparison should be true (ie not in shadow) if r<=texture
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL);

		//Shadow comparison should generate an INTENSITY result
		glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_INTENSITY);

		//Set alpha test to discard false comparisons
		glAlphaFunc(GL_GEQUAL, 0.99f);
		glEnable(GL_ALPHA_TEST);
	}
	glCallList(baseList);
	DrawScene();

	{
		//Disable textures and texgen
		glDisable(GL_TEXTURE_2D);

		glDisable(GL_TEXTURE_GEN_S);
		glDisable(GL_TEXTURE_GEN_T);
		glDisable(GL_TEXTURE_GEN_R);
		glDisable(GL_TEXTURE_GEN_Q);

		//Restore other states
		glDisable(GL_LIGHTING);
		glDisable(GL_ALPHA_TEST);
	}

	glutSwapBuffers();
	glutPostRedisplay();
}

void init_shadowMap()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//Shading states
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//Depth states
	glClearDepth(1.0f);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_CULL_FACE);

	//We use glScale when drawing the scene
	glEnable(GL_NORMALIZE);
	//Create the shadow map texture
	glGenTextures(1, &shadowMapTexture);
	glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
	glTexImage2D(	GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowMapSize, shadowMapSize, 0,
		GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	//Use the color as the ambient and diffuse material
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	//White specular material color, shininess 16
	glMaterialfv(GL_FRONT, GL_SPECULAR, white);
	glMaterialf(GL_FRONT, GL_SHININESS, 16.0f);

	glPushMatrix();

	glLoadIdentity();
	gluPerspective(60.0f, (float)window_width/window_height, .1f, 100.0f);
	glGetFloatv(GL_MODELVIEW_MATRIX, cameraProjectionMatrix);

	glLoadIdentity();
	gluLookAt(cameraPosition.x, cameraPosition.y, cameraPosition.z,
		0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f);
	glGetFloatv(GL_MODELVIEW_MATRIX, cameraViewMatrix);

	glLoadIdentity();
	gluPerspective(60.0f, 1.0f, 2.0f, 8.0f);
	glGetFloatv(GL_MODELVIEW_MATRIX, lightProjectionMatrix);

	glLoadIdentity();
	gluLookAt(	lightPosition.x, lightPosition.y, lightPosition.z,
		0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f);
	glGetFloatv(GL_MODELVIEW_MATRIX, lightViewMatrix);

	glPopMatrix();

	GLfloat	fogColor[4] = {0.6, 0.6, 0.6, 1.0};
	glEnable(GL_FOG);
	{
		glFogi (GL_FOG_MODE, GL_LINEAR);
		glFogfv (GL_FOG_COLOR, fogColor);
		glFogf (GL_FOG_START, 3.0);
		glFogf (GL_FOG_END,15.0);
		glHint (GL_FOG_HINT, GL_DONT_CARE);
		glClearColor(0.3, 0.3, 0.3, 1.0);
	}
}

bool initGL(int argc, char **argv)
{
	glutInit(&argc, (char**)argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);//(GLUT_RGBA | GLUT_DOUBLE| GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Yang chen Scene.");

	// initialize necessary OpenGL extensions
	glewInit();
	if (! glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(1.0, 1.0, 1.0, 1.0);
	//glClearColor(0.0, 0.0, 0.0, 1.0);

	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_NORMALIZE);
	//glEnable( GL_CULL_FACE );
	glDisable(GL_CULL_FACE);
	glEnable( GL_TEXTURE_2D );
	// viewport
	glViewport(0, 0, window_width, window_height);

	// set view matrix

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);


	{
		init_shadowMap();
	}

	return true;
}

void global_Initialize()
{
	{
		g_strCurrentPath = GetModuleDir();
	}
	{
		std::string s_qstrCurrentData;
		std::stringstream ss;
		time_t now;

		now = time(0);
		tm *tnow = localtime(&now);
		ss << (1900+tnow->tm_year) << "-" << (tnow->tm_mon+1) << "-" << (tnow->tm_mday);
		s_qstrCurrentData = (ss.str());
		ss.str("");
		ss << (tnow->tm_hour) << "-" << (tnow->tm_min) << "-" << (tnow->tm_sec);
		s_qstrCurrentTime = (ss.str());
		s_qstrCurrentTimeStamp = s_qstrCurrentData + std::string(_T("-")) + s_qstrCurrentTime;
	}
	{
		INIReader reader(std::string(g_strCurrentPath + "\\configure.ini").c_str());

		if (reader.ParseError() < 0) {
			std::cout << "Can't load 'configure.ini'\n";
			MyExit;
			return ;
		}
		std::cout << "Config loaded from 'configure.ini': hasCoord="
			<< reader.GetInteger("Obj", "hasCoord", -1) << ", normalizeMesh="
			<< reader.Get("Obj", "normalizeMesh", "UNKNOWN") << "\n"; 

		g_hasCoords = (0 != reader.GetInteger("Obj", "hasCoord", 0) );
		g_hasVerticeNormal = (0 != reader.GetInteger("Obj", "hasVerticeNormal", 0));
		g_normalizeMesh = (0 != reader.GetInteger("Obj", "normalizeMesh", 0));
		g_strMeshName = reader.Get("Obj", "meshName", "armadillo_1M_Normal.obj");
		g_strMeshId = reader.Get("Obj", "meshId", "armadillo");
		g_strTextureName = reader.Get("Obj", "textureName", "steak.png");

		int nSamplePointSize = reader.GetInteger("Material", "materialPointSize", 0);
		g_vecMaterialPoint.resize(nSamplePointSize);
		g_vecMaterialPointStiff.resize(nSamplePointSize);
		char tmpChar;
		std::stringstream ss;
		std::string ssContent;
		for (int v=0;v<nSamplePointSize;++v)
		{
			ss.str("");
			ss << "materialPoint_" << v;
			ssContent = reader.Get("Material", ss.str().c_str(), "");
			printf("[%s][%s]\n",ss.str().c_str(),ssContent.c_str());
			{
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					ssContent,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				g_vecMaterialPoint[v][0] = boost::lexical_cast<float>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				g_vecMaterialPoint[v][1] = boost::lexical_cast<float>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				g_vecMaterialPoint[v][2] = boost::lexical_cast<float>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
			}
			printf("{%f,%f,%f}[%f]\n",g_vecMaterialPoint[v][0],g_vecMaterialPoint[v][1],g_vecMaterialPoint[v][2],g_vecMaterialPointStiff[v]);
			ss.str("");
			ss << "materialStiff_" << v;
			g_vecMaterialPointStiff[v] = (float)reader.GetReal("Material",ss.str().c_str(),0.f);


		}
	}
}