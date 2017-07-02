#include "StdAfx.h"
#include "MyScene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
using glm::vec3;

#include <GL/freeglut.h>
#include "TexRender.h"

#include "Rotation/arcball.h"

#include "SOIL.h"//save texture

#include "VR_GlobalVariable.h"

#include "VR_Geometry_MeshDataStruct.h"

Matrix4fT   MyTransform   = {  1.0f,  0.0f,  0.0f,  0.0f,				// NEW: Final Transform
	0.0f,  1.0f,  0.0f,  0.0f,
	0.0f,  0.0f,  1.0f,  0.0f,
	0.0f,  0.0f,  0.0f,  1.0f };

Matrix3fT   LastRot     = {  1.0f,  0.0f,  0.0f,					// NEW: Last Rotation
	0.0f,  1.0f,  0.0f,
	0.0f,  0.0f,  1.0f };

Matrix3fT   ThisRot     = {  1.0f,  0.0f,  0.0f,					// NEW: This Rotation
	0.0f,  1.0f,  0.0f,
	0.0f,  0.0f,  1.0f };

int myRed=115,myGreen=115,MyBlue=115;

const vec3 planeColor(173.f/255,128.f/255,74.f/255);
const vec3 modelColor(115.f/255,115.f/255,115.f/255);
const vec3 modelColor4Ka(vec3(115.f/255,115.f/255,115.f/255)*0.05f);


extern float bunny_level4[222][4];
extern float armadillo_level4[155][4];
extern float obj_grid_armadillo_unify_grid_level6_classic[3657][4];

namespace YC
{
	MyScene::MyScene(bool showFPS, float g_FOV, float g_near, float g_far
		,const vec3& _lightPos
		,const vec3& _camerPos
		,const vec3& _eyeCenter
		,const vec3& _zUp
#if SHOW_SHADOWMAP
		,int _shadowMapWidth
		,int _shadowMapHeight
#endif
		)
		:ArcBall(windowWidth,windowHeight),m_showFPS(showFPS)
		,FVOY(g_FOV),MyFar(g_far),MyNear(g_near)
		,lightPos(_lightPos)//(-0.8f, 0.825f, 0.75f),
		,eyeCenter(_eyeCenter)//(0.0f,0.1f,0.0f),
		,zUp(_zUp)//(0.0f,1.f,0.0f),
		,camerPos(_camerPos)//(0.0f,0.05f,0.85f)
#if SHOW_SHADOWMAP
		,shadowMapWidth(_shadowMapWidth),shadowMapHeight(_shadowMapHeight)
#endif
	{
		angle = 0.f;
		m_TexRender.BuildFont();
	}


	MyScene::~MyScene(void)
	{
		delete m_scene_plane;
		delete m_scene_ObjMesh;
#if SHOW_SHADOWMAP
		delete lightFrustum;
#endif
	}

	void MyScene::initScene()
	{
		compileAndLinkShader();

		glClearColor(0.0f,0.0f,0.0f,1.0f);
		//glClearColor(0.0,0.0,0.0,1.0);
		//glClearColor(1.0,1.0,1.0,1.0);
		glEnable(GL_DEPTH_TEST);

		//Armadillo_34w
		//Bunny_7w
		//Sphere_2880
//#define _OBJ_NAME "D:/MyWorkSpace2014/MyMesh/OBJ/Bunny_7w.obj" 
		m_scene_ObjMesh = new MyVBOMesh("",true,true);
		m_scene_OctreeGrid = new MyVBOLineSet;
#if USE_FEM

		YC::Geometry::MeshDataStruct tmp_objInfo;
		//tmp_objInfo.loadPLY(_PLY_NAME);
		tmp_objInfo.loadOBJ(GlobalVariable::g_strMeshPath.c_str(),false);
		m_scene_ObjMesh->loadOBJ(tmp_objInfo);

		//obj_grid_armadillo_unify_grid_level6_classic
#if CellLevel < 5
		const int nCellCount = sizeof(armadillo_level4) / sizeof(armadillo_level4[0]);
		m_physicalSimulation.initialize(armadillo_level4,nCellCount,tmp_objInfo);
#else
		const int nCellCount = sizeof(obj_grid_armadillo_unify_grid_level6_classic) / sizeof(obj_grid_armadillo_unify_grid_level6_classic[0]);
		m_physicalSimulation.initialize(obj_grid_armadillo_unify_grid_level6_classic,nCellCount,tmp_objInfo);
#endif
		
		

		{
			const int nLineCount = nCellCount * 12 * _nExternalMemory;
			float * pos = new float [nLineCount*2*3];
			int * idx = new int [nLineCount*2];
			memset(pos,'\0',nLineCount*2*3*sizeof(float));
			memset(idx,'\0',nLineCount*2*sizeof(int));
			m_scene_OctreeGrid->initialize(nLineCount,pos,idx);
			m_scene_OctreeGrid->setLineCount(0);
			delete [] pos;
			delete [] idx;
		}
		glLineWidth(0.5f);
#endif

#if SHOW_SHADOWMAP


		m_scene_plane = new VBOPlane(40.0f, 40.0f, 2, 2);
		float scale = 2.0f;

		setupFBO();

		GLuint programHandle = prog.getHandle();
		pass1Index = glGetSubroutineIndex( programHandle, GL_FRAGMENT_SHADER, "recordDepth");
		pass2Index = glGetSubroutineIndex( programHandle, GL_FRAGMENT_SHADER, "shadeWithShadow");

		shadowBias = mat4( vec4(0.5f,0.0f,0.0f,0.0f),
			vec4(0.0f,0.5f,0.0f,0.0f),
			vec4(0.0f,0.0f,0.5f,0.0f),
			vec4(0.5f,0.5f,0.5f,1.0f)
			);

		lightFrustum = new Frustum(Projection::PERSPECTIVE);
		
		lightFrustum->orient( lightPos, eyeCenter, zUp);
		lightFrustum->setPerspective( FVOY, 1.0f, 1.0f, 25.0f);
		lightPV = shadowBias * lightFrustum->getProjectionMatrix() * lightFrustum->getViewMatrix();

		prog.setUniform("Light.Intensity", vec3(0.5f));

		prog.setUniform("ShadowMap", 0);
#else
#endif

#if SHOWFPS
		cutilCheckError( sdkCreateTimer( &timer));
#endif
	}

	void MyScene::showPlane()
	{		
		//prog.setUniform("Kd", 0.7f, 0.5f, 0.3f);
		prog.setUniform(PREFIX"Kd", planeColor);
		prog.setUniform(PREFIX"Ks", 0.1f, 0.1f, 0.1f);
		prog.setUniform(PREFIX"Ka", planeColor);
		prog.setUniform(PREFIX"Shininess", 1.0f);

		model = mat4(1.0f);
		model *= glm::translate(vec3(0.0f,-0.25f,0.0f));
		//model *= glm::translate(vec3(0.0f,-0.45f,0.0f));
		setMatrices();
		m_scene_plane->render();
	}

	void MyScene::showMesh()
	{		
		prog.setUniform(PREFIX"Ka", modelColor4Ka);
		prog.setUniform(PREFIX"Kd", modelColor);
		prog.setUniform(PREFIX"Ks", vec3(0.35f,0.35f,0.35f));
		prog.setUniform(PREFIX"Shininess", 180.0f);

		model = mat4(1.0f);
		
		model *= glm::rotate(180.0f, vec3(0.0f,1.0f,0.0f));
		model *= glm::scale(0.5f,0.5f,0.5f);
		//model *= glm::scale(0.1f,0.1f,0.1f);
		//model *= glm::translate(vec3(0.0f,0.0f,-0.5f));
		model *= mat4(MyTransform.s.M00,MyTransform.s.M10,MyTransform.s.M20,MyTransform.s.M30,
			MyTransform.s.M01,MyTransform.s.M11,MyTransform.s.M21,MyTransform.s.M31,
			MyTransform.s.M02,MyTransform.s.M12,MyTransform.s.M22,MyTransform.s.M32,
			MyTransform.s.M03,MyTransform.s.M13,MyTransform.s.M23,MyTransform.s.M33);
		setMatrices();
		//m_scene_test_teapot->render();
		m_scene_ObjMesh->render();
		//m_scene_OctreeGrid->render();
	}
#if SHOW_SHADOWMAP

#if 1
	void MyScene::drawScene()
	{
		showMesh();
		showPlane();
		model = mat4(1.0f);
	}
#endif

	void MyScene::render()
	{
		prog.use();
#if SHOWFPS
		cutilCheckError(sdkStartTimer(&timer)); 
		glBindTexture(GL_TEXTURE_2D, m_depthTex);
#endif

#if USE_FEM
		m_physicalSimulation.simulationOnCPU(m_nTimeStep++);

		m_scene_ObjMesh->loadDisplacementOBJ(m_physicalSimulation.getObjMesh());

		const int nCellCount = m_physicalSimulation.getCellSize();
		const int nLineCount = nCellCount * 12;
		float3* pos = new float3[nLineCount*2];
		int2* index_Lines = new int2[nLineCount];
		memset(pos,'\0',sizeof(float3)*nLineCount*2);
		memset(index_Lines,'\0',sizeof(int2)*nLineCount);

		m_physicalSimulation.generateDisplaceLineSet(&pos,&index_Lines);
		m_scene_OctreeGrid->updateLineSet(nLineCount,pos,index_Lines);
		m_nTimeStep = m_nTimeStep & (128-1);
		
#endif
		// Pass 1 (shadow map generation)
		view = lightFrustum->getViewMatrix();
		projection = lightFrustum->getProjectionMatrix();
		glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
		glClear(GL_DEPTH_BUFFER_BIT);
		glViewport(0,0,shadowMapWidth,shadowMapHeight);
		glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &pass1Index);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);
#if USE_PCF
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(2.5f,10.0f);
#endif

		drawScene();

#if USE_PCF
		glCullFace(GL_BACK);
		glDisable(GL_POLYGON_OFFSET_FILL);
#endif

		// Pass 2 (render)
		view = glm::lookAt(camerPos, eyeCenter, zUp);
		//view = glm::lookAt(cameraPos,vec3(0.0f),vec3(0.0f,1.0f,0.0f));
		prog.setUniform("Light.Position", view * vec4(lightFrustum->getOrigin(),1.0f));
		projection = glm::perspective(FVOY, (float)aspect, MyNear, MyFar);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0,0,width,height);
		glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &pass2Index);
		glDisable(GL_CULL_FACE);
		drawScene();

#if SHOWFPS
		cutilCheckError(sdkStopTimer(&timer));
		computeFPS();
		//glutSwapBuffers();
#endif
	}
#endif

	void MyScene::resize(int w, int h)
	{
		glViewport(0,0,w,h);
		width = w;
		height = h;

		aspect = ((float)width) / height;
		projection = glm::perspective(FVOY, (float)aspect, MyNear, MyFar);

		ArcBall.setBounds((GLfloat)width, (GLfloat)height); 

#if SHOWFPS
		m_MyGLSLFont.resize(width,height);
#endif
	}

	void MyScene::setMatrices()
	{
		mat4 mv = view * model;
		prog.setUniform("ModelViewMatrix", mv);
		prog.setUniform("NormalMatrix",
			mat3( vec3(mv[0]), vec3(mv[1]), vec3(mv[2]) ));
		prog.setUniform("MVP", projection * mv);

#if SHOW_SHADOWMAP
		prog.setUniform("ShadowMatrix", lightPV * model);
#endif
	}

	void MyScene::compileAndLinkShader()
	{
#if SHOW_SHADOWMAP
#if USE_PCF
		
		if( ! prog.compileShaderFromFile("./MyShader/pcf.vs",GLSLShader::VERTEX) )
		{
			printf("Vertex shader failed to compile!\n%s",
				prog.log().c_str());
			exit(1);
		}
		if( ! prog.compileShaderFromFile("./MyShader/pcf.fs",GLSLShader::FRAGMENT))
		{
			printf("Fragment shader failed to compile!\n%s",
				prog.log().c_str());
			exit(1);
		}
#endif
#endif
		
		if( ! prog.link() )
		{
			printf("Shader program failed to link!\n%s",
				prog.log().c_str());
			exit(1);
		}

		if( ! prog.validate() )
		{
			printf("Program failed to validate!\n%s",
				prog.log().c_str());
			exit(1);
		}

		prog.use();
	}	
}

#include "MyScene_StableFunction.h"
