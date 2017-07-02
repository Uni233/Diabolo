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

#include "VR_GPU_Geometry_Octree.h"

#include "VR_Color.h"

#include <boost/format.hpp>

#include "VR_MACRO.h"

#include "BladeTraceData/BladeTraceDefines.h"

using namespace YC::GlobalVariable;

extern int nCurrentTick,nLastTick;

#if SHOWFPS
StopWatchInterface * timer;
#endif

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

//int myRed=251,myGreen=135,MyBlue=38;
//int myRed=251,myGreen=194,MyBlue=99;
//int myRed=255,myGreen=167,MyBlue=99;
//int myRed=192,myGreen=133,MyBlue=84;
int myRed=115,myGreen=115,MyBlue=115;

const vec3 planeColor(173.f/255,128.f/255,74.f/255);
const vec3 modelColor(115.f/255,115.f/255,115.f/255);
const vec3 modelColor4Ka(vec3(115.f/255,115.f/255,115.f/255)*0.05f);

extern float obj_grid_sphere_unify_grid_level6[7000][4];
extern float obj_grid_bunny_unify_grid_level6[4657][4];
extern float obj_grid_steak_unify_grid_level6[2976][4];
extern float obj_grid_armadillo_unify_grid_level6[3026][4];
extern float obj_grid_armadillo_unify_grid_level6_classic[3657][4];
extern float obj_grid_bunny_unify_grid_level5[1081][4];
extern float obj_beam_Zup_unify_grid[2048][4];

extern void func();

namespace YC
{
	MyScene::MyScene(bool showFPS, float g_FOV, float g_near, float g_far
		,const vec3& _lightPos
		,const vec3& _camerPos
		,const vec3& _eyeCenter
		,const vec3& _zUp
		,SceneType t
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
		,m_SceneType(t)
#if SHOW_SHADOWMAP
		,shadowMapWidth(_shadowMapWidth),shadowMapHeight(_shadowMapHeight)
#endif
	{
		m_nTimeStep = 0;
		angle = 0.f;
		m_TexRender.BuildFont();
	}


	MyScene::~MyScene(void)
	{
		delete m_scene_plane;
		delete m_scene_test_torus;
		delete m_scene_test_teapot;
		delete m_scene_ObjMesh;
#if SHOW_SHADOWMAP
		delete lightFrustum;
#endif
	}

	void MyScene::initScene()
	{
		compileAndLinkShader();

		//glClearColor(0.0f,0.0f,0.0f,1.0f);
		//glClearColor(0.0,0.0,0.0,1.0);
		glClearColor(1.0,1.0,1.0,1.0);
		glEnable(GL_DEPTH_TEST);

		//armadillo_1M_Normal
		//Armadillo_34w
		//Bunny_7w
		//Sphere_2880
		//steak_v_vn_vt_f
#define _NULL_STRING ""
#define _OBJ_NAME "D:/MyWorkSpace/MyMesh/OBJ/armadillo_1M_Normal.obj" 
#define _PLY_NAME "D:/MyWorkSpace/MyMesh/ply/heart.ply" 
		

		m_scene_CuttingTriangle = new MyVBOLineSet;
		m_scene_ObjMesh = new MyVBOMesh(_NULL_STRING,true,true);
		m_scene_OctreeGrid = new MyVBOLineSet;
		m_scene_CuttingTraceLines = new MyVBOLineSet;

		if (Physical_Scene == m_SceneType)
		{
			/*LogInfo("%s %d\n",__FILE__,__LINE__);
			func();*/
			YC::Geometry::MeshDataStruct tmp_objInfo;
			//tmp_objInfo.loadPLY(_PLY_NAME);
			tmp_objInfo.loadOBJ(_OBJ_NAME,false);
			m_scene_ObjMesh->loadOBJ(tmp_objInfo);
			
			//obj_grid_bunny_unify_grid_level5
			//obj_grid_bunny_unify_grid_level6
			//obj_grid_armadillo_unify_grid_level6
			//obj_grid_armadillo_unify_grid_level6_classic
			//float obj_beam_Zup_unify_grid[1600][4]
			const int nCellCount = sizeof(obj_grid_armadillo_unify_grid_level6_classic) / sizeof(obj_grid_armadillo_unify_grid_level6_classic[0]);
			m_physicalSimulation.initialize(obj_grid_armadillo_unify_grid_level6_classic,nCellCount,tmp_objInfo);
			std::vector< unsigned int > vecVBOID;
			vecVBOID.push_back(m_scene_ObjMesh->getVBOHandle(0));
			vecVBOID.push_back(m_scene_ObjMesh->getVBOHandle(1));
			//vecVBOID.push_back(m_scene_ObjMesh->getVBOHandle(2));
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

				vecVBOID.push_back(m_scene_OctreeGrid->getVBOHandle(0));
				vecVBOID.push_back(m_scene_OctreeGrid->getVBOHandle(1));
			}
			{
				//make blade trace
				//ArmadilloMiddleBlade
				const int nLineCount = 1074;
				float * pos = new float [nLineCount*2*3];
				int * idx = new int [nLineCount*2];
				memset(pos,'\0',nLineCount*2*3*sizeof(float));
				memset(idx,'\0',nLineCount*2*sizeof(int));

				for (int i=0;i<nLineCount;++i)
				{
					int linePosBase = i * 2 * 3;
					int lineIdxBase = i * 2;

					for (int j=0;j<6;++j)
					{
						pos[ linePosBase+j ] = ArmadilloMiddleBlade[i][j];
					}
					idx[lineIdxBase] = lineIdxBase;
					idx[lineIdxBase+1] = lineIdxBase+1;
				}
				m_scene_CuttingTraceLines->initialize(nLineCount,pos,idx);
				m_scene_CuttingTraceLines->setLineCount(nLineCount);
				delete [] pos;
				delete [] idx;
				vecVBOID.push_back(m_scene_CuttingTraceLines->getVBOHandle(0));
				vecVBOID.push_back(m_scene_CuttingTraceLines->getVBOHandle(1));
				
			}
			{
				const int nLineCount = 3;
				float * pos = new float [nLineCount*2*3];
				int * idx = new int [nLineCount*2];
				memset(pos,'\0',nLineCount*2*3*sizeof(float));
				memset(idx,'\0',nLineCount*2*sizeof(int));

				float ArmadilloMiddleBladeTmp[3][6]={
					{-0.005100,0.0000 ,-0.100000,-0.005100,1.5000 ,-1.50000},
					{-0.005100,0.0000 ,-0.100000,-0.005100,0.0000 ,1.200000},
					{-0.005100,1.5000 ,-1.50000,-0.005100,0.0000 ,1.200000}
				};
				for (int i=0;i<nLineCount;++i)
				{
					int linePosBase = i * 2 * 3;
					int lineIdxBase = i * 2;

					for (int j=0;j<6;++j)
					{
						pos[ linePosBase+j ] = ArmadilloMiddleBladeTmp[i][j];
					}
					idx[lineIdxBase] = lineIdxBase;
					idx[lineIdxBase+1] = lineIdxBase+1;
				}
				
				m_scene_CuttingTriangle->initialize(nLineCount,pos,idx);
				m_scene_CuttingTriangle->setLineCount(nLineCount);
				delete [] pos;
				delete [] idx;
				vecVBOID.push_back(m_scene_CuttingTriangle->getVBOHandle(0));
				vecVBOID.push_back(m_scene_CuttingTriangle->getVBOHandle(1));
			}
			m_physicalSimulation.registerVBOID(vecVBOID);
			/*LogInfo("%s %d\n",__FILE__,__LINE__);
			func();*/
		}
		else if (Geometry_Scene == m_SceneType)
		{
			YC::Geometry::MeshDataStruct tmp_objInfo;
			tmp_objInfo.loadOBJ(_OBJ_NAME,false);
			YC::Geometry::GPU::VR_Octree octree;
			octree.spliteCube(tmp_objInfo,g_octreeFineLevel);
			octree.exportOctreeGrid("d:\\a.cpp",g_octreeFineLevel-1);
			m_MyVBOLineSet = octree.makeVBOLineSet();
			/*for (int j=0;j<g_octreeFineLevel;++j)
			{
				MyVBOLineSet * tmp = octree.makeVBOLineSet(j);
				vecOctreePerLevel.push_back(tmp);
				LogInfo("Level = %d, nLinesCount = %d\n",j,tmp->getLineCount());
				
			}
			MyPause;*/
			m_scene_ObjMesh->loadOBJ(tmp_objInfo);
		}
		else if (Cutting_Scene == m_SceneType)
		{
			//glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
			YC::Geometry::MeshDataStruct tmp_objInfo;
			tmp_objInfo.loadPLY(_PLY_NAME);
			//tmp_objInfo.loadOBJ(_OBJ_NAME,false);
			m_scene_ObjMesh->loadOBJ(tmp_objInfo);
			const int nCellCount = sizeof(obj_grid_bunny_unify_grid_level5) / sizeof(obj_grid_bunny_unify_grid_level5[0]);
			//m_physicalSimulation.initialize(obj_grid_bunny_unify_grid_level5,nCellCount*0,tmp_objInfo);

			m_MyVBOLineSet_AABB = m_physicalSimulation.getAABB();
			m_MyVBOLineSet_BladeList = new MyVBOLineSet();
			m_MyVBOMesh_BladeTriangleList = new MyVBOMesh(MyNull,false,false,false);
			
			{
				const int nLineCount = 999;
				float * pos = new float [nLineCount*2*3];
				int * idx = new int [nLineCount*2];
				memset(pos,'\0',nLineCount*2*3*sizeof(float));
				memset(idx,'\0',nLineCount*2*sizeof(int));
				m_MyVBOLineSet_BladeList->initialize(nLineCount,pos,idx);
				m_MyVBOLineSet_BladeList->setLineCount(0);
				delete [] pos;
				delete [] idx;

				m_MyVBOMesh_BladeTriangleList->initialize(nLineCount,nLineCount);
			}

			std::vector< unsigned int > vecVBOID;
			vecVBOID.push_back(m_MyVBOLineSet_BladeList->getVBOHandle(0));
			vecVBOID.push_back(m_MyVBOLineSet_BladeList->getVBOHandle(1));

			vecVBOID.push_back(m_MyVBOMesh_BladeTriangleList->getVBOHandle(0));
			vecVBOID.push_back(m_MyVBOMesh_BladeTriangleList->getVBOHandle(1));

			vecVBOID.push_back(m_scene_ObjMesh->getVBOHandle(0));
			vecVBOID.push_back(m_scene_ObjMesh->getVBOHandle(1));
			m_physicalSimulation.registerVBOID(vecVBOID);

		}
		else
		{
			MyError("Unsupport Scene Type! (Physical, Geometry)");
		}

#if SHOW_SHADOWMAP

		m_scene_test_teapot = new VBOTeapot(14, mat4(1.0f));
		m_scene_plane = new VBOPlane(40.0f, 40.0f, 2, 2);
		float scale = 2.0f;
		m_scene_test_torus = new VBOTorus(0.7f * scale,0.3f * scale,50,50);

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

		/*glLineWidth(4.5f);
		glPointSize(3.5f);*/
#else
		//view = glm::lookAt(vec3(0.0f,3.0f,5.0f), vec3(0.0f,0.75f,0.0f), vec3(0.0f,1.0f,0.0f));
		m_scene_plane = new VBOPlane(40.0f, 40.0f, 2, 2);

		view = glm::lookAt(camerPos, eyeCenter, zUp);

		projection = mat4(1.0f);

		angle = 0.957283f;

		prog.setUniform("LightIntensity", vec3(0.85f)/*vec3(0.9f,0.9f,0.9f)*/ );

		//float c = 0.15f;
		//vec3 lightPos = vec3(0.0f,c * 5.25f, c * 7.5f);  // World coords
		prog.setUniform("LightPosition", lightPos );
#endif

//#if SHOWFPS
//		cutilCheckError( sdkCreateTimer( &timer));
//#endif
		/*LogInfo("%s %d\n",__FILE__,__LINE__);
		func();*/
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

	void MyScene::showLineSet()
	{
		//for (int i=0;i<vecOctreePerLevel.size();++i)
		int i=2;
		{
			vec3 levelColor(YC::Colors::colorTemplage[i+1][0],YC::Colors::colorTemplage[i+1][1],YC::Colors::colorTemplage[i+1][2]);
			prog.setUniform(PREFIX"Ka", modelColor4Ka);
			prog.setUniform(PREFIX"Kd", levelColor);
			prog.setUniform(PREFIX"Ks", vec3(0.35f,0.35f,0.35f));
			prog.setUniform(PREFIX"Shininess", 180.0f);

			model = mat4(1.0f);

			model *= glm::scale(0.5f,0.5f,0.5f);
			model *= mat4(MyTransform.s.M00,MyTransform.s.M10,MyTransform.s.M20,MyTransform.s.M30,
				MyTransform.s.M01,MyTransform.s.M11,MyTransform.s.M21,MyTransform.s.M31,
				MyTransform.s.M02,MyTransform.s.M12,MyTransform.s.M22,MyTransform.s.M32,
				MyTransform.s.M03,MyTransform.s.M13,MyTransform.s.M23,MyTransform.s.M33);
			setMatrices();
			vecOctreePerLevel[i]->render();
		}
	}

	void MyScene::showMesh()
	{		
		prog.setUniform(PREFIX"Ka", modelColor4Ka);
		prog.setUniform(PREFIX"Kd", modelColor);
		prog.setUniform(PREFIX"Ks", vec3(0.35f,0.35f,0.35f));
		prog.setUniform(PREFIX"Shininess", 180.0f);

		model = mat4(1.0f);
		
#if ModelNameIsArmadillo
		model *= glm::rotate(180.0f, vec3(0.0f,1.0f,0.0f));
#endif
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
		m_scene_OctreeGrid->render();
		//m_scene_CuttingTraceLines->render();
		m_scene_CuttingTriangle->render();
		//m_MyVBOLineSet->render();
	}
#if SHOW_SHADOWMAP

#if 1
	void MyScene::drawScene()
	{
		showMesh();
		//showLineSet();
		showPlane();
		model = mat4(1.0f);
	}
#else
	void MyScene::drawScene()
	{
		vec3 color = vec3(0.7f,0.5f,0.3f);
		prog.setUniform("Material.Ka", color * 0.05f);
		prog.setUniform("Material.Kd", color);
		prog.setUniform("Material.Ks", vec3(0.9f,0.9f,0.9f));
		prog.setUniform("Material.Shininess", 150.0f);
		model = mat4(1.0f);
		model *= glm::translate(vec3(0.0f,0.0f,0.0f));
		model *= glm::rotate(-90.0f, vec3(1.0f,0.0f,0.0f));
		setMatrices();
		m_scene_test_teapot->render();

		prog.setUniform("Material.Ka", color * 0.05f);
		prog.setUniform("Material.Kd", color);
		prog.setUniform("Material.Ks", vec3(0.9f,0.9f,0.9f));
		prog.setUniform("Material.Shininess", 150.0f);
		model = mat4(1.0f);
		model *= glm::translate(vec3(0.0f,2.0f,5.0f));
		model *= glm::rotate(-45.0f, vec3(1.0f,0.0f,0.0f));
		setMatrices();
		m_scene_test_torus->render();

		prog.setUniform("Material.Kd", 0.25f, 0.25f, 0.25f);
		prog.setUniform("Material.Ks", 0.0f, 0.0f, 0.0f);
		prog.setUniform("Material.Ka", 0.05f, 0.05f, 0.05f);
		prog.setUniform("Material.Shininess", 1.0f);
		model = mat4(1.0f);
		model *= glm::translate(vec3(0.0f,0.0f,0.0f));
		setMatrices();
		m_scene_plane->render();
		model = mat4(1.0f);
		model *= glm::translate(vec3(-5.0f,5.0f,0.0f));
		model *= glm::rotate(-90.0f,vec3(0.0f,0.0f,1.0f));
		setMatrices();
		m_scene_plane->render();
		model = mat4(1.0f);
		model *= glm::translate(vec3(0.0f,5.0f,-5.0f));
		model *= glm::rotate(90.0f,vec3(1.0f,0.0f,0.0f));
		setMatrices();
		m_scene_plane->render();
		model = mat4(1.0f);
	}
#endif

	void MyScene::render()
	{
		static unsigned int nTriangleSize,nLineSize;

		prog.use();
#if SHOWFPS
		cutilCheckError(sdkStartTimer(&timer)); 
		glBindTexture(GL_TEXTURE_2D, m_depthTex);
#endif
#if 1
		m_physicalSimulation.simulationOnCUDA(m_nTimeStep++,
			m_scene_ObjMesh->getVAOHandle(),
			m_scene_OctreeGrid->getVBOHandle(0),m_scene_OctreeGrid->getVBOHandle(1),
			m_scene_ObjMesh->getVBOHandle(0),m_scene_ObjMesh->getVBOHandle(1),m_scene_ObjMesh->getVBOHandle(2),
			nTriangleSize,nLineSize);

		m_scene_ObjMesh->setElements(3*nTriangleSize);
		m_scene_OctreeGrid->setLineCount(nLineSize);

		m_nTimeStep = m_nTimeStep & (128-1);
#else
		m_scene_ObjMesh->setElements(m_physicalSimulation.getTriangleMesh(m_scene_ObjMesh->getVAOHandle(),m_scene_ObjMesh->getVBOHandle(0),m_scene_ObjMesh->getVBOHandle(1),m_scene_ObjMesh->getVBOHandle(2)));
		m_MyVBOMesh_BladeTriangleList->setElements(m_physicalSimulation.getBladeTriangleList(0,m_MyVBOMesh_BladeTriangleList->getVBOHandle(0),m_MyVBOMesh_BladeTriangleList->getVBOHandle(1)));
		m_MyVBOLineSet_BladeList->setLineCount(m_physicalSimulation.getBladeList(0,m_MyVBOLineSet_BladeList->getVBOHandle(0),m_MyVBOLineSet_BladeList->getVBOHandle(1)));
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

		drawScene();//drawSceneMeshCutting();//drawScene();

#if USE_PCF
		glCullFace(GL_BACK);
		glDisable(GL_POLYGON_OFFSET_FILL);
#endif
		
		//spitOutDepthBuffer(); // This is just used to get an image of the depth buffer

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

		drawScene();//drawSceneMeshCutting();//drawScene();

#if SHOWFPS
		computeFPS();
		cutilCheckError(sdkStopTimer(&timer));
		//glutSwapBuffers();
#endif
	}
#else
	void MyScene::render()
	{
#if SHOWFPS
		cutilCheckError(sdkStartTimer(&timer)); 
#endif

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		showMesh();

		showPlane();

#if SHOWFPS
		cutilCheckError(sdkStopTimer(&timer));
		computeFPS();
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
#else
		if( ! prog.compileShaderFromFile("./MyShader/shadowmap.vs",GLSLShader::VERTEX) )
		{
			printf("Vertex shader failed to compile!\n%s",
				prog.log().c_str());
			exit(1);
		}
		if( ! prog.compileShaderFromFile("./MyShader/shadowmap.fs",GLSLShader::FRAGMENT))
		{
			printf("Fragment shader failed to compile!\n%s",
				prog.log().c_str());
			exit(1);
		}
#endif
#else
		if( ! prog.compileShaderFromFile("./MyShader/halfway.vs",GLSLShader::VERTEX) )
		{
			printf("Vertex shader failed to compile!\n%s",
				prog.log().c_str());
			exit(1);
		}
		if( ! prog.compileShaderFromFile("./MyShader/halfway.fs",GLSLShader::FRAGMENT))
		{
			printf("Fragment shader failed to compile!\n%s",
				prog.log().c_str());
			exit(1);
		}
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

#if USE_Mesh_Cutting
	void MyScene::drawSceneMeshCutting()
	{
		showMesh();
		showLineSetAABB();
		showPlane();
		model = mat4(1.0f);
	}

	void MyScene::showLineSetAABB()
	{
		//for (int i=0;i<vecOctreePerLevel.size();++i)
		int i=2;
		{
			vec3 levelColor(YC::Colors::colorTemplage[i+1][0],YC::Colors::colorTemplage[i+1][1],YC::Colors::colorTemplage[i+1][2]);
			prog.setUniform(PREFIX"Ka", modelColor4Ka);
			prog.setUniform(PREFIX"Kd", levelColor);
			prog.setUniform(PREFIX"Ks", vec3(0.35f,0.35f,0.35f));
			prog.setUniform(PREFIX"Shininess", 180.0f);

			model = mat4(1.0f);

			model *= glm::scale(0.5f,0.5f,0.5f);
			model *= mat4(MyTransform.s.M00,MyTransform.s.M10,MyTransform.s.M20,MyTransform.s.M30,
				MyTransform.s.M01,MyTransform.s.M11,MyTransform.s.M21,MyTransform.s.M31,
				MyTransform.s.M02,MyTransform.s.M12,MyTransform.s.M22,MyTransform.s.M32,
				MyTransform.s.M03,MyTransform.s.M13,MyTransform.s.M23,MyTransform.s.M33);
			setMatrices();
			m_MyVBOLineSet_AABB->render();
			m_MyVBOLineSet_BladeList->render();
			m_MyVBOMesh_BladeTriangleList->render();
		}
	}
#endif
}

#include "MyScene_StableFunction.h"
