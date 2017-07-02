#pragma once
#include "MySceneBase.h"

#include "glslprogram.h"
#include "vboplane.h"
#include "MyVBOMesh.h"
#include <glm/glm.hpp>
using glm::mat4;

#include "math.h"												    // NEW: Needed For Sqrtf
#include "Rotation/ArcBall.h"	
#include "VR_MACRO.h"


#if SHOW_SHADOWMAP
#include "frustum.h"
#endif

#if SHOWFPS
#include <helper_timer.h>
#include "VR_GLSL_Font.h"

#endif

#if SHOW_SHADOWMAP
#define PREFIX "Material."
#else
#define PREFIX ""
#endif

#include "TexRender.h"

#if USE_FEM
#include "VR_Physic_FEM_Simulation.h"
#include "MyVBOLineSet.h"
#endif

namespace YC
{
	class MyScene : public MySceneBase
	{
	public:
		MyScene(bool showFPS, float g_FOV, float g_near, float g_far
			,const vec3& _lightPos
			,const vec3& _camerPos
			,const vec3& _eyeCenter
			,const vec3& _zUp
#if SHOW_SHADOWMAP
			,int _shadowMapWidth
			,int _shadowMapHeight
#endif
			);
		virtual ~MyScene(void);

	public:
		void initScene();
		void update( float t );
		void render();
		void resize(int, int);

		void mouse(int x,int y);
		void motion(int x,int y);

	private:
		bool m_showFPS;
		TexRender m_TexRender;
		GLSLProgram prog;
		mat4 model;
		mat4 view;
		mat4 projection;

		const vec3 lightPos;
		const vec3 eyeCenter;
		const vec3 zUp;
		const vec3 camerPos;

		void setMatrices();
		void compileAndLinkShader();

		VBOPlane * m_scene_plane;
		MyVBOMesh * m_scene_ObjMesh;
		MyVBOLineSet * m_scene_OctreeGrid;
		int width,height;
		float angle;
		float aspect;

		void showPlane();
		void showMesh();

		const float FVOY;// (60.f)
		const float MyNear;// (0.3f)
		const float MyFar;//  (100.f)

		//For mouse interactive
		ArcBallT    ArcBall;				                // NEW: ArcBall Instance
		Point2fT MousePt;

		GLuint m_depthTex;

#if SHOW_SHADOWMAP
		//For ShadowMap + PCF
		GLuint shadowFBO, pass1Index, pass2Index;
		const int shadowMapWidth;
		const int shadowMapHeight;
		mat4 lightPV;
		mat4 shadowBias;
		Frustum *lightFrustum;		
		
		void setupFBO();
		void drawScene();
		void spitOutDepthBuffer();
#endif

#if SHOWFPS
		StopWatchInterface * timer;
		void computeFPS();
		MyGLSLFont m_MyGLSLFont;
#endif

#if USE_FEM
		Physics::GPU::VR_Physics_FEM_Simulation m_physicalSimulation;
		unsigned int m_nTimeStep;
#endif
	};
}
