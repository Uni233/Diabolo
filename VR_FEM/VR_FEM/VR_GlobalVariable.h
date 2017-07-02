#ifndef _YC_GLOBALVARIABLE_H
#define _YC_GLOBALVARIABLE_H

#include "VR_MACRO.h"
#include "VR_Global_Define.h"
#include <string>
#include <vector>
#include <glm/glm.hpp>
using glm::mat4;
using glm::vec3;

namespace YC
{
	namespace GlobalVariable
	{
		extern std::string g_strCurrentPath;
		extern std::string s_qstrCurrentTime;
		extern std::string s_qstrCurrentTimeStamp;

		extern bool g_fullscreen;
		extern bool g_hasCoords;
		extern bool g_hasVerticeNormal;
		extern bool g_normalizeMesh;
		extern bool g_showFPS;
		extern bool g_simulation;
		extern float g_FOV;
		extern float g_near;
		extern float g_far;
		extern std::string g_strMeshName;
		extern std::string g_strMeshId;
		extern std::string g_strTextureName;
		extern std::string g_strMeshPath;
		extern std::vector< YC::MyVec3 > g_vecMaterialPoint;
		extern std::vector< float > g_vecMaterialPointStiff;
		extern float g_bladeForce;
		extern float g_tmpBladeForce;
		extern float g_hasGravity;
		extern float g_domainGravity[3];
		extern int g_isApplyBladeForceCurrentFrame;

		extern float g_dbYoungModulus;
		extern int g_bcMinCount;
		extern int g_bcMaxCount;
		extern float g_externalForceFactor;
		extern float g_scriptForceFactor;

		extern vec3 camerPos;
		extern vec3 lightPos;
		extern vec3 eyeCenter;
		extern vec3 zUp;

		extern bool g_useCrossBlade;
		extern bool g_useScriptBlade;
		extern int shadowMapWidth;
		extern int shadowMapHeight;

		void initGlobalVariable(int argc, char *argv[]);
	}
}
#endif//_YC_GLOBALVARIABLE_H