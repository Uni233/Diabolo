#ifndef _VR_MACRO_H
#define _VR_MACRO_H

namespace YC
{
	
namespace MACRO
{
	#define Platform_Windows (1)
	#define PlatForm_Linux   (2)
	#define PlatformType Platform_Windows

	#define MyError(X)do{printf("%s\n",#X);exit(66);}while(false)
	#define MyPause system("pause")
	#define MyExit exit(66)
	#define Q_ASSERT(X) do{if (!(X)){printf("%s\n",#X);system("pause");}}while(false)
	#define MyNotice

	#define windowTitle "VR Physics Scene"
	#define LogInfo printf
	#define cutilCheckError
	#define windowTitle "VR Physics Scene"
	#define windowWidth (800)
	#define windowHeight (600)
	#define MyBufferSize (256)
	#define _nExternalMemory (2)

	#define Invalid_Id (-1)
	#define _CellTypeFEM (0)
	#define _CellTypeEFG (1)
	#define _CellTypeCouple (2)
	#define _CellTYPEInvalid (3)

	#define MyZero (0)
	#define MyNull (0)
	#define X__AXIS (0)
	#define Y__AXIS (1)
	#define Z__AXIS (2)

#define CellLevel (4)
#if CellLevel == 4
	#define CellRaidus   (0.015625*4)
#elif CellLevel == 5
	#define CellRaidus   (0.015625*2)
#elif CellLevel == 6
	#define CellRaidus   (0.015625)
#else
	#error UnSupport Cell Level (4,5,6).
#endif

#define SHOW_SHADOWMAP (1)
#if SHOW_SHADOWMAP
	#define USE_PCF (1)
#endif
#define SHOWFPS (1)

	#define USE_CUDA (1)
	#define USE_FEM (1)
	#define USE_CO_RATION (1)
	#define MaterialMatrixSize (6)
	#define MyDIM (3)
	#define ValueScaleFactor (10000000000)
	typedef enum{FEM=_CellTypeFEM,INVALIDTYPE=_CellTYPEInvalid} CellType;
}
}

#endif//_VR_MACRO_H