#ifndef _vrFileSystem_h_
#define _vrFileSystem_h_

#include "vrTypes.h"
namespace VR
{
	namespace FileSystem
	{
		vrString get_currentpath();
		void parser_filepath(const vrString& filePath, vrString& strParentPath, vrString& strFileName, vrString& strStem, vrString& strExtension);
		int get_filenames(const vrString& dir, std::vector< vrString >& filenames);
		//generic: D : / MyWorkspace / MyPaper / 2.Utility / MyGitTestProject / libTools / x64 / Release
		//native : D : \MyWorkspace\MyPaper\2.Utility\MyGitTestProject\libTools\x64\Release
		//quoted : "D:\MyWorkspace\MyPaper\2.Utility\MyGitTestProject\libTools\x64\Release"
		//Components :
		//[D:][/ ][MyWorkspace][MyPaper][2.Utility][MyGitTestProject][libTools][x64][Release]
		void printFileSystemInfo();
		void printPathParts(const vrString& p1);
	}
}
#endif//_vrFileSystem_h_