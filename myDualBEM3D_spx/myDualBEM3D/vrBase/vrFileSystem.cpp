#ifndef _vrFileSystem_h_
#define _vrFileSystem_h_

#include "vrTypes.h"
#include <boost/filesystem.hpp>
#include <iostream>
namespace fs = boost::filesystem;
namespace VR
{
	namespace FileSystem
	{
		vrString get_currentpath()
		{
			fs::path cwd = fs::current_path();
			return cwd.generic_string();
		}

		void parser_filepath(const vrString& filePath, vrString& strParentPath, vrString& strFileName, vrString& strStem, vrString& strExtension)
		{
			const fs::path p1 = filePath;
			strParentPath = p1.parent_path().string();
			strFileName = p1.filename().string();
			strExtension = p1.extension().string();
			strStem = p1.stem().string();
		}

		vrSizt_t get_filenames(const std::string& dir, std::vector<std::string>& filenames)
		{
			fs::path path(dir);
			if (!fs::exists(path))
			{
				return -1;
			}

			fs::directory_iterator end_iter;
			for (fs::directory_iterator iter(path); iter != end_iter; ++iter)
			{
				if (fs::is_regular_file(iter->status()))
				{
					filenames.push_back(iter->path().string());
				}

				if (fs::is_directory(iter->status()))
				{
					get_filenames(iter->path().string(), filenames);
				}
			}

			return filenames.size();
		}
		//generic: D : / MyWorkspace / MyPaper / 2.Utility / MyGitTestProject / libTools / x64 / Release
		//native : D : \MyWorkspace\MyPaper\2.Utility\MyGitTestProject\libTools\x64\Release
		//quoted : "D:\MyWorkspace\MyPaper\2.Utility\MyGitTestProject\libTools\x64\Release"
		//Components :
		//[D:][/ ][MyWorkspace][MyPaper][2.Utility][MyGitTestProject][libTools][x64][Release]
		void printFileSystemInfo()
		{
			fs::path cwd = fs::current_path();
			std::cout << "generic: " << cwd.generic_string() << '\n';
			std::cout << "native: " << cwd.string() << '\n';
			std::cout << "quoted: " << cwd << '\n';

			std::cout << "Components: \n";
			for (iterAllOf(dir,cwd))
			{
				std::cout << '[' << (*dir).string() << ']'; // each part
			}
			/*for (const auto& dir : cwd) {
				std::cout << '[' << dir.string() << ']';
			}*/
			std::cout << '\n';
		}

		void printPathParts(const vrString& strPath)
		{
			const fs::path p1(strPath);
			std::cout << "For path: " << p1.string() << '\n';

			if (p1.is_relative()) {
				std::cout << "\tPath is relative\n";
			}
			else {
				assert(p1.is_absolute());
				std::cout << "\tPath is absolute\n";
			}

			if (p1.has_root_name())
				std::cout << "Root name: "
				<< p1.root_name().string() << '\n';

			if (p1.has_root_directory())
				std::cout << "Root directory: "
				<< p1.root_directory().string() << '\n';

			if (p1.has_root_path())
				std::cout << "Root path: "
				<< p1.root_path().string() << '\n';
			if (p1.has_parent_path())
				std::cout << "Parent path: "
				<< p1.parent_path().string() << '\n';

			if (p1.has_relative_path())
				std::cout << "Relative path: "
				<< p1.relative_path().string() << '\n';

			if (p1.has_filename())
				std::cout << "File name: "
				<< p1.filename().string() << '\n';

			if (p1.has_extension())
				std::cout << "Extension: "
				<< p1.extension().string() << '\n';

			if (p1.has_stem())
				std::cout << "Stem: " << p1.stem().string() << '\n';

			std::cout << '\n';
		}
	}
}
#endif//_vrFileSystem_h_