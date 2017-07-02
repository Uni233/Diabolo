#ifndef _vrConfigureParser_h_
#define _vrConfigureParser_h_
#include "vrTypes.h"
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_pod.hpp>

#include <boost/algorithm/string.hpp>
#include <vector>
namespace VR
{
	namespace ConfigureParser
	{
		typedef std::map< vrString, vrString > vrPropertyMap;
		extern vrString makeKey(const vrLpsz strSection, const vrLpsz strName);
		extern void getConfPropertyValueStr(const vrPropertyMap& propertyMap, const vrString& strKey, vrString& _value);

		template< class T >
		void getConfPropertyValue(const vrPropertyMap& propertyMap, const vrString& strKey, T& _value)
		{
			BOOST_STATIC_ASSERT(boost::is_pod< T >::value);
			_value = boost::lexical_cast<T>(vrCStr(propertyMap.at(strKey)));
		}

		template<>
		extern void getConfPropertyValue< vrVec4>(const vrPropertyMap& propertyMap, const vrString& strKey, vrVec4& _value);
		
		template<>
		extern void getConfPropertyValue< vrVec3>(const vrPropertyMap& propertyMap, const vrString& strKey, vrVec3& _value);

		template<>
		extern void getConfPropertyValue< bool >(const vrPropertyMap& propertyMap, const vrString& strKey, bool& _value);

		extern vrInt getConfPropertyValueInt(const vrPropertyMap& propertyMap, const vrString& strKey);
		extern vrFloat getConfPropertyValueFloat(const vrPropertyMap& propertyMap, const vrString& strKey);
		extern void parser_configurefile(const vrString& strConf, vrPropertyMap& propertyMap);

#if 0
		// Nouse_Test_boost.cpp : 定义控制台应用程序的入口点。
		//

#include "stdafx.h"
#include <string>
#include <iostream>

#include <boost/format.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
		int _tmain(int argc, _TCHAR* argv[])
		{

			std::wcout.imbue(std::locale("chs"));
			// 待分割的字符串
			std::wstring strTag = _T("I Come from China");

			std::vector<std::wstring> vecSegTag;
			// boost::is_any_of这里相当于分割规则了
			boost::split(vecSegTag, strTag, boost::is_any_of(_T(" ,，")));

			for (size_t i = 0; i < vecSegTag.size(); i++)
			{
				std::wcout << vecSegTag[i] << std::endl;
			}

			vecSegTag.clear();
			std::wstring strTag2 = _T("我叫小明，你呢,今天天气不错");
			boost::split(vecSegTag, strTag2, boost::is_any_of(_T(" ,，")));

			for (size_t i = 0; i < vecSegTag.size(); i++)
			{
				std::wcout << vecSegTag[i] << std::endl;
			}
			getchar();
			return 0;
		}


#endif
		
	}//namespace ConfigureParser
}//namespace VR

#endif//_vrConfigureParser_h_