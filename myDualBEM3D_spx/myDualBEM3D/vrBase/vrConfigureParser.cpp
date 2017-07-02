#include "vrConfigureParser.h"
//#include "vrLog.h"
#include "vrMacro.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/lexical_cast.hpp>
#include "vrIterator.h"

#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cctype>

namespace VR
{
	namespace ConfigureParser
	{
		vrString makeKey(const vrLpsz lpszSection, const vrLpsz lpszName)
		{
			return string_format("%s.%s", lpszSection, lpszName);
		}

		void getConfPropertyValueStr(const vrPropertyMap& propertyMap, const vrString& strKey, vrString& _value)
		{
			_value = propertyMap.at(strKey);
		}

		vrInt getConfPropertyValueInt(const vrPropertyMap& propertyMap, const vrString& strKey)
		{
			return boost::lexical_cast<vrInt>(vrCStr(propertyMap.at(strKey)));
		}

		vrFloat getConfPropertyValueFloat(const vrPropertyMap& propertyMap, const vrString& strKey)
		{
			return boost::lexical_cast<vrFloat>(vrCStr(propertyMap.at(strKey)));
		}

		void parser_configurefile(const vrString& strConf, vrPropertyMap& propertyMap)
		{
			boost::property_tree::ptree pt;
			boost::property_tree::ini_parser::read_ini(strConf, pt);

			for (iterAllOf(itr,propertyMap))
			{
				(*itr).second = pt.get<vrString>((*itr).first);
				//infoLog << (itr.first) << " " << (itr.second);
			}
			/*for (vrPropertyMap::iterator& itr : propertyMap)
			{
				itr.second = pt.get<vrString>(itr.first);				
			}*/
		}

		template<>
		void getConfPropertyValue< vrVec4>(const vrPropertyMap& propertyMap, const vrString& strKey, vrVec4& _value)
		{
			using namespace boost::algorithm;
			using namespace std;
			vrString str = propertyMap.at(strKey);
			str.erase(
				remove_if(str.begin(), str.end(), static_cast<int(*)(int)>(isspace)),
				str.end());
			vector< vrString > strVec;
			split(strVec, str, is_any_of(",;"));
			vrASSERT(4 == strVec.size());
			_value[0] = boost::lexical_cast<vrFloat>(vrCStr(strVec[0]));
			_value[1] = boost::lexical_cast<vrFloat>(vrCStr(strVec[1]));
			_value[2] = boost::lexical_cast<vrFloat>(vrCStr(strVec[2]));
			_value[3] = boost::lexical_cast<vrFloat>(vrCStr(strVec[3]));
		}

		template<>
		void getConfPropertyValue< vrVec3>(const vrPropertyMap& propertyMap, const vrString& strKey, vrVec3& _value)
		{
			using namespace boost::algorithm;
			using namespace std;
			vrString str = propertyMap.at(strKey);
			str.erase(
				remove_if(str.begin(), str.end(), static_cast<int(*)(int)>(isspace)),
				str.end());
			vector< vrString > strVec;
			split(strVec, str, is_any_of(",;"));
			vrASSERT(3 == strVec.size());
			_value[0] = boost::lexical_cast<vrFloat>(vrCStr(strVec[0]));
			_value[1] = boost::lexical_cast<vrFloat>(vrCStr(strVec[1]));
			_value[2] = boost::lexical_cast<vrFloat>(vrCStr(strVec[2]));
		}

		bool to_bool(const vrString& str /*need lower string : "true" or "false"*/)
		{			
			std::istringstream is(str);
			bool b;
			is >> std::boolalpha >> b;
			return b;
		}

		template<>
		void getConfPropertyValue< bool >(const vrPropertyMap& propertyMap, const vrString& strKey, bool& _value)
		{
			vrString str = propertyMap.at(strKey);
			std::transform(str.begin(), str.end(), str.begin(), ::tolower);	
			vrASSERT((0 == std::strcmp("true", vrCStr(str))) || (0 == std::strcmp("false", vrCStr(str))));
			_value = to_bool(str);
		}

	}//namespace ConfigureParser
}//namespace VR