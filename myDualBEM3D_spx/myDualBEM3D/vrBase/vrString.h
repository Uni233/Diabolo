#ifndef _vrString_h_
#define _vrString_h_


#include <string>
#include <boost/format.hpp>

#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
namespace VR
{
	typedef std::string vrString;
	
#if 0

	vrString awesome_printf_helper(boost::format& f);

	template<class T, class... Args>
	vrString awesome_printf_helper(boost::format& f, T&& t, Args&&... args){
		return awesome_printf_helper(f % std::forward<T>(t), std::forward<Args>(args)...);
	}

	template<typename... Arguments>
	vrString string_format(std::string const& fmt, Arguments&&... args)
	{
		boost::format f(fmt);
		return awesome_printf_helper(f, std::forward<Arguments>(args)...);
	}
#endif

	std::string string_format(const std::string fmt_str, ...) ;
}//namespace VR

#endif//_vrString_h_