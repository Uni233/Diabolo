#ifndef _vrPrintf_h_
#define _vrPrintf_h_
#include <iostream>
#include <algorithm>
#include <iterator>

#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr

namespace VR
{
	void vrPrintf(const std::string fmt_str, ...) ;

#if 0
	template<class TFirst, class... TOther>
	void vrPrintf(const char* format, TFirst&& first, TOther&&... other)
	{
		boost::format fmt(format);
		string_format(fmt, first, other...);
		//infoLog << fmt.str();
		std::cout << fmt.str() << std::endl;
	}
#endif

	template< class T >
	void vrPrintContainer(const T& container)
	{
		std::copy(container.begin(), container.end(), std::ostream_iterator< T::value_type >(std::cout, " "));
	}
}//namespace VR

#endif//_vrPrintf_h_