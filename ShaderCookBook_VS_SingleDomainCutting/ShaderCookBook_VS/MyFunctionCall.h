#pragma once
#include <iostream>
#include <boost/current_function.hpp> 
class FunctionCall
{
public:
	FunctionCall(const char * s):str(s)
	{
		std::cout << "begin " << str  << std::endl;
	}
	~FunctionCall(void)
	{
		std::cout << "end " << str  << std::endl;
	}
	const char* str;
};

//#define MyFunctionCall
#define MyFunctionCall FunctionCall tmp(BOOST_CURRENT_FUNCTION)