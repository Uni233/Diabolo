#ifndef _vrHead_h_
#define _vrHead_h_

#include "vrMacro.h"
#include "vrTypes.h"
#if vrLoggingPP
//#define ELPP_NO_DEFAULT_LOG_FILE
//#define ELPP_THREAD_SAFE
//#define ELPP_FORCE_USE_STD_THREAD
#include "log/easylogging++.h"

#define infoLog LOG(INFO)
#define debugLog LOG(DEBUG)
#define warningLog LOG(WARNING)
//#define globalLog LOG(GLOBAL)
#define traceLog LOG(TRACE)
#define fatalLog LOG(FATAL)
#define errorLog LOG(ERROR)
#define verboseLog VLOG(1)
#else
#define infoLog std::cout << std::endl 
#define errorLog std::cout << std::endl 
#endif//vrLoggingPP
#endif//_vrHead_h_