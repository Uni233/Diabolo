#include <igl/colon.h>
#include <igl/harmonic.h>
#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include <igl/viewer/Viewer.h>
#include <algorithm>
#include <iostream>
#include "VR_Global_Define.h"

#if USE_TBB
#include "tbb/task_scheduler_init.h"
#pragma comment(lib,"tbb.lib")
#endif


namespace YC
{
	
	extern bool pre_draw(igl::viewer::Viewer & viewer);
	extern bool key_down(igl::viewer::Viewer &viewer, unsigned char key, int mods);
	extern bool init_global_variable(igl::viewer::Viewer & viewer);
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;  
  using namespace tbb;

#if USE_TBB
  task_scheduler_init init;
#endif
  
  // Plot the mesh with pseudocolors
  igl::viewer::Viewer viewer;
  YC::init_global_variable(viewer);
  
  viewer.launch();
}
