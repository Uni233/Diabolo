#ifndef _vrTypes_h_
#define _vrTypes_h_

#include "vrMacro.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <vector>
#include <map>
#include <set>
#include "vrString.h"
#include "vrIterator.h"
#include "vrPrintf.h"
#include <assert.h>
#include <boost/smart_ptr.hpp>
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <stddef.h>

namespace VR
{
#if DoublePrecision
	typedef double vrFloat;
	typedef int   vrInt;
	typedef unsigned int vrUnsigned;
	typedef char const * vrLpsz;
	typedef Eigen::MatrixXd vrMat;
	typedef Eigen::Vector2d vrVec2;
	typedef Eigen::Vector3d vrVec3;
	typedef Eigen::Vector4d vrVec4;
	

	typedef vrVec3 vrPoint;
	typedef Eigen::MatrixXi vrMatI;
	typedef Eigen::VectorXi vrVecI;
	typedef Eigen::Vector3i vrVec3I;
	typedef Eigen::VectorXd vrVec;
	typedef Eigen::Matrix3d vrMat3;
	typedef Eigen::Matrix4d vrMat4;
	typedef Eigen::SparseMatrix<vrFloat, vrMatStoreType> vrSpMat;

	typedef Eigen::Quaterniond vrQuaternion;

#define VR_GL_FLOAT GL_FLOAT 

	typedef glm::vec2 vrGLMVec2;
	typedef glm::vec3 vrGLMVec3;
	typedef glm::vec4 vrGLMVec4;
	typedef glm::mat3 vrGLMMat3;
	typedef glm::mat4 vrGLMMat4;

	typedef std::size_t vrSizt_t;

	struct vrInt2
	{
		vrInt x, y;
	};

	struct vrFloat2
	{
		vrGLMVec3::value_type x, y;
	};

	struct vrFloat3
	{
		vrGLMVec3::value_type x, y, z;
	};

	extern vrFloat3 make_vrFloat3(vrFloat x, vrFloat y, vrFloat z);

	struct vrFloat4
	{
		vrGLMVec3::value_type x, y, z, w;
	};

	extern vrGLMVec3 make_glm_vec3( const vrVec3& src);

#else //DoublePrecision
	typedef float vrFloat;
#define VR_GL_FLOAT GL_FLOAT 
	typedef int   vrInt;
	typedef unsigned int vrUnsigned;
	typedef char const * vrLpsz;
	typedef Eigen::MatrixXf vrMat;
	typedef Eigen::Vector2f vrVec2;
	typedef Eigen::Vector3f vrVec3;
	typedef Eigen::Vector4f vrVec4;

	typedef vrVec3 vrPoint;
	typedef Eigen::MatrixXi vrMatI;
	typedef Eigen::VectorXi vrVecI;
	typedef Eigen::Vector3i vrVec3I;
	typedef Eigen::VectorXf vrVec;
	typedef Eigen::Matrix3f vrMat3;
	typedef Eigen::Matrix4f vrMat4;
	typedef Eigen::SparseMatrix<vrFloat, vrMatStoreType> vrSpMat;

	typedef Eigen::Quaternionf vrQuaternion;

	typedef glm::vec2 vrGLMVec2;
	typedef glm::vec3 vrGLMVec3;
	typedef glm::vec4 vrGLMVec4;
	typedef glm::mat3 vrGLMMat3;
	typedef glm::mat4 vrGLMMat4;

	typedef std::size_t vrSizt_t;

	struct vrInt2
	{
		vrInt x, y;
	};

	struct vrFloat2
	{
		vrFloat x, y;
	};

	struct vrFloat3
	{
		vrFloat x, y, z;
	};

	extern vrFloat3 make_vrFloat3(vrFloat x, vrFloat y, vrFloat z);

	struct vrFloat4
	{
		vrFloat x, y, z, w;
	};

	extern vrGLMVec3 make_glm_vec3(const vrVec3& src);

#endif//DoublePrecision
	typedef enum{ FEM = 0, EFG = 1, COUPLE = 2, INVALIDTYPE = 3 } CellType;
	
	template <unsigned n>
	struct const_unsigned {
		static const unsigned value = n;
	};

	/*template <vrFloat n>
	struct const_vrFloat {
		static const vrFloat value = n;
	};*/

	template <vrInt n>
	struct const_vrInt {
		static const vrInt value = n;
	};

	namespace Geometry
	{
		/*const int shape_Function_Count_In_FEM = 8;
		const int gauss_Sample_Point = 8;
		const int n_tensor_pols = 8;
		const int dofs_per_cell = 24;
		const int dofs_per_cell_8 = 8;
		const int vertexs_per_cell = 8;
		const int dimensions_per_vertex = 3;
		const int first_dof_idx = 0;
		const int max_dofs_per_face = 12;
		const int faces_per_cell = 6;
		const int vertexs_per_face = 4;
		const int sons_per_cell = 8;
		const int lines_per_quad = 4;
		const int lines_per_cell = 12;
		const int vertexes_per_line = 2;
		const int subQuads_per_quad = 4;
		const int subLines_per_line = 2;*/
	}//namespace Geometry

	namespace Order
	{
		/*typedef enum{ vex0 = 0, vex1 = 1, vex2 = 2, vex3 = 3, vex4 = 4, vex5 = 5, vex6 = 6, vex7 = 7 } VertexOrder;
		
		static vrVec3 step[Geometry::vertexs_per_cell] = { vrVec3(-1, -1, -1), vrVec3(1, -1, -1),
			vrVec3(-1, 1, -1), vrVec3(1, 1, -1),
			vrVec3(-1, -1, 1), vrVec3(1, -1, 1),
			vrVec3(-1, 1, 1), vrVec3(1, 1, 1) };

		static unsigned int index[8][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };

		static unsigned linePairs[12][2] = { { 0, 1 }, { 4, 5 }, { 6, 7 }, { 2, 3 },
		{ 0, 2 }, { 1, 3 }, { 5, 7 }, { 4, 6 },
		{ 0, 4 }, { 1, 5 }, { 3, 7 }, { 2, 6 } };
		static unsigned int indexQuads[Geometry::faces_per_cell][Geometry::vertexs_per_face] = { { 0, 2, 4, 6 }, { 1, 3, 5, 7 }, { 0, 1, 4, 5 }, { 2, 3, 6, 7 }, { 0, 1, 2, 3 }, { 4, 5, 6, 7 } };
		static unsigned int indexQuads_Tri[Geometry::faces_per_cell][2][3] =
		{
			{ { 0, 4, 6 }, { 0, 6, 2 } },
			{ { 1, 3, 7 }, { 5, 1, 7 } },
			{ { 0, 1, 5 }, { 4, 0, 5 } },
			{ { 7, 3, 2 }, { 2, 6, 7 } },
			{ { 1, 2, 3 }, { 1, 0, 2 } },
			{ { 5, 7, 6 }, { 5, 6, 4 } }
		};*/
	}//namespace Order

	namespace Colors
	{
		typedef enum{ black = 0, blue = 1, green = 2, indigoBlue = 3, red = 4, pink = 5, yellow = 6, white = 7 } MyColors;
		static vrFloat colorTemplage[10][3] = {
			{ 0.0f, 0.0f, 0.0f },//black
			{ 0.0f, 0.0f, 1.0f },//blue
			{ 0.0f, 1.0f, 0.0f },//green
			{ 0.0f, 1.0f, 1.0f },//dian blue
			{ 1.0f, 0.0f, 0.0f },//red
			{ 1.0f, 0.0f, 1.0f },//pink
			{ 1.0f, 1.0f, 0.0f },//yellow
			{ 1.0f, 1.0f, 1.0f } };//white

		static vrVec3 purple(80.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0);
		static vrVec3 gold(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0);
	}//namespace Colors

}//namespace VR

#include "vrCoordinate.h"

#endif//_vrTypes_h_