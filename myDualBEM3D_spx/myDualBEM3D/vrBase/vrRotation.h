#ifndef _vrRotation_h_
#define _vrRotation_h_

#include "vrBase.h"
#include <boost/algorithm/clamp.hpp>
namespace VR
{
	namespace Cgmath
	{
		//!the myVector representing the origin
		const vrVec3 ORIGIN(0, 0, 0);
		//!the unit myVector associated to the x axys
		const vrVec3 X_AXIS_Coord(1, 0, 0);
		//!the unit myVector associated to the y axys
		const vrVec3 Y_AXIS_Coord(0, 1, 0);
		//!the unit myVector associated to the z axys
		const vrVec3 Z_AXIS_Coord(0, 0, 1);
		//!the greek pi constant
		const vrFloat G_PI = 3.14159265359;
		//! greek pi / 2
		const vrFloat G_HALF_PI = 1.570796326795;
		//!2 * greek pi
		const vrFloat G_DOUBLE_PI = 6.28318530718;

#if DoublePrecision
		const vrFloat epsilon = 1e-12;
#else
		const vrFloat epsilon = 1e-7;
#endif

		inline vrFloat DegToRad(const vrFloat& angle)
		{
			return angle*0.01745329251994;
		}

		inline void clamp(vrVec3& theVec, const vrFloat& low, const vrFloat& high)
		{
			boost::algorithm::clamp(theVec[0], low, high);
			boost::algorithm::clamp(theVec[1], low, high);
			boost::algorithm::clamp(theVec[2], low, high);
		}
	}

	namespace Interactive
	{
		enum AxisSet
		{
			NO_AXES,
			CAMERA_AXES,
			BODY_AXES,
			OTHER_AXES
		};

		typedef vrVec2 EPoint;
		typedef vrVec3 MyPoint;
		typedef unsigned long myCOLORREF;
		typedef vrVec4 ERect;
		typedef unsigned UINT;
		typedef vrMat3 MyMatrix_3X3;
		typedef vrMat4 tmatrix;
		typedef vrVec3 myVector;
		typedef vrFloat myReal;
#if DoublePrecision
		typedef Eigen::AngleAxisd vrAngleAxis;
#else
		typedef Eigen::AngleAxisf vrAngleAxis;
#endif
		
		typedef vrQuaternion unitquaternion;

		

		class vrBallController
		{
		public:
			void setCuttingTrack(bool flag){ m_isCuttingTrack = flag; }
			bool m_isCuttingTrack;//fault be false
			std::vector< MyPoint > m_vecTrackPoint;
			EPoint m_startPoint;
		private:
			bool bDrawConstraints;
			MyMatrix_3X3 bodyorientation;
			int angleKeyIncrement;
			void DrawConstraints();
			myVector* GetUsedAxisSet();
			myVector BallColor;
			//bool bProjectionMethod2;
			bool bDrawBallArea;
			int GLdisplayList;
			unitquaternion currentQuat;
			unitquaternion previousQuat;
			myReal radius;
			myReal winWidth;
			myReal winHeight;
			myReal xprev;
			myReal yprev;
			EPoint center;
			bool mouseButtonDown;
			AxisSet whichConstraints;
			int currentAxisIndex;
			myVector cameraAxes[3];
			myVector bodyAxes[3];
			myVector* otherAxes;
			int otherAxesNum;

			void InitDisplayLists();
			void initVars(void);
			void ProjectOnSphere(myVector& v) const;
			unitquaternion RotationFromMove(const myVector& vfrom, const myVector& vto);
			myVector ConstrainToAxis(const myVector& loose, const myVector& axis);
			int NearestConstraintAxis(const myVector& loose);
		public:
			bool GetDrawConstraints();
			void SetDrawConstraints(bool flag = true);
			void DrawBall();
			int GetAngleKeyIncrement();
			void SetAngleKeyIncrement(int ang);
			void UseConstraints(AxisSet constraints);
			void ToggleMethod();
			void SetAlternateMethod(bool flag = true);
			myVector GetColorV();
			myCOLORREF GetColor();
			void SetColor(myCOLORREF col);
			void SetColorV(myVector colvec);
			void DrawBallLimit();
			vrBallController();
			vrBallController(const myReal& rad);
			vrBallController(const myReal& rad, const unitquaternion& initialOrient);
			vrBallController(const vrBallController& other);
			virtual ~vrBallController();
			vrBallController& operator=(const vrBallController& other);
			void Resize(const myReal& newRadius);
			void ClientAreaResize(const ERect& newSize);
			void MouseDown(const EPoint& location);
			void MouseUp(const EPoint& location);
			void MouseMove(const EPoint& location);
			void IssueGLrotation();
			void Key(UINT nChar);

			MyMatrix_3X3 getRotation();
			void getRotationInfo(myReal& x, myReal& y, myReal& z, myReal& w);
		};
	}//namespace BallController
}//namespace VR
#endif//_vrRotation_h_