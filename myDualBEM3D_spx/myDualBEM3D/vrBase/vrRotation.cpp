#include "vrRotation.h"

#include <GL/freeglut.h>
namespace VR
{
	
	namespace Interactive
	{
		//---------------------------------------------------------------------------
		// inlines
		void EpsilonCorrect(myVector& vecDst, const myVector& vecSrc)
		{
			if (std::abs(vecDst[0]) < Cgmath::epsilon && std::abs(vecDst[1]) < Cgmath::epsilon && std::abs(vecDst[2]) < Cgmath::epsilon)
			{
				vecDst = vecSrc;
			}
		}

		inline vrBallController::~vrBallController()
		{
			if (otherAxes) delete[] otherAxes;
		}

		inline vrBallController::vrBallController(const vrBallController& other) :m_isCuttingTrack(false)
		{
			*this = other;
		}

		inline void vrBallController::Resize(const myReal& newRadius)
		{
			radius = newRadius;
		}

		void vrBallController::ClientAreaResize(const ERect& newSize)
		{

			winWidth = myReal(newSize[2] - newSize[0]/*newSize.Width()*/);
			winHeight = myReal(newSize[3] - newSize[1]/*newSize.Height()*/);
			//	center=EPoint( (newSize.right-newSize.left)/2 , (newSize.bottom-newSize.top)/2);
		}

		inline vrBallController::vrBallController() :m_isCuttingTrack(false)
		{
			initVars();
		}

		/*inline myCOLORREF vrBallController::GetColor()
		{
		#define RGB(r,g,b)          ((COLORREF)(((BYTE)(r)|((WORD)((BYTE)(g))<<8))|(((DWORD)(BYTE)(b))<<16)))
		return RGB(int(BallColor.x()*255),int(BallColor.y()*255),int(BallColor.z()*255));
		}*/

		inline myVector vrBallController::GetColorV()
		{
			return BallColor;
		}

		inline void vrBallController::SetAlternateMethod(bool flag)
		{
			//bProjectionMethod2 = flag;
		}

		inline void vrBallController::ToggleMethod()
		{
			//if (bProjectionMethod2) bProjectionMethod2 = false;
			//else bProjectionMethod2 = true;
		}

		inline void vrBallController::UseConstraints(AxisSet constraints)
		{
			whichConstraints = constraints;
		}

		inline int vrBallController::GetAngleKeyIncrement()
		{
			return angleKeyIncrement;
		}

		inline void vrBallController::SetAngleKeyIncrement(int ang)
		{
			angleKeyIncrement = abs(ang) % 360;
		}

		inline bool vrBallController::GetDrawConstraints()
		{
			return bDrawConstraints;
		}

		void vrBallController::SetDrawConstraints(bool flag)
		{
			bDrawConstraints = flag;
		}

		vrBallController::vrBallController(const myReal& rad) :m_isCuttingTrack(false)
		{
			initVars();
			radius = boost::algorithm::clamp(rad, 0.1, 1);
		}


		vrBallController::vrBallController(const myReal& rad, const unitquaternion& initialOrient) :m_isCuttingTrack(false)
		{
			initVars();
			radius = boost::algorithm::clamp(rad, 0.1, 1);
			currentQuat = initialOrient;
		}

		vrBallController& vrBallController::operator=(const vrBallController& other)
		{
			if (this == &other) return *this;
			initVars();
			currentQuat = other.currentQuat;
			previousQuat = other.previousQuat;
			radius = other.radius;
			winWidth = other.winWidth;
			winHeight = other.winHeight;
			otherAxesNum = other.otherAxesNum;
			otherAxes = new myVector[otherAxesNum];
			for (int c = 0; c<otherAxesNum; c++) otherAxes[c] = other.otherAxes[c];
			BallColor = other.BallColor;
			return *this;
		}

		void vrBallController::MouseDown(const EPoint& location)
		{
			xprev = (2 * location[0] - winWidth) / winWidth;
			yprev = (winHeight - 2 * location[1]) / winHeight;

			previousQuat = currentQuat;
			mouseButtonDown = true;
			if (m_isCuttingTrack)
			{
				m_vecTrackPoint.clear();
				m_startPoint = location;
			}
		}


		void vrBallController::MouseUp(const EPoint& location)
		{
			mouseButtonDown = false;
			xprev = yprev = 0.0;
			bDrawBallArea = false;
			// save current rotation axes for bodyAxes constraint at next rotation
			bodyorientation = currentQuat.matrix();
			bodyAxes[0] = myVector(bodyorientation(0, 0), bodyorientation(1, 0), bodyorientation(2, 0));
			bodyAxes[1] = myVector(bodyorientation(0, 1), bodyorientation(1, 1), bodyorientation(2, 1));
			bodyAxes[2] = myVector(bodyorientation(0, 2), bodyorientation(1, 2), bodyorientation(2, 2));

			//	vdown=vcurr=ORIGIN;// per rubberbanding
		}


		void vrBallController::MouseMove(const EPoint& location)
		{
			myReal xcurr = (2 * location[0] - winWidth) / winWidth;
			myReal ycurr = (winHeight - 2 * location[1]) / winHeight;
			myVector vfrom(xprev, yprev, 0);
			myVector vto(xcurr, ycurr, 0);

			if (mouseButtonDown)
			{
				// find the two points on sphere according to the projection method
				ProjectOnSphere(vfrom);				
				ProjectOnSphere(vto);
				
				if (m_isCuttingTrack)
				{
					m_vecTrackPoint.push_back(MyPoint(vfrom.x(), vfrom.y(), vfrom.z()));
				}
				// modify the vectors according to the active constraint
				if (whichConstraints != NO_AXES)
				{
					myVector* axisSet = GetUsedAxisSet();
					vfrom = ConstrainToAxis(vfrom, axisSet[currentAxisIndex]);
					vto = ConstrainToAxis(vto, axisSet[currentAxisIndex]);
				};
				// get the corresponding unitquaternion
				unitquaternion lastQuat = RotationFromMove(vfrom, vto);			
				{
					//currentQuat *= lastQuat;
					unitquaternion& other = lastQuat;
					myReal temp = currentQuat.w();
					currentQuat.w() = currentQuat.w()*other.w() - currentQuat.vec().dot(other.vec());
					currentQuat.vec() = temp*other.vec() + other.w()*currentQuat.vec() + (currentQuat.vec().cross( other.vec()));
				}
				
				xprev = xcurr;
				yprev = ycurr;
			}
			else if (whichConstraints != NO_AXES)
			{
				ProjectOnSphere(vto);
				currentAxisIndex = NearestConstraintAxis(vto);
			};
		}

		void vrBallController::getRotationInfo(myReal& x, myReal& y, myReal& z, myReal& w)
		{
			x = currentQuat.x();
			y = currentQuat.y();
			z = currentQuat.z();
			w = currentQuat.w();
		}

		MyMatrix_3X3 vrBallController::getRotation()
		{
			MyMatrix_3X3 rt = currentQuat.matrix().transpose();
			return rt;
		}

		void vrBallController::IssueGLrotation()
		{
			//vrExit;
#ifdef DoublePrecision
			//glMultMatrixd(currentQuat.matrix().data());
			tmatrix tmp; tmp.setZero(); 
			tmp.block(0,0,3,3) = currentQuat.matrix().transpose();
			tmp.coeffRef(3,3) = 1.0;
			glMultMatrixd(tmp.data());
#else
			glMultMatrixf(currentQuat.matrix().data());
#endif
		}

		void vrBallController::ProjectOnSphere(myVector& v) const
		{
			myReal rsqr = radius*radius;
			myReal dsqr = v.x()*v.x() + v.y()*v.y();
			{
				// if relatively "inside" sphere project to sphere else on hyperbolic sheet
				if (dsqr<(rsqr*0.5))	v.z() = sqrt(rsqr - dsqr);
				else v.z() = rsqr / (2 * sqrt(dsqr));
			};
		}

		unitquaternion vrBallController::RotationFromMove(const myVector& vfrom, const myVector& vto)
		{
			{
				// calculate axis of rotation and correct it to avoid "near zero length" rot axis
				myVector rotaxis = (vto.cross(vfrom));
				EpsilonCorrect(rotaxis, Cgmath::X_AXIS_Coord);
				// find the amount of rotation
				myVector d(vfrom - vto);
				myReal t = d.norm() / (2.0*radius);
				boost::algorithm::clamp(t, -1.0, 1.0);
				myReal phi = 2.0*asin(t);
				
				myVector ror = std::sin(phi / 2)*rotaxis.normalized();
				return unitquaternion(std::cos(phi / 2), ror.x(), ror.y(), ror.z());
				/*return unitquaternion(Eigen::AngleAxisd(phi, rotaxis));
				return unitquaternion(std::cos(phi / 2), rotaxis.x(), rotaxis.y(), rotaxis.z());*/
			}
		}

		void vrBallController::Key(UINT nChar)
		{
			switch (nChar)
			{
			case VK_UP:
			case VK_NUMPAD8:
				currentQuat *= unitquaternion(Cgmath::DegToRad(angleKeyIncrement), Cgmath::X_AXIS_Coord.x(), Cgmath::X_AXIS_Coord.y(), Cgmath::X_AXIS_Coord.z());
				break;
			case VK_DOWN:
			case VK_NUMPAD2:
				currentQuat *= unitquaternion(Cgmath::DegToRad(-angleKeyIncrement), Cgmath::X_AXIS_Coord.x(), Cgmath::X_AXIS_Coord.y(), Cgmath::X_AXIS_Coord.z());
				break;
			case VK_RIGHT:
			case VK_NUMPAD6:
				currentQuat *= unitquaternion(Cgmath::DegToRad(angleKeyIncrement), Cgmath::Y_AXIS_Coord.x(), Cgmath::Y_AXIS_Coord.y(), Cgmath::Y_AXIS_Coord.z());
				break;
			case VK_LEFT:
			case VK_NUMPAD4:
				currentQuat *= unitquaternion(Cgmath::DegToRad(-angleKeyIncrement), Cgmath::Y_AXIS_Coord.x(), Cgmath::Y_AXIS_Coord.y(), Cgmath::Y_AXIS_Coord.z());
				break;
			case VK_PRIOR:
			case VK_NUMPAD9:
				currentQuat *= unitquaternion(Cgmath::DegToRad(angleKeyIncrement), Cgmath::Z_AXIS_Coord.x(), Cgmath::Z_AXIS_Coord.y(), Cgmath::Z_AXIS_Coord.z());
				break;
			case VK_HOME:
			case VK_NUMPAD7:
				currentQuat *= unitquaternion(Cgmath::DegToRad(-angleKeyIncrement), Cgmath::Z_AXIS_Coord.x(), Cgmath::Z_AXIS_Coord.y(), Cgmath::Z_AXIS_Coord.z());
				break;
			case VK_DELETE:
			case VK_NUMPAD5:
				currentQuat = unitquaternion(0, Cgmath::X_AXIS_Coord.x(), Cgmath::X_AXIS_Coord.y(), Cgmath::X_AXIS_Coord.z());
				break;
			case VK_ESCAPE:
				currentQuat = previousQuat;
				MouseUp(EPoint(-1, -1));
				break;
			case VK_TAB:
				if (mouseButtonDown && whichConstraints != NO_AXES)
				{
					currentAxisIndex = (currentAxisIndex + 1) % 3;
					currentQuat = previousQuat;
				}
				break;
			};
		}



		void vrBallController::DrawBallLimit()
		{
		}

		void vrBallController::InitDisplayLists()
		{
			GLdisplayList = glGenLists(1);
			if (GLdisplayList != 0)
			{
				GLUquadricObj* qobj = gluNewQuadric();
				gluQuadricDrawStyle(qobj, GLU_LINE);
				gluQuadricNormals(qobj, GLU_NONE);
				glNewList(GLdisplayList, GL_COMPILE);
				gluDisk(qobj, radius, radius, 24, 1);
				glEndList();
				gluDeleteQuadric(qobj);
			}
		}

		void vrBallController::initVars()
		{
			winWidth = winHeight = 0;
			previousQuat = currentQuat = unitquaternion(0, 1, 0, 0);
			mouseButtonDown = bDrawBallArea/* = bProjectionMethod2*/ = bDrawConstraints = false;
			//bProjectionMethod2 = true;
			xprev = yprev = 0.0;
			center = EPoint(0, 0),
				radius = 0.6;
			GLdisplayList = currentAxisIndex = otherAxesNum = 0;
			BallColor = myVector(0.0, 0.5, 1.0);
			otherAxes = NULL;
			whichConstraints = NO_AXES;
			cameraAxes[0] = bodyAxes[0] = Cgmath::X_AXIS_Coord;
			cameraAxes[1] = bodyAxes[1] = Cgmath::Y_AXIS_Coord;
			cameraAxes[2] = bodyAxes[2] = Cgmath::Z_AXIS_Coord;
			bodyorientation = bodyorientation.Identity();//.loadIdentity();
			angleKeyIncrement = 5;
		}

		void vrBallController::SetColor(myCOLORREF col)
		{
			BallColor.x() = GetRValue(col) / 255.0;
			BallColor.y() = GetGValue(col) / 255.0;
			BallColor.z() = GetBValue(col) / 255.0;
		}

		void vrBallController::SetColorV(myVector colvec)
		{
			Cgmath::clamp(colvec, 0, 1);
			BallColor = colvec;
		}

		myVector vrBallController::ConstrainToAxis(const myVector& loose, const myVector& axis)
		{
			myVector onPlane;
			register myReal norm;
			onPlane = loose - axis*(axis.dot(loose));
			norm = onPlane.norm();
			if (norm > 0)
			{
				if (onPlane.z() < 0.0) onPlane = -onPlane;
				return (onPlane /= sqrt(norm));
			};
			if (axis.z() == 1) onPlane = Cgmath::X_AXIS_Coord;
			else
			{
				onPlane = myVector(-axis.y(), axis.x(), 0);
				onPlane.normalize();
			}
			return (onPlane);
		}

		int vrBallController::NearestConstraintAxis(const myVector& loose)
		{
			myVector* axisSet = GetUsedAxisSet();
			myVector onPlane;
			register vrFloat max, dot;
			register vrInt i, nearest;
			max = -1;
			nearest = 0;
			if (whichConstraints == OTHER_AXES)
			{
				for (i = 0; i<otherAxesNum; i++)
				{
					onPlane = ConstrainToAxis(loose, axisSet[i]);
					dot = onPlane.dot(loose);
					if (dot>max) max = dot; nearest = i;
				}
			}
			else
			{
				for (i = 0; i<3; i++)
				{
					onPlane = ConstrainToAxis(loose, axisSet[i]);
					dot = onPlane.dot(loose);
					if (dot>max)
					{
						max = dot;
						nearest = i;
					};
				}
			};
			return (nearest);
		}

		myVector* vrBallController::GetUsedAxisSet()
		{
			myVector* axes = NULL;
			switch (whichConstraints)
			{
			case CAMERA_AXES: axes = cameraAxes;
				break;
			case BODY_AXES: axes = bodyAxes;
				break;
			case OTHER_AXES: axes = otherAxes;
				break;
			};
			return axes;
		}

		void vrBallController::DrawConstraints()
		{
		}

		void vrBallController::DrawBall()
		{
			// change the projection matrix to identity (no view transformation )
			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
			glLoadIdentity();
			// reset the transformations
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
			// prepare the circle display list the first time
			if (GLdisplayList == 0) InitDisplayLists();
			// disable lighting and depth testing
			glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_TEST);
			glDisable(GL_LIGHTING);
			glDisable(GL_DEPTH_TEST);
			// draw the constraints or the ball limit if appropriate
			if (bDrawConstraints && whichConstraints != NO_AXES) DrawConstraints();
			else if (bDrawBallArea) DrawBallLimit();
			glPopAttrib();
			// restore the modelview and projection matrices
			glPopMatrix();
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
			glMatrixMode(GL_MODELVIEW);
		}
	}//namespace BallController
}//namespace VR