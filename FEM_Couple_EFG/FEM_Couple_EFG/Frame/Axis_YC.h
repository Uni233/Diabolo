#ifndef AXIS_YC_H
#define AXIS_YC_H

#include "Vec_YC.h"
#include "Quater_YC.h"
#include <GL/glu.h>
#include <map>

class  Axis
{
public:
    typedef float MySReal;
    typedef Vec3<MySReal> Vector3;
    typedef Quater<MySReal> Quaternion; ///< alias
public:

    Axis(MySReal len=(MySReal)1);
    Axis(const Vector3& len);
    Axis(const Vector3& center, const Quaternion &orient, const Vector3& length);
    Axis(const Vector3& center, const double orient[4][4], const Vector3& length);
    Axis(const double *mat, const Vector3& length);
    Axis(const Vector3& center, const Quaternion &orient, MySReal length=(MySReal)1);
    Axis(const Vector3& center, const double orient[4][4], MySReal length=(MySReal)1);
    Axis(const double *mat, MySReal length=(MySReal)1.0);

    ~Axis();

    void update(const Vector3& center, const Quaternion& orient = Quaternion());
    void update(const Vector3& center, const double orient[4][4]);
    void update(const double *mat);

    void draw();

    static void draw(const Vector3& center, const Quaternion& orient, const Vector3& length);
    static void draw(const Vector3& center, const double orient[4][4], const Vector3& length);
    static void draw(const double *mat, const Vector3& length);
    static void draw(const Vector3& center, const Quaternion& orient, MySReal length=(MySReal)1);
    static void draw(const Vector3& center, const double orient[4][4], MySReal length=(MySReal)1);
    static void draw(const double *mat, MySReal length=(MySReal)1.0);

    //Draw a nice vector (cylinder + cone) given 2 points and a radius (used to draw the cylinder)
    static void draw(const Vector3& center, const Vector3& ext, const double& radius);
    //Draw a cylinder given two points and the radius of the extremities (to have a cone, simply set one radius to zero)
    static void draw(const Vector3& center, const Vector3& ext, const double& r1, const double& r2);
private:

    Vector3 length;
    double matTransOpenGL[16];

    GLUquadricObj *quadratic;
    GLuint displayList;

    void initDraw();

    static std::map < std::pair<std::pair<float,float>,float>, Axis* > axisMap;
    static Axis* get(const Vector3& len);

};

#endif // AXIS_YC_H
