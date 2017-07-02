#ifndef VEC_YC_H
#define VEC_YC_H

#include <Eigen/Core>
#include <Eigen/Dense>

template<class MyReal = float>
class Vec3
{
public:
    Vec3(){elems.setZero();}
    Vec3(MyReal x,MyReal y,MyReal z):elems(x,y,z){}
    Vec3(const Eigen::Vector3f& v):elems(v){}
    MyReal operator[](int index) const
    {
        return elems[index];
    }

    Vec3<MyReal> operator-(const Vec3<MyReal>& v1)const
    {
        return Vec3<MyReal>(elems - v1.elems);
    }

    Vec3<MyReal> operator*(const MyReal& scalar)const
    {
        return Vec3<MyReal>(elems*scalar);
    }

    Vec3<MyReal> operator+(const Vec3<MyReal>& v1)const
    {
        return Vec3<MyReal>(elems + v1.elems);
    }

    Vec3<MyReal> cross( Vec3<MyReal>& v1)
    {
        return Vec3<MyReal>(elems.cross( v1.elems));
    }

    void normalize()
    {
        elems.normalize();
    }

	MyReal norm()
	{
		return elems.norm();
	}

    MyReal& x(){return elems[0];}
    MyReal& y(){return elems[1];}
    MyReal& z(){return elems[2];}

    Eigen::Vector3f elems;
};

#endif // VEC_YC_H
