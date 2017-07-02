#ifndef QUATER_H
#define QUATER_H

#include <assert.h>
#include <limits>
#include <math.h>
#include <iostream>
#include <stdio.h>

#include "Vec_YC.h"
#include "Mat_YC.h"




#define RENORMCOUNT 50

#define EQUALITY_THRESHOLD (0.0000001f)

template<class MyReal>
class  Quater
{
private:
    MyReal _q[4];

public:

    typedef MyReal value_type;

    Quater();
    ~Quater();
    Quater(MyReal x, MyReal y, MyReal z, MyReal w);
    template<class Real2>
    Quater(const Real2 q[]) { for (int i=0; i<4; i++) _q[i] = (MyReal)q[i]; }
    template<class Real2>
    Quater(const Quater<Real2>& q) { for (int i=0; i<4; i++) _q[i] = (MyReal)q[i]; }
    Quater( const Vec3<MyReal>& axis, MyReal angle );

    static Quater identity()
    {
        return Quater(0,0,0,1);
    }


    /// Cast into a standard C array of elements.
    const MyReal* ptr() const
    {
        return this->_q;
    }

    /// Cast into a standard C array of elements.
    MyReal* ptr()
    {
        return this->_q;
    }

    /// Normalize a quaternion
    void normalize();

    void clear()
    {
        _q[0]=0.0;
        _q[1]=0.0;
        _q[2]=0.0;
        _q[3]=1.0;
    }

    //void fromFrame(defaulttype::Vec<3,MyReal>& x, defaulttype::Vec<3,MyReal>&y, defaulttype::Vec<3,MyReal>&z);
    void fromFrame(Vec3<MyReal>& x, Vec3<MyReal>&y, Vec3<MyReal>&z);


    //void fromMatrix(const defaulttype::Matrix3 &m);
    void fromMatrix( Matrix3<MyReal> &m);

    template<class Mat33>
    void toMatrix(Mat33 &m) const
    {
		m.coeffRef(0,0)/*[0][0]*/ = /*(typename Mat33::Real)*/ (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
		m.coeffRef(0,1)/*[0][1]*/ = /*(typename Mat33::Real)*/ (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
		m.coeffRef(0,2)/*[0][2]*/ = /*(typename Mat33::Real)*/ (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));

		m.coeffRef(1,0)/*[1][0]*/ = /*(typename Mat33::Real)*/ (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
		m.coeffRef(1,1)/*[1][1]*/ = /*(typename Mat33::Real)*/ (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
		m.coeffRef(1,2)/*[1][2]*/ = /*(typename Mat33::Real) */(2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));

		m.coeffRef(2,0)/*[2][0]*/ = /*(typename Mat33::Real)*/ (2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
		m.coeffRef(2,1)/*[2][1]*/ =/* (typename Mat33::Real)*/ (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
		m.coeffRef(2,2)/*[2][2]*/ = /*(typename Mat33::Real)*/ (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
    }

    /// Apply the rotation to a given vector
    template<class Vec>
    Vec rotate( const Vec& v ) const
    {
        return Vec(
                (typename Vec::value_type)((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0f * (_q[0] * _q[1] - _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] + _q[1] * _q[3])) * v[2]),
                (typename Vec::value_type)((2.0f * (_q[0] * _q[1] + _q[2] * _q[3]))*v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]))*v[2]),
                (typename Vec::value_type)((2.0f * (_q[2] * _q[0] - _q[1] * _q[3]))*v[0] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]))*v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2])
                );

    }

    /// Apply the inverse rotation to a given vector
    template<class Vec>
    Vec inverseRotate( const Vec& v ) const
    {
        return Vec(
                (typename Vec::value_type)((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0f * (_q[0] * _q[1] + _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] - _q[1] * _q[3])) * v[2]),
                (typename Vec::value_type)((2.0f * (_q[0] * _q[1] - _q[2] * _q[3]))*v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]))*v[2]),
                (typename Vec::value_type)((2.0f * (_q[2] * _q[0] + _q[1] * _q[3]))*v[0] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]))*v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2])
                );

    }

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    //template <class T>
    //friend Quater<T> operator+(Quater<T> q1, Quater<T> q2);
    Quater<MyReal> operator+(const Quater<MyReal> &q1) const;

    Quater<MyReal> operator*(const Quater<MyReal> &q1) const;

    Quater<MyReal> operator*(const MyReal &r) const;
    Quater<MyReal> operator/(const MyReal &r) const;
    void operator*=(const MyReal &r);
    void operator/=(const MyReal &r);

    /// Given two Quaters, multiply them together to get a third quaternion.
    //template <class T>
    //friend Quater<T> operator*(const Quater<T>& q1, const Quater<T>& q2);

    //Quater quatVectMult(const defaulttype::Vec<3,MyReal>& vect);
    Quater quatVectMult(const Vec3<MyReal>& vect);

    //Quater vectQuatMult(const defaulttype::Vec<3,MyReal>& vect);
    Quater vectQuatMult(const Vec3<MyReal>& vect);

    MyReal& operator[](int index)
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    const MyReal& operator[](int index) const
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    Quater inverse() const;

    //defaulttype::Vec<3,MyReal> toEulerVector() const;
    Vec3<MyReal> toEulerVector() const;


    /*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.

     \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.

     When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
     between the Quaternions' orientations, by "flipping" the source Quaternion if needed (see
     negate()). */
    void slerp(const Quater& a, const Quater& b, MyReal t, bool allowFlip=true);

    // A useful function, builds a rotation matrix in Matrix based on
    // given quaternion.

    void buildRotationMatrix(MyReal m[4][4]) const;
    void writeOpenGlMatrix( double* m ) const;
    void writeOpenGlMatrix( float* m ) const;

    //void buildRotationMatrix(MATRIX4x4 m);

    //void buildRotationMatrix(Matrix &m);

    // This function computes a quaternion based on an axis (defined by
    // the given vector) and an angle about which to rotate.  The angle is
    // expressed in radians.

    //Quater axisToQuat(defaulttype::Vec<3,MyReal> a, MyReal phi);
    //void quatToAxis(defaulttype::Vec<3,MyReal> & a, MyReal &phi);
    Quater axisToQuat(Vec3<MyReal> a, MyReal phi);
    void quatToAxis(Vec3<MyReal> & a, MyReal &phi);


    //static Quater createQuaterFromFrame(const defaulttype::Vec<3, MyReal> &lox, const defaulttype::Vec<3, MyReal> &loy,const defaulttype::Vec<3, MyReal> &loz);
    static Quater createQuaterFromFrame(const Vec3<MyReal> &lox, const Vec3<MyReal> &loy,const Vec3<MyReal> &loz);

    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quater createFromRotationVector(const V& a)
    {
        MyReal phi = (MyReal)sqrt(a*a);
        if( phi < 1.0e-5 )
            return Quater(0,0,0,1);
        else
        {
            MyReal nor = 1/phi;
            MyReal s = (MyReal)sin(phi/2);
            return Quater( a[0]*s*nor, a[1]*s*nor,a[2]*s*nor, (MyReal)cos(phi/2) );
        }
    }

    /// Create a quaternion from Euler
    static Quater createQuaterFromEuler( Vec3<MyReal> v)
    {
        MyReal quat[4];      MyReal a0 = v.elems[0];
        MyReal a1 = v.elems[1];
        MyReal a2 = v.elems[2];
        quat[3] = cos(a0/2)*cos(a1/2)*cos(a2/2) + sin(a0/2)*sin(a1/2)*sin(a2/2);
        quat[0] = sin(a0/2)*cos(a1/2)*cos(a2/2) - cos(a0/2)*sin(a1/2)*sin(a2/2);
        quat[1] = cos(a0/2)*sin(a1/2)*cos(a2/2) + sin(a0/2)*cos(a1/2)*sin(a2/2);
        quat[2] = cos(a0/2)*cos(a1/2)*sin(a2/2) - sin(a0/2)*sin(a1/2)*cos(a2/2);
        Quater quatResult( quat[0], quat[1], quat[2], quat[3] );
        return quatResult;
    }

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quater createFromRotationVector(T a0, T a1, T a2 )
    {
        MyReal phi = (MyReal)sqrt((MyReal)(a0*a0+a1*a1+a2*a2));
        if( phi < 1.0e-5 )
            return Quater(0,0,0,1);
        else
        {
            MyReal nor = 1/phi;
            MyReal s = (MyReal)sin(phi/2.0);
            return Quater( a0*s*nor, a1*s*nor,a2*s*nor, (MyReal)cos(phi/2.0) );
        }
    }
    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quater set(const V& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quater set(T a0, T a1, T a2) { return createFromRotationVector(a0,a1,a2); }

    /// Return the quaternion resulting of the movement between 2 quaternions
    Quater quatDiff( Quater a, const Quater& b)
    {
        // If the axes are not oriented in the same direction, flip the axis and angle of a to get the same convention than b
        if (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]<0)
        {
            a[0] = -a[0];
            a[1] = -a[1];
            a[2] = -a[2];
            a[3] = -a[3];
        }

        Quater q = b.inverse() * a;
        return q;
    }

    /// Return the eulerian vector resulting of the movement between 2 quaternions
    Vec3<MyReal> angularDisplacement( Quater a, const Quater& b)
    {
        return quatDiff(a,b).toEulerVector();
    }


    // Print the quaternion (C style)
    void print();
    Quater<MyReal> slerp(Quater<MyReal> &q1, MyReal t);
    Quater<MyReal> slerp2(Quater<MyReal> &q1, MyReal t);

    void operator+=(const Quater& q2);
    void operator*=(const Quater& q2);

    bool operator==(const Quater& q) const
    {
        for (int i=0; i<4; i++)
            if ( fabs( _q[i] - q._q[i] ) > EQUALITY_THRESHOLD ) return false;
        return true;
    }

    bool operator!=(const Quater& q) const
    {
        for (int i=0; i<4; i++)
            if ( fabs( _q[i] - q._q[i] ) > EQUALITY_THRESHOLD ) return true;
        return false;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const Quater& v )
    {
        out<<v._q[0]<<" "<<v._q[1]<<" "<<v._q[2]<<" "<<v._q[3];
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, Quater& v )
    {
        in>>v._q[0]>>v._q[1]>>v._q[2]>>v._q[3];
        return in;
    }

    enum { static_size = 4 };
    static unsigned int size() {return 4;};

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 4 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for quaternions)
    enum { spatial_dimensions = 3 };
};

// Constructor
template<class MyReal>
Quater<MyReal>::Quater()
{
    _q[0] = _q[1] = _q[2] = 0.0;
    _q[3] = 1.0;
}

template<class MyReal>
Quater<MyReal>::Quater(MyReal x, MyReal y, MyReal z, MyReal w)
{
    _q[0] = x;
    _q[1] = y;
    _q[2] = z;
    _q[3] = w;
}

template<class MyReal>
Quater<MyReal>::Quater( const Vec3<MyReal>& axis, MyReal angle )
{
    axisToQuat(axis,angle);
}

// Destructor
template<class MyReal>
Quater<MyReal>::~Quater()
{
}

/// Given two rotations, e1 and e2, expressed as quaternion rotations,
/// figure out the equivalent single rotation and stuff it into dest.
/// This routine also normalizes the result every RENORMCOUNT times it is
/// called, to keep error from creeping in.
///   NOTE: This routine is written so that q1 or q2 may be the same
///  	   as dest (or each other).
template<class MyReal>
//Quater<MyReal> operator+(Quater<MyReal> q1, Quater<MyReal> q2) const
Quater<MyReal> Quater<MyReal>::operator+(const Quater<MyReal> &q1) const
{
    static int	count	= 0;

    MyReal		t1[4], t2[4], t3[4];
    MyReal		tf[4];
    Quater<MyReal>	ret;

    t1[0] = _q[0] * q1._q[3];
    t1[1] = _q[1] * q1._q[3];
    t1[2] = _q[2] * q1._q[3];

    t2[0] = q1._q[0] * _q[3];
    t2[1] = q1._q[1] * _q[3];
    t2[2] = q1._q[2] * _q[3];

    // cross product t3 = q2 x q1
    t3[0] = (q1._q[1] * _q[2]) - (q1._q[2] * _q[1]);
    t3[1] = (q1._q[2] * _q[0]) - (q1._q[0] * _q[2]);
    t3[2] = (q1._q[0] * _q[1]) - (q1._q[1] * _q[0]);
    // end cross product

    tf[0] = t1[0] + t2[0] + t3[0];
    tf[1] = t1[1] + t2[1] + t3[1];
    tf[2] = t1[2] + t2[2] + t3[2];
    tf[3] = _q[3] * q1._q[3] -
            (_q[0] * q1._q[0] + _q[1] * q1._q[1] + _q[2] * q1._q[2]);

    ret._q[0] = tf[0];
    ret._q[1] = tf[1];
    ret._q[2] = tf[2];
    ret._q[3] = tf[3];

    if (++count > RENORMCOUNT)
    {
        count = 0;
        ret.normalize();
    }

    return ret;
}

template<class MyReal>
//Quater<MyReal> operator*(const Quater<MyReal>& q1, const Quater<MyReal>& q2) const
Quater<MyReal> Quater<MyReal>::operator*(const Quater<MyReal>& q1) const
{
    Quater<MyReal>	ret;

    ret._q[3] = _q[3] * q1._q[3] -
            (_q[0] * q1._q[0] +
                    _q[1] * q1._q[1] +
                    _q[2] * q1._q[2]);
    ret._q[0] = _q[3] * q1._q[0] +
            _q[0] * q1._q[3] +
            _q[1] * q1._q[2] -
            _q[2] * q1._q[1];
    ret._q[1] = _q[3] * q1._q[1] +
            _q[1] * q1._q[3] +
            _q[2] * q1._q[0] -
            _q[0] * q1._q[2];
    ret._q[2] = _q[3] * q1._q[2] +
            _q[2] * q1._q[3] +
            _q[0] * q1._q[1] -
            _q[1] * q1._q[0];

    return ret;
}

template<class MyReal>
Quater<MyReal> Quater<MyReal>::operator*(const MyReal& r) const
{
    Quater<MyReal>  ret;
    ret[0] = _q[0] * r;
    ret[1] = _q[1] * r;
    ret[2] = _q[2] * r;
    ret[3] = _q[3] * r;
    return ret;
}


template<class MyReal>
Quater<MyReal> Quater<MyReal>::operator/(const MyReal& r) const
{
    Quater<MyReal>  ret;
    ret[0] = _q[0] / r;
    ret[1] = _q[1] / r;
    ret[2] = _q[2] / r;
    ret[3] = _q[3] / r;
    return ret;
}

template<class MyReal>
void Quater<MyReal>::operator*=(const MyReal& r)
{
    Quater<MyReal>  ret;
    _q[0] *= r;
    _q[1] *= r;
    _q[2] *= r;
    _q[3] *= r;
}


template<class MyReal>
void Quater<MyReal>::operator/=(const MyReal& r)
{
    Quater<MyReal>  ret;
    _q[0] /= r;
    _q[1] /= r;
    _q[2] /= r;
    _q[3] /= r;
}


template<class MyReal>
Quater<MyReal> Quater<MyReal>::quatVectMult(const Vec3<MyReal>& vect)
{
    Quater<MyReal>	ret;

    ret._q[3] = (MyReal) (-(_q[0] * vect[0] + _q[1] * vect[1] + _q[2] * vect[2]));
    ret._q[0] = (MyReal) (_q[3] * vect[0] + _q[1] * vect[2] - _q[2] * vect[1]);
    ret._q[1] = (MyReal) (_q[3] * vect[1] + _q[2] * vect[0] - _q[0] * vect[2]);
    ret._q[2] = (MyReal) (_q[3] * vect[2] + _q[0] * vect[1] - _q[1] * vect[0]);

    return ret;
}

template<class MyReal>
Quater<MyReal> Quater<MyReal>::vectQuatMult(const Vec3<MyReal>& vect)
{
    Quater<MyReal>	ret;

    ret[3] = (MyReal) (-(vect[0] * _q[0] + vect[1] * _q[1] + vect[2] * _q[2]));
    ret[0] = (MyReal) (vect[0] * _q[3] + vect[1] * _q[2] - vect[2] * _q[1]);
    ret[1] = (MyReal) (vect[1] * _q[3] + vect[2] * _q[0] - vect[0] * _q[2]);
    ret[2] = (MyReal) (vect[2] * _q[3] + vect[0] * _q[1] - vect[1] * _q[0]);

    return ret;
}

template<class MyReal>
Quater<MyReal> Quater<MyReal>::inverse() const
{
    Quater<MyReal>	ret;

    MyReal		norm	= sqrt(_q[0] * _q[0] +
            _q[1] * _q[1] +
            _q[2] * _q[2] +
            _q[3] * _q[3]);

    if (norm != 0.0f)
    {
        norm = 1.0f / norm;
        ret._q[3] = _q[3] * norm;
        for (int i = 0; i < 3; i++)
        {
            ret._q[i] = -_q[i] * norm;
        }
    }
    else
    {
        for (int i = 0; i < 4; i++)
        {
            ret._q[i] = 0.0;
        }
    }

    return ret;
}

/// Quater<MyReal>s always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
/// If they don't add up to 1.0, dividing by their magnitude will
/// renormalize them.
template<class MyReal>
void Quater<MyReal>::normalize()
{
    int		i;
    MyReal	mag;

    mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
    if( mag != 0)
    {
        for (i = 0; i < 4; i++)
        {
            _q[i] /= sqrt(mag);
        }
    }
}

template<class MyReal>
void Quater<MyReal>::fromFrame(Vec3<MyReal>& x, Vec3<MyReal>&y, Vec3<MyReal>&z)
{

    Matrix3<MyReal> R(x,y,z);
    R.transpose();
    this->fromMatrix(R);


}

template<class MyReal>
void Quater<MyReal>::fromMatrix( Matrix3<MyReal> &m)
{
    MyReal tr, s;

    tr = (MyReal)(m.x().x() + m.y().y() + m.z().z());

    // check the diagonal
    if (tr > 0)
    {
        s = (float)sqrt (tr + 1);
        _q[3] = s * 0.5f; // w OK
        s = 0.5f / s;
        _q[0] = (MyReal)((m.z().y() - m.y().z()) * s); // x OK
        _q[1] = (MyReal)((m.x().z() - m.z().x()) * s); // y OK
        _q[2] = (MyReal)((m.y().x() - m.x().y()) * s); // z OK
    }
    else
    {
        if (m.y().y() > m.x().x() && m.z().z() <= m.y().y())
        {
            s = (MyReal)sqrt ((m.y().y() - (m.z().z() + m.x().x())) + 1.0f);

            _q[1] = s * 0.5f; // y OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[2] = (MyReal)((m.y().z() + m.z().y()) * s); // z OK
            _q[0] = (MyReal)((m.x().y() + m.y().x()) * s); // x OK
            _q[3] = (MyReal)((m.x().z() - m.z().x()) * s); // w OK
        }
        else if ((m.y().y() <= m.x().x()  &&  m.z().z() > m.x().x())  ||  (m.z().z() > m.y().y()))
        {
            s = (MyReal)sqrt ((m.z().z() - (m.x().x() + m.y().y())) + 1.0f);

            _q[2] = s * 0.5f; // z OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[0] = (MyReal)((m.z().x() + m.x().z()) * s); // x OK
            _q[1] = (MyReal)((m.y().z() + m.z().y()) * s); // y OK
            _q[3] = (MyReal)((m.y().x() - m.x().y()) * s); // w OK
        }
        else
        {
            s = (MyReal)sqrt ((m.x().x() - (m.y().y() + m.z().z())) + 1.0f);

            _q[0] = s * 0.5f; // x OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[1] = (MyReal)((m.x().y() + m.y().x()) * s); // y OK
            _q[2] = (MyReal)((m.z().x() + m.x().z()) * s); // z OK
            _q[3] = (MyReal)((m.z().y() - m.y().z()) * s); // w OK
        }
    }
}

// template<class MyReal> template<class Mat33>
//     void Quater<MyReal>::toMatrix(Mat33 &m) const
// {
// 	m[0][0] = (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
// 	m[0][1] = (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
// 	m[0][2] = (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
//
// 	m[1][0] = (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
// 	m[1][1] = (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
// 	m[1][2] = (float) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));
//
// 	m[2][0] = (float) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
// 	m[2][1] = (float) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
// 	m[2][2] = (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
// }

/// Build a rotation matrix, given a quaternion rotation.
template<class MyReal>
void Quater<MyReal>::buildRotationMatrix(MyReal m[4][4]) const
{
    m[0][0] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0][1] = (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0][2] = (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[0][3] = 0;

    m[1][0] = (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1][1] = (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[1][2] = (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[1][3] = 0;

    m[2][0] = (2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[2][1] = (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2][2] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[2][3] = 0;

    m[3][0] = 0;
    m[3][1] = 0;
    m[3][2] = 0;
    m[3][3] = 1;
}
/// Write an OpenGL rotation matrix
/*template<class MyReal>
void Quater<MyReal>::writeOpenGlMatrix(double *m) const
{
    m[0*4+0] = (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0*4+1] = (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0*4+2] = (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[0*4+3] = 0.0f;

    m[1*4+0] = (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1*4+1] = (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[1*4+2] = (float) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[1*4+3] = 0.0f;

    m[2*4+0] = (float) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[2*4+1] = (float) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2*4+2] = (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[2*4+3] = 0.0f;

    m[3*4+0] = 0.0f;
    m[3*4+1] = 0.0f;
    m[3*4+2] = 0.0f;
    m[3*4+3] = 1.0f;
}
*/
/// Write an OpenGL rotation matrix
template<class MyReal>
void Quater<MyReal>::writeOpenGlMatrix(double *m) const
{
    m[0*4+0] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[1*4+0] = (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[2*4+0] = (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[3*4+0] = 0.0f;

    m[0*4+1] = (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1*4+1] = (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[2*4+1] = (float) (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[3*4+1] = 0.0f;

    m[0*4+2] = (float) (2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[1*4+2] = (float) (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2*4+2] = (float) (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[3*4+2] = 0.0f;

    m[0*4+3] = 0.0f;
    m[1*4+3] = 0.0f;
    m[2*4+3] = 0.0f;
    m[3*4+3] = 1.0f;
}

/// Write an OpenGL rotation matrix
template<class MyReal>
void Quater<MyReal>::writeOpenGlMatrix(float *m) const
{
    m[0*4+0] = (float) (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[1*4+0] = (float) (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[2*4+0] = (float) (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[3*4+0] = 0.0f;

    m[0*4+1] = (float) (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1*4+1] = (float) (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[2*4+1] = (float) (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[3*4+1] = 0.0f;

    m[0*4+2] = (float) (2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[1*4+2] = (float) (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2*4+2] = (float) (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[3*4+2] = 0.0f;

    m[0*4+3] = 0.0f;
    m[1*4+3] = 0.0f;
    m[2*4+3] = 0.0f;
    m[3*4+3] = 1.0f;
}

/// Given an axis and angle, compute quaternion.
template<class MyReal>
Quater<MyReal> Quater<MyReal>::axisToQuat(Vec3<MyReal> a, MyReal phi)
{
    if( a.norm() < std::numeric_limits<MyReal>::epsilon() )
    {
//		std::cout << "zero norm quaternion" << std::endl;
        _q[0] = _q[1] = _q[2] = (MyReal)0.0f;
        _q[3] = (MyReal)1.0f;

        return Quater();
    }

   // a = a / a.norm();
	a.normalize();
    _q[0] = (MyReal)a.x();
    _q[1] = (MyReal)a.y();
    _q[2] = (MyReal)a.z();

    _q[0] = _q[0] * (MyReal)sin(phi / 2.0);
    _q[1] = _q[1] * (MyReal)sin(phi / 2.0);
    _q[2] = _q[2] * (MyReal)sin(phi / 2.0);

    _q[3] = (MyReal)cos(phi / 2.0);

    return *this;
}

/// Given a quaternion, compute an axis and angle
template<class MyReal>
void Quater<MyReal>::quatToAxis(Vec3<MyReal> & a, MyReal &phi)
{
    const double  sine  = sin( acos(_q[3]) );

    if (!sine)
        a = Vec3<MyReal>(0.0,1.0,0.0);
    else
        a = Vec3<MyReal>(_q[0],_q[1],_q[2])/ sine;

    phi =  (MyReal) (acos(_q[3]) * 2.0) ;
}


template<class MyReal>
Vec3<MyReal> Quater<MyReal>::toEulerVector() const
{
    Quater<MyReal> q = *this;
    q.normalize();

    double angle = acos(q._q[3]) * 2;

    Vec3<MyReal> v(q._q[0], q._q[1], q._q[2]);

    double norm = sqrt( (double) (v.x() * v.x() + v.y() * v.y() + v.z() * v.z()) );
    if (norm > 0.0005)
    {
        v.elems /= norm;
		
        v.elems *= angle;
    }

    return v;
}

/*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.

 \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.

 When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
 between the Quaternions' orientations, by "flipping" the source Quaternion if needed (see
 negate()). */
template<class MyReal>
void Quater<MyReal>::slerp(const Quater& a, const Quater& b, MyReal t, bool allowFlip)
{
    MyReal cosAngle =  (MyReal)(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);

    MyReal c1, c2;
    // Linear interpolation for close orientations
    if ((1.0 - fabs(cosAngle)) < 0.01)
    {
        c1 = 1.0f - t;
        c2 = t;
    }
    else
    {
        // Spherical interpolation
        MyReal angle    = (MyReal)acos((MyReal)fabs((MyReal)cosAngle));
        MyReal sinAngle = (MyReal)sin((MyReal)angle);
        c1 = (MyReal)sin(angle * (1.0f - t)) / sinAngle;
        c2 = (MyReal)sin(angle * t) / sinAngle;
    }

    // Use the shortest path
    if (allowFlip && (cosAngle < 0.0f))
        c1 = -c1;

    _q[0] = c1*a[0] + c2*b[0];
    _q[1] = c1*a[1] + c2*b[1];
    _q[2] = c1*a[2] + c2*b[2];
    _q[3] = c1*a[3] + c2*b[3];
}

///// Output quaternion
//template<class MyReal>
//    std::ostream& operator<<(std::ostream& out, Quater<MyReal> Q)
//{
//	return (out << "(" << Q._q[0] << "," << Q._q[1] << "," << Q._q[2] << ","
//				<< Q._q[3] << ")");
//}

template<class MyReal>
Quater<MyReal> Quater<MyReal>::slerp(Quater<MyReal> &q1, MyReal t)
{
    Quater<MyReal> q0_1;
    for (unsigned int i = 0 ; i<3 ; i++)
        q0_1[i] = -_q[i];

    q0_1[3] = _q[3];

    q0_1 = q1 * q0_1;

    Vec3<MyReal> axis, temp;
    MyReal angle;

    q0_1.quatToAxis(axis, angle);

    temp = axis * sin(t * angle);
    for (unsigned int i = 0 ; i<3 ; i++)
        q0_1[i] = temp[i];

    q0_1[3] = cos(t * angle);
    q0_1 = q0_1 * (*this);
    return q0_1;
}

// Given an axis and angle, compute quaternion.
template<class MyReal>
Quater<MyReal> Quater<MyReal>::slerp2(Quater<MyReal> &q1, MyReal t)
{
    // quaternion to return
    Quater<MyReal> qm;

    // Calculate angle between them.
    double cosHalfTheta = _q[3] * q1[3] + _q[0] * q1[0] + _q[1] * q1[1] + _q[2] * q1[2];
    // if qa=qb or qa=-qb then theta = 0 and we can return qa
    if (fabs(cosHalfTheta) >= 1.0)
    {
        qm[3] = _q[3]; qm[0] = _q[0]; qm[1] = _q[1]; qm[2] = _q[2];
        return qm;
    }
    // Calculate temporary values.
    double halfTheta = acos(cosHalfTheta);
    double sinHalfTheta = sqrt(1.0 - cosHalfTheta*cosHalfTheta);
    // if theta = 180 degrees then result is not fully defined
    // we could rotate around any axis normal to qa or qb
    if (fabs(sinHalfTheta) < 0.001)  // fabs is floating point absolute
    {
        qm[3] = (MyReal)(_q[3] * 0.5 + q1[3] * 0.5);
        qm[0] = (MyReal)(_q[0] * 0.5 + q1[0] * 0.5);
        qm[1] = (MyReal)(_q[1] * 0.5 + q1[1] * 0.5);
        qm[2] = (MyReal)(_q[2] * 0.5 + q1[2] * 0.5);
        return qm;
    }
    double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
    double ratioB = sin(t * halfTheta) / sinHalfTheta;
    //calculate Quaternion.
    qm[3] = (MyReal)(_q[3] * ratioA + q1[3] * ratioB);
    qm[0] = (MyReal)(_q[0] * ratioA + q1[0] * ratioB);
    qm[1] = (MyReal)(_q[1] * ratioA + q1[1] * ratioB);
    qm[2] = (MyReal)(_q[2] * ratioA + q1[2] * ratioB);
    return qm;

}

template<class MyReal>
Quater<MyReal> Quater<MyReal>::createQuaterFromFrame(const Vec3<MyReal> &lox, const Vec3<MyReal> &loy,const Vec3<MyReal> &loz)
{
    Quater<MyReal> q;
    Matrix3<MyReal> m;

    for (unsigned int i=0 ; i<3 ; i++)
    {
        m[i][0] = lox[i];
        m[i][1] = loy[i];
        m[i][2] = loz[i];
    }
    q.fromMatrix(m);
    return q;
}

/// Print quaternion (C style)
template<class MyReal>
void Quater<MyReal>::print()
{
    printf("(%f, %f ,%f, %f)\n", _q[0], _q[1], _q[2], _q[3]);
}

template<class MyReal>
void Quater<MyReal>::operator+=(const Quater<MyReal>& q2)
{
    static int	count	= 0;

    MyReal t1[4], t2[4], t3[4];
    Quater<MyReal> q1 = (*this);
    t1[0] = q1._q[0] * q2._q[3];
    t1[1] = q1._q[1] * q2._q[3];
    t1[2] = q1._q[2] * q2._q[3];

    t2[0] = q2._q[0] * q1._q[3];
    t2[1] = q2._q[1] * q1._q[3];
    t2[2] = q2._q[2] * q1._q[3];

    // cross product t3 = q2 x q1
    t3[0] = (q2._q[1] * q1._q[2]) - (q2._q[2] * q1._q[1]);
    t3[1] = (q2._q[2] * q1._q[0]) - (q2._q[0] * q1._q[2]);
    t3[2] = (q2._q[0] * q1._q[1]) - (q2._q[1] * q1._q[0]);
    // end cross product

    _q[0] = t1[0] + t2[0] + t3[0];
    _q[1] = t1[1] + t2[1] + t3[1];
    _q[2] = t1[2] + t2[2] + t3[2];
    _q[3] = q1._q[3] * q2._q[3] -
            (q1._q[0] * q2._q[0] + q1._q[1] * q2._q[1] + q1._q[2] * q2._q[2]);

    if (++count > RENORMCOUNT)
    {
        count = 0;
        normalize();
    }
}

template<class MyReal>
void Quater<MyReal>::operator*=(const Quater<MyReal>& q1)
{
    Quater<MyReal> q2 = *this;
    _q[3] = q2._q[3] * q1._q[3] -
            (q2._q[0] * q1._q[0] +
                    q2._q[1] * q1._q[1] +
                    q2._q[2] * q1._q[2]);
    _q[0] = q2._q[3] * q1._q[0] +
            q2._q[0] * q1._q[3] +
            q2._q[1] * q1._q[2] -
            q2._q[2] * q1._q[1];
    _q[1] = q2._q[3] * q1._q[1] +
            q2._q[1] * q1._q[3] +
            q2._q[2] * q1._q[0] -
            q2._q[0] * q1._q[2];
    _q[2] = q2._q[3] * q1._q[2] +
            q2._q[2] * q1._q[3] +
            q2._q[0] * q1._q[1] -
            q2._q[1] * q1._q[0];
}


#endif // QUATER_H
