#ifndef MAT_YC_H
#define MAT_YC_H

#include "Vec_YC.h"

template<class MyReal = float>
class Matrix3
{
public:
    Matrix3(){}
    Vec3<MyReal>& x(){return matrix[0];}
    Vec3<MyReal>& y(){return matrix[1];}
    Vec3<MyReal>& z(){return matrix[2];}
    Vec3<MyReal> matrix[3];
};

#endif // MAT_YC_H
