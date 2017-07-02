#include "stdafx.h"

#include "objParser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

extern std::string g_strMeshId;
extern std::vector< VR_FEM::MyPoint > g_vecMaterialPoint;

namespace VR_FEM
{
	MyFloat objParser::minIn3(MyFloat a, MyFloat b, MyFloat c)const
	{
		return (a < b) ? 
			( a < c ? a : c ) :
			( b < c ? b : c);
	}

	MyFloat objParser::maxIn3(MyFloat a, MyFloat b, MyFloat c)const
	{
		return (a > b) ? 
			(a > c ? a : c) : 
			(b > c ? b : c);
	}

	bool objParser::parser(const char* lpszFilePath)
	{
		
		std::ifstream infile(lpszFilePath);
		std::cout << "begin parser obj file : " << lpszFilePath << std::endl;
        Q_ASSERT(infile.is_open());
		Q_ASSERT(m_dataPtr);
		clear();
		std::vector< MyDenseVector >& vertices = m_dataPtr->vertices;
		std::vector< MyDenseVector >& verticeNormals = m_dataPtr->verticeNormals;
		std::vector< MyDenseVector >& normalizeVertices = m_dataPtr->normalizeVertices;
		std::vector< std::pair<MyFloat,MyFloat> >& coords = m_dataPtr->coords;
		std::vector< MyVectorI >& face_indicies = m_dataPtr->face_indicies;
		std::vector< MyVectorI >& vertexNormal_indicies = m_dataPtr->vertexNormal_indicies;
		std::vector< MyVectorI >& coord_indicies = m_dataPtr->coord_indicies;

		MyFloat& m_translation_x = m_dataPtr->m_translation_x;
		MyFloat& m_translation_y = m_dataPtr->m_translation_y;
		MyFloat& m_translation_z = m_dataPtr->m_translation_z;
		MyFloat& m_maxDiameter = m_dataPtr->m_maxDiameter;

		MyInt& m_nCoordCount = m_dataPtr->m_nCoordCount;
		MyInt& m_nVerticeCount = m_dataPtr->m_nVerticeCount;
		MyInt& m_nVerticeNormalCount = m_dataPtr->m_nVerticeNormalCount;
		MyInt& m_nTriangleCount = m_dataPtr->m_nTriangleCount;

        std::stringstream ss;
        MyFloat db1,db2,db3;
        int i1,i2,i3;
        char * lpszLineBuffer = new char[BufferSize];
		int nFirstIdx;
		Eigen::Vector3i tmpFaceIdx,tmpNormalIdx,tmpCoordIdx;
		MyDenseVector   tmpVertexNormal,tmpVertex;
		std::pair<MyFloat,MyFloat> tmpCoord;
        while(!infile.eof())
        {
            std::stringstream ss;
            memset(lpszLineBuffer,'\0',BufferSize);
            infile.getline(lpszLineBuffer,BufferSize);
            if ('v' == lpszLineBuffer[0] && ' ' == lpszLineBuffer[1])
            {
				nFirstIdx = 1;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpVertex[0] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertex[1] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertex[2] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				vertices.push_back(tmpVertex);
				/*std::cout << tmpVertex << std::endl;
				exit(66);*/
				
			}
			else if ('v' == lpszLineBuffer[0] && 'n' == lpszLineBuffer[1])
			{
				nFirstIdx = 2;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpVertexNormal[0] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertexNormal[1] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertexNormal[2] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				verticeNormals.push_back(tmpVertexNormal);
				/*std::cout << tmpVertexNormal << std::endl;
				exit(66);*/
			}
            else if ('f' == lpszLineBuffer[0])
            {
				nFirstIdx = 1;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				
				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
													boost::algorithm::is_any_of( " /\t\n\r" ),
													boost::algorithm::token_compress_on 
												  ) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpFaceIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				if (m_hasCoord)
				{
					tmpCoordIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
					++iStr;
				}
				if (m_hasVerticeNormal)
				{
					tmpNormalIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
					++iStr;
				}				

				tmpFaceIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				if (m_hasCoord)
				{
					tmpCoordIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
					++iStr;
				}
				if (m_hasVerticeNormal)
				{
					tmpNormalIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
					++iStr;
				}

				tmpFaceIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				if (m_hasCoord)
				{
					tmpCoordIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
					++iStr;
				}
				if (m_hasVerticeNormal)
				{
					tmpNormalIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
					++iStr;
				}

				face_indicies.push_back(tmpFaceIdx);

				if (m_hasCoord)
				{
					coord_indicies.push_back(tmpCoordIdx);
				}
				if (m_hasVerticeNormal)
				{
					vertexNormal_indicies.push_back(tmpNormalIdx);	
				}
				
            }
			else if ('v' == lpszLineBuffer[0] && 't' == lpszLineBuffer[1])
			{
				nFirstIdx = 2;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpCoord.first = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpCoord.second = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;

				coords.push_back(tmpCoord);
				
			}			
            else if ('#' == lpszLineBuffer[0])
            {
                //printf(lpszLineBuffer);
            }
        }
		
        printf("[vertices is %d] [verticeNormals is %d][coords is %d][face is %d]\n",vertices.size(),verticeNormals.size(),coords.size(),face_indicies.size());
        delete []lpszLineBuffer;


        {
            //normalize
            MyFloat rx,ry,rz;
            MyFloat xmin = FLT_MAX ,xmax = -FLT_MAX ,ymin = FLT_MAX ,ymax = -FLT_MAX ,zmin = FLT_MAX ,zmax = -FLT_MAX ;

            const long verticesNum = vertices.size();
            for (int i = 0; i < verticesNum; ++i)
            {
                rx = vertices[i][0];
                if (xmin > rx)
                {
                    xmin = rx;
                }
                if (xmax < rx)
                {
                    xmax = rx;
                }

                ry = vertices[i][1];
                if (ymin > ry)
                {
                    ymin = ry;
                }
                if (ymax < ry)
                {
                    ymax = ry;
                }

                rz = vertices[i][2];
                if (zmin > rz)
                {
                    zmin = rz;
                }
                if (zmax < rz)
                {
                    zmax = rz;
                }

            }

            printf("x : %f,%f \t  y : %f,%f \t  z : %f,%f \t",xmax,xmin,ymax,ymin,zmax,zmin);

            MyFloat maxDiameter,minDiameter,maxRadius,minRadius,X_diameter(xmax - xmin),Y_diameter(ymax - ymin),Z_diameter(zmax - zmin);
			MyFloat X_radius(X_diameter/2.f),Y_radius(Y_diameter/2.f),Z_radius(Z_diameter/2.f);
			MyFloat massCenter_x(xmin + X_diameter / 2.f),massCenter_y(ymin + Y_diameter/2.f),massCenter_z(zmin + Z_diameter/2.f);
		
			maxDiameter = maxIn3(X_diameter	, Y_diameter, Z_diameter);
			minDiameter = minIn3(X_diameter	, Y_diameter, Z_diameter);
			maxRadius   = maxIn3(X_radius	, Y_radius	, Z_radius);
			minRadius   = minIn3(X_radius	, Y_radius	, Z_radius);

			float translation_x(maxRadius-massCenter_x), translation_y(maxRadius-massCenter_y), translation_z(maxRadius-massCenter_z);

			
			float x_scale = 1.0 / (maxDiameter) ;
			float y_scale = 1.0 / (maxDiameter) ;
			float z_scale = 1.0 / (maxDiameter) ;
			printf("maxDiameter is %f translation_x(%f,%f,%f) \n",maxDiameter,translation_x, translation_y, translation_z);
			m_maxDiameter = maxDiameter;
			m_translation_x = translation_x; 
			m_translation_y = translation_y;
			m_translation_z = translation_z;
		
			normalizeVertices.clear();
			for (int j=0;j< vertices.size() ; ++j)
			{
				MyDenseVector& refPoint = vertices[j];
				normalizeVertices.push_back(MyDenseVector( x_scale * (refPoint.x()+ translation_x) ,
													y_scale * (refPoint.y() + translation_y) ,
													z_scale * (refPoint.z() + translation_z) ));
			}

			//Q_ASSERT(g_strMeshId == std::string("armadillo"));
			
        }

		m_nCoordCount = coords.size();
		m_nVerticeCount = vertices.size();
		m_nVerticeNormalCount = verticeNormals.size();
		m_nTriangleCount = face_indicies.size();
		return true;
	}

	void objParser::makeScalarSamplePoint(ObjParserData* dataPtr,const std::vector< MyPoint >& vecNativePointSet , std::vector< MyPoint >& vecScalarPointSet)
	{
		Q_ASSERT(dataPtr);
		float x_scale = 1.0 / (dataPtr->m_maxDiameter) ;
		float y_scale = 1.0 / (dataPtr->m_maxDiameter) ;
		float z_scale = 1.0 / (dataPtr->m_maxDiameter) ;

		vecScalarPointSet.clear();
		const MyInt nSamplePointSize = vecNativePointSet.size();
		for (MyInt v=0;v<nSamplePointSize;++v)
		{
			vecScalarPointSet.push_back(MyDenseVector(  x_scale * (vecNativePointSet[v][0]+ dataPtr->m_translation_x) ,
														y_scale * (vecNativePointSet[v][1] + dataPtr->m_translation_y) ,
														z_scale * (vecNativePointSet[v][2] + dataPtr->m_translation_z) ));
		}
	}

	void objParser::clear()
	{
		Q_ASSERT(m_dataPtr);
		std::vector< MyDenseVector >& vertices = m_dataPtr->vertices;
		std::vector< MyDenseVector >& verticeNormals = m_dataPtr->verticeNormals;
		std::vector< MyDenseVector >& normalizeVertices = m_dataPtr->normalizeVertices;
		std::vector< std::pair<MyFloat,MyFloat> >& coords = m_dataPtr->coords;
		std::vector< MyVectorI >& face_indicies = m_dataPtr->face_indicies;
		std::vector< MyVectorI >& vertexNormal_indicies = m_dataPtr->vertexNormal_indicies;
		std::vector< MyVectorI >& coord_indicies = m_dataPtr->coord_indicies;

		MyFloat& m_translation_x = m_dataPtr->m_translation_x;
		MyFloat& m_translation_y = m_dataPtr->m_translation_y;
		MyFloat& m_translation_z = m_dataPtr->m_translation_z;
		MyFloat& m_maxDiameter = m_dataPtr->m_maxDiameter;

		MyInt& m_nCoordCount = m_dataPtr->m_nCoordCount;
		MyInt& m_nVerticeCount = m_dataPtr->m_nVerticeCount;
		MyInt& m_nVerticeNormalCount = m_dataPtr->m_nVerticeNormalCount;
		MyInt& m_nTriangleCount = m_dataPtr->m_nTriangleCount;

		bool& m_hasCoord = m_dataPtr->m_hasCoord;
		bool& m_hasVerticeNormal = m_dataPtr->m_hasVerticeNormal;

		m_nCoordCount = MyZero;
		m_nVerticeCount = MyZero;
		m_nVerticeNormalCount = MyZero;
		m_nTriangleCount = MyZero;

		m_hasCoord = false;
		m_hasVerticeNormal = false;
		/*delete [] m_arrayCoord;
		delete [] m_arrayVertice;
		delete [] m_arrayVerticeNormal;

		delete [] m_arrayTriangleIndice;
		delete [] m_arrayCoordIndice;
		delete [] m_arrayVerticeNormalIndice;

		m_arrayCoord = MyNull;
		m_arrayVertice = MyNull;
		m_arrayVerticeNormal = MyNull;

		m_arrayTriangleIndice = MyNull;
		m_arrayCoordIndice = MyNull;
		m_arrayVerticeNormalIndice = MyNull;*/


		m_translation_x = MyZero;
		m_translation_y = MyZero;
		m_translation_z = MyZero;
		m_maxDiameter = 1.f;

		vertices.clear();
		verticeNormals.clear();
		normalizeVertices.clear();
		coords.clear();
		face_indicies.clear();
		vertexNormal_indicies.clear();
		coord_indicies.clear();
	}
}