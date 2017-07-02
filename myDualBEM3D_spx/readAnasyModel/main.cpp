#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

typedef std::string vrString;
typedef char const * vrLpsz;
typedef std::size_t vrSizt_t;
typedef float vrFloat;
typedef int vrInt;

struct node
{
	float x,y,z;
};

struct element
{
	int x,y,z;
};

struct boundary
{
	int parent0,parent1;
	int edgeVtx0,edgeVtx1;
};

void trimString( vrString & str ) 
{
	char const * whiteSpace = " \t\n\r";
	vrSizt_t location;
	location = str.find_first_not_of(whiteSpace);
	str.erase(0,location);
	location = str.find_last_not_of(whiteSpace);
	str.erase(location + 1);
}

int main()
{

	std::string fileName("D:/MyWorkspace/1.MyCode/1.PhysicCode/2.BEM/FractureBEM-master-4-21/FractureBEM-master/examples/notched_bar.nodes");
	std::ifstream objStream(fileName);

	using namespace std;

	std::vector< node > points;
	std::vector< element > elements;
	std::vector< boundary > elements_boundary;

	
	vrInt nFaces = 0;

	if( !objStream ) {
		std::cout << "Unable to open OBJ file: " << fileName;
		exit(66);
	}

	vrString line, token;
	std::vector< vrInt > face;

	getline( objStream, line );
	while( !objStream.eof() ) {
		trimString(line);
		if( line.length( ) > 0 ) {
			istringstream lineStream( line );

			node currentNode;
			int nouse1,nouse2;
			lineStream >> nouse1 >> nouse2 >> currentNode.x >> currentNode.y >> currentNode.z;
			printf("%d %d %f %f %f\n",nouse1,nouse2,currentNode.x , currentNode.y , currentNode.z);
			points.push_back(currentNode);
		}
		getline( objStream, line );
	}

	objStream.close();

	fileName = std::string ("D:/MyWorkspace/1.MyCode/1.PhysicCode/2.BEM/FractureBEM-master-4-21/FractureBEM-master/examples/notched_bar.elements");
	objStream = std::ifstream(fileName);
	if( !objStream ) {
		std::cout << "Unable to open OBJ file: " << fileName;
		exit(66);
	}

	getline( objStream, line );
	while( !objStream.eof() ) {
		trimString(line);
		if( line.length( ) > 0 ) {
			istringstream lineStream( line );

			element currentElement;
			int nouse1,nouse2,nouse3;
			lineStream >> nouse1 >> nouse2 >> nouse3 >> currentElement.x >> currentElement.y >> currentElement.z;
			printf("%d %d %f %f %f\n",nouse1,nouse2,currentElement.x , currentElement.y , currentElement.z);
			elements.push_back(currentElement);
		}
		getline( objStream, line );
	}

	objStream.close();

	fileName = std::string ("D:/MyWorkspace/1.MyCode/1.PhysicCode/2.BEM/FractureBEM-master-4-21/FractureBEM-master/examples/notched_bar.boundary");
	objStream = std::ifstream(fileName);
	if( !objStream ) {
		std::cout << "Unable to open OBJ file: " << fileName;
		exit(66);
	}
	
	getline( objStream, line );
	while( !objStream.eof() ) {
		trimString(line);
		if( line.length( ) > 0 ) {
			istringstream lineStream( line );

			boundary currentElement;
			int nouse1,nouse2;
			lineStream >> nouse1 >> nouse2 >> currentElement.parent0 >> currentElement.parent1 >> currentElement.edgeVtx0 >> currentElement.edgeVtx1;
			printf("%d %d %f %f %f\n",nouse1,nouse2,currentElement.parent0 , currentElement.parent1 , currentElement.edgeVtx0 ,currentElement.edgeVtx1);
			elements_boundary.push_back(currentElement);
		}
		getline( objStream, line );
	}

	objStream.close();

	std::ofstream outfile("d:\\v_notched_bar.obj");

	for (int i=0;i<points.size();++i)
	{
		outfile << "v " << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
	}

	for (int i=0;i<elements.size();++i)
	{
		outfile << "f " << elements[i].x << " " << elements[i].y << " " << elements[i].z << std::endl;
	}

	outfile.close();


	std::ofstream outfile_boundary("d:\\v_notched_bar_boundary.obj");

	for (int i=0;i<points.size();++i)
	{
		outfile_boundary << "v " << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
	}

	for (int i=0;i<elements_boundary.size();++i)
	{
		int tri0_id = elements_boundary[i].parent0;
		int tri1_id = elements_boundary[i].parent1;

		element tri0 = elements[tri0_id-1];
		element tri1 = elements[tri1_id-1];

		outfile_boundary << "# shared line vertex pair: " << elements_boundary[i].edgeVtx0 << " " << elements_boundary[i].edgeVtx1 << std::endl;
		outfile_boundary << "f " << tri0.x << " " << tri0.y << " " << tri0.z << std::endl;
		outfile_boundary << "f " << tri1.x << " " << tri1.y << " " << tri1.z << std::endl;
		outfile_boundary << std::endl;

	}

	outfile_boundary.close();
}