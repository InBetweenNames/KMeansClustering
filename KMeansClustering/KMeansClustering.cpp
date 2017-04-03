// KMeansClustering.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <iostream>

int __cdecl main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " <vectorfile> <outfile>" << std::endl;
		return 0;
	}

	const std::string vectorFile{ argv[1] };
	const std::string outFile{ argv[2] };

    return 0;
}

