// KMeansClustering.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

std::pair<std::vector<std::string>, Eigen::MatrixXd> readVectors(const std::string& vectorFile)
{
	std::vector<std::string> ids;

	std::ifstream vs{ vectorFile };
	Eigen::Index M, N;
	vs >> M >> N;

	std::cout << "Document matrix is " << M << "x" << N << std::endl;

	//Store matrix transposed such that rows are contiguous (Eigen is column major)
	Eigen::MatrixXd vectors{ N, M };

	for (Eigen::Index i = 0; i < M; i++)
	{
		std::string id;
		vs >> id;
		ids.emplace_back(id);
		for (Eigen::Index j = 0; j < N; j++)
		{
			vs >> vectors(j, i);
		}
	}
	
	return { ids, vectors };

}

int __cdecl main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " <vectorfile> <outfile>" << std::endl;
		return 0;
	}

	const std::string vectorFile{ argv[1] };
	const std::string outFile{ argv[2] };

	std::cout << "Reading " << vectorFile << "..." << std::endl;

	const auto docVecs = readVectors(vectorFile);

	std::cout << "Determining optimal value of K" << std::endl;

	std::cout << "Determining optimal clustering with K = ?" << std::endl;

	std::cout << "Writing optimal clustering to " << outFile << std::endl;

    return 0;
}

