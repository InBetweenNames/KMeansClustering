// KMeansClustering.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <random>
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

template<typename T>
double RSSk(const T& centroid, const std::vector<Eigen::Index>& cluster, const Eigen::MatrixXd& X)
{
	double sum = 0;

	for (const auto& i : cluster)
	{
		const auto diff = X.col(i) - centroid;
		sum += diff.squaredNorm();
	}

	return sum;
}

double RSS(const Eigen::MatrixXd& centroids, const std::vector<std::vector<Eigen::Index>>& clusters, const Eigen::MatrixXd& X)
{
	double sum = 0;
	const Eigen::Index K = static_cast<Eigen::Index>(clusters.size());
	for (Eigen::Index i = 0; i < K; i++)
	{
		sum += RSSk(centroids.col(i), clusters[i], X);
	}

	return sum;
}

std::tuple<std::vector<std::vector<Eigen::Index>>, double> computeKMeans(const Eigen::MatrixXd& X, const Eigen::Index K, std::mt19937& rnd)
{
	std::uniform_int_distribution<Eigen::Index> dist{ 0,X.cols() };

	Eigen::VectorXi seeds{ K };
	Eigen::MatrixXd centroids{ X.rows(), K };
	for (Eigen::Index i = 0; i < K; i++)
	{
		seeds(i) = dist(rnd);
		centroids.col(i) = X.col(seeds(i));
	}

	double prevRSS = -2;
	double currRSS = -1;

	std::vector<std::vector<Eigen::Index>> clusters;

	while (currRSS - prevRSS > 0.0001)
	{
		clusters = std::vector<std::vector<Eigen::Index>>(K);
		//Reassignment of vectors
		for (Eigen::Index i = 0; i < X.cols(); i++)
		{
			Eigen::Index index;
			//Optimization: don't compute full norm, just squared norm
			const auto minDistance = (centroids.colwise() - X.col(i)).colwise().squaredNorm().minCoeff(&index);
			clusters[index].emplace_back(i);
		}

		//Recomputation of centroids
		for (Eigen::Index i = 0; i < K; i++)
		{
			Eigen::VectorXd sum = Eigen::VectorXd::Zero(X.rows());
			for (Eigen::Index j = 0; j < clusters[i].size(); j++)
			{
				sum += X.col(clusters[i][j]);
			}

			centroids.col(i) = sum / clusters[i].size();
		}

		prevRSS = currRSS;
		currRSS = RSS(centroids, clusters, X);
	}

	return { clusters, currRSS };

}

std::tuple<std::vector<std::vector<Eigen::Index>>,Eigen::Index> findOptimalClustering(const Eigen::MatrixXd& X)
{
	//Select random seeds deterministically (use pi to show this value is not chosen favourably)
	std::mt19937 rnd{ 314159 };

	const Eigen::Index maxK = 100; //TODO: change to X.cols()
	constexpr Eigen::Index maxIter = 10;

	//Try K up to a limit
	double bestAICfactor = std::numeric_limits<double>::max();
	Eigen::Index bestK = -1;
	std::vector<std::vector<Eigen::Index>> bestClustering;

	std::vector<double> minRSSk;
	std::vector<double> AICminRSSk;
	for (Eigen::Index k = 1; k < maxK; k++)
	{
		std::cout << "Trying K = " << k << std::endl;
		double minRSS = std::numeric_limits<double>::max();
		std::vector<std::vector<Eigen::Index>> localBestClustering;
		for (Eigen::Index i = 0; i < maxIter; i++)
		{
			const auto res = computeKMeans(X, k, rnd);
			const auto rss = std::get<1>(res);
			if (rss < minRSS)
			{
				minRSS = rss;
				localBestClustering = std::get<0>(res);
			}
		}
		minRSSk.emplace_back(minRSS);
		const double AICfactor = minRSS + 2*X.rows()*k;
		AICminRSSk.emplace_back(AICfactor);
		if (AICfactor < bestAICfactor)
		{
			bestAICfactor = AICfactor;
			bestK = k;
			bestClustering = localBestClustering;
		}
	}

	return { bestClustering, bestK };
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

	std::cout << "Determining optimal clustering using heuristic" << std::endl;
	const auto res = findOptimalClustering(docVecs.second);
	const auto K = std::get<1>(res);
	const auto clusters = std::get<0>(res);

	std::cout << "Optimal clustering has K = " << K << std::endl;

	std::cout << "Writing optimal clustering to " << outFile << std::endl;

    return 0;
}

