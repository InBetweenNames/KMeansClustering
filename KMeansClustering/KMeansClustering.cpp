// KMeansClustering.cpp : Defines the entry point for the console application.
//

#define EIGEN_MAX_ALIGN_BYTES 32

#include <Eigen/Core>
#include <atomic>
#include <fstream>
#include <mutex>
#include <iostream>
#include <random>
#include <string>
#include <thread>
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

double RSSk(const Eigen::VectorXd& centroid, const std::vector<Eigen::Index>& cluster, const Eigen::MatrixXd& X)
{

	Eigen::MatrixXd clusterVectors{ X.rows(), cluster.size() };

	for (size_t i = 0; i < cluster.size(); i++)
	{
		clusterVectors.col(i) = X.col(cluster[i]);
	}

	const auto diffs = clusterVectors.colwise() - centroid;
	const auto sum = diffs.colwise().squaredNorm().sum();

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

	//double prevRSS = std::numeric_limits<double>::lowest();
	//double currRSS = -1;

	std::vector<std::vector<Eigen::Index>> clusters;

	int nIterations = 0;
	//Perform 10 iterations maximum
	while (/*std::abs(currRSS - prevRSS) > 100 &&*/ nIterations < 15)
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

			for (const auto j : clusters[i])
			{
				sum += X.col(j);
			}

			centroids.col(i) = sum / clusters[i].size();
		}

		/*prevRSS = currRSS;
		currRSS = RSS(centroids, clusters, X);*/
		nIterations++;
	}

	//std::cout << "Converged in " << nIterations << " iterations" << std::endl;

	double currRSS = RSS(centroids, clusters, X);

	return { clusters, currRSS };

}

bool is_file_exist(const std::string&fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

std::tuple<std::vector<std::vector<Eigen::Index>>,Eigen::Index> findOptimalClustering(const Eigen::MatrixXd& X)
{

	const Eigen::Index maxK = X.cols();
	constexpr Eigen::Index maxIter = 10;

	//Try K up to a limit
	double bestAICfactor = std::numeric_limits<double>::max();
	Eigen::Index bestK = -1;
	std::vector<std::vector<Eigen::Index>> bestClustering;

	bool exists = is_file_exist("optimal.csv");

	std::mutex fileMutex;
	std::ofstream testedKs{ "optimal.csv", std::ofstream::app };
	if (!exists)
	{
		testedKs << "K,minRSS,AICminRSSk" << std::endl;
	}

	std::atomic<Eigen::Index> nextK{ 10 };

	auto&& f = [&bestAICfactor, &bestK, &bestClustering, &fileMutex, &testedKs, &X, &nextK, maxIter, maxK]()
	{
		//Select random seeds completely randomly

		std::mt19937 rnd{ std::random_device{}() };

		bool running = true;
		while (running)
		{
			const auto k = nextK.fetch_add(10);
			if (k > maxK)
			{
				running = false;
				return;
			}

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
			//minRSSk.emplace_back(minRSS);
			const double AICfactor = minRSS + 2 * X.rows()*k;
			fileMutex.lock();
			testedKs << k << "," << minRSS << "," << AICfactor << std::endl;
			if (AICfactor < bestAICfactor)
			{
				bestAICfactor = AICfactor;
				bestK = k;
				bestClustering = localBestClustering;
			}
			fileMutex.unlock();
			//AICminRSSk.emplace_back(AICfactor);

		}
	};
	
	const auto nThreads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (size_t i = 0; i < nThreads; i++)
	{
		threads.emplace_back(f);
	}

	for (size_t i = 0; i < nThreads; i++)
	{
		threads[i].join();
	}

	//std::vector<double> minRSSk;
	//std::vector<double> AICminRSSk;
	/*for (Eigen::Index k = 400; k < maxK; k += 1)
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
		testedKs << k << "," << minRSS << "," << AICfactor << std::endl;
		AICminRSSk.emplace_back(AICfactor);
		if (AICfactor < bestAICfactor)
		{
			bestAICfactor = AICfactor;
			bestK = k;
			bestClustering = localBestClustering;
		}
	}*/

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

