# K-Means Clustering Engine

Author: Shane Peelar (peelar@uwindsor.ca)

If you choose to use this tool in your project, please cite me in your report.  You can use this BiBTeX entry to conveniently do so:

~~~
@article{peelar, title={K-Means Clustering Engine}, url={https://github.com/InBetweenNames/KMeansClustering}, author={Peelar, Shane M}}
~~~

This code is intended for the 60-538 course Information Retrieval at the University of Windsor.  A Visual Studio 2017 solution is provided for the code.

~~~
Usage: ./KMeansClustering <vectorfile> <outfile> (-K <n>)
~~~

Where:
~~~
<vectorfile>: A file with document vectors, each line having the following form: <id> <class> <vectors>
<outfile>: The resultant clustering will be written to this file
~~~

See my [vectorize.py tool]() to generate these from metadata files easily.

If the `-K` option is provided, then a clustering will be generated using the number of clusters provided as `K`.
If the `-K` option is omitted, then optimization will be performed to determine the best value of `K` for your dataset.
Beware that optimization can take a long time, and will consume all available CPU resources on your computer!
The best purity, minRSS, and AICminRSSk will be written to the file `optimal.csv`.
Often, you only need to evaluate the first 1000 possible K values.


