# APSS
This code implements Approximate Proximity Search Scheme (APSS) using a Unique Reed Solomon Library with support of erasures.

The implementaion is two-fold; First implementing a setup (setup.py) that creates a map between eLSH evaluations of each keyword and secret-shares of that keyword index. Shares are sampled using Reed-Solomon encoding algorithm. Before creating the map, setup checks if the eLSHes are well-spread by creating and checking their equality matrix.

In the second part we implemented search.py, that retrieves some code shares from the map based on the query LSH evaluation. Then search run Reed-Solomon decoding to retrieve the hidden index.

In the main.py, we have sampled a dataset, and queries and called the above procedures.
