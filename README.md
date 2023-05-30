# APSS
This code implements Approximate Proximity Search Scheme (APSS) using a Unique Reed Solomon Library with support of erasures.

The implementaion is two-fold; First implementing a setup (setup.py) that creates a map between eLSH evaluations of each keyword and secret-shares of that keyword index. Shares are sampled using Reed-Solomon encoding algorithm. Before creating the map, setup checks if the eLSHes are well-spread by creating and checking their equality matrix.

In the second part we implemented search.py, that retrieves some code shares from the map based on the query LSH evaluation. Then search run Reed-Solomon decoding to retrieve the hidden index.

In the main.py, we have sampled a dataset, and queries and called the above procedures.

To run the code you can execute the following command in the project path:

python3 main.py --dataset_size=10000 --lsh_size=21 --nb_eLSHes=631 --nb_matches_needed=22 --eps_t=90 --eps_f=50 --error_rate_percent=10

You can select a set of working parameters from table 1, 2, or 3 of the paper.

--dataset_size : is the intended dataset size
--lsh_size : is the size of each eLSH, or in other words number of concatenated LSHes. Corresponding to \alpha in the paper.
--nb_eLSHes : is the number of required eLSHes. Corresponding to n in the paper.
--nb_matches_needed : is the required number of matches so that the decoding algorithm works. Corresponding to k in the paper.
--eps_t : is the rate of mathing to close things for each LSH.
--eps_f : is the rate of matching to far thing for each LSH.
--error_rate_percent : is the mean rate of errors, when sampling query from actual data.
