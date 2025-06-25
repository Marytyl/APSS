import sys
import scipy
import argparse
import math
import csv
import random
import time
import matplotlib.pyplot as plt
import os, glob, numpy
import json
from concurrent.futures import ProcessPoolExecutor


import eLSH as eLSH_import
from LSH import LSH

from os import path
import sys
sys.path.append(path.abspath('./UniReedSolomonm'))
from UniReedSolomonm import rs

from b4_objs import node_data, Iris, to_iris
# from OMapE.b4_objs import node_data

from setup import gen_eq_matrix, is_valid_eq, sample_codes, gen_dict, gen_eq_matrix_parallel
from search import search_query_dict


def read_fvector(filePath):
    with open(filePath) as f:
        for line in f.readlines():
            temp_str = numpy.fromstring(line, sep=",")
            return [int(x) for x in temp_str]

# Cache for population and weights
_cached_population_weights = None


def sample_errors(vector_size, error_file):
    global _cached_population_weights

    error_dist = None
    with open(error_file, 'r') as fp:
        error_dist = json.load(fp)
    
    if _cached_population_weights is None:
        population = []
        weights = []
        for errors in error_dist.keys():
            population.append(int(errors))
            weights.append(error_dist[errors])
            _cached_population_weights = (population, weights)
    else:
        population, weights = _cached_population_weights

    # Sample errors
    nb_errors = random.choices(population, weights)[0]
    nb_errors = random.choices(population, weights)[0]
    error_fraction = round(nb_errors/vector_size,3)
    
    return nb_errors, error_fraction

def build_rand_dataset(M, vec_size, t, show_hist = False, error_file = None):
    dataset = []
    queries = []
    queries_error_fraction = []
    queries_error_nb = []
    errors_table = []
    for i in range(M):
        feature = [random.getrandbits(1) for i in range(vec_size)]
        dataset.append(feature)

    # for i in range(M):
        query = dataset[i][:]  # need to be careful to copy value ! (keep the [:] !!)

        # sample errors from distribution
        nb_errors, fraction = sample_errors(vec_size, error_file)
        queries_error_fraction.append(fraction)
        queries_error_nb.append(nb_errors)
        # print("Errors from normal distribution : " + str(nb_errors))
        errors_table.append(nb_errors)

        # randomly sample error bits to be inverted
        error_bits = random.sample(range(vec_size), nb_errors)
        for b in error_bits:
            query[b] = (query[b] + 1) % 2

        queries.append(query)

    if show_hist == 1:
        for j in range(0, len(dataset)):
            queries[j] = []
            for i in range(0, 100):
                nb_errors, fraction = sample_errors(n)
                temp_query = dataset[j].copy()
                error_bits = random.sample(range(n), nb_errors)
                for b in error_bits:
                    temp_query[b] = (temp_query[b] + 1) % 2
                queries[j].append(temp_query)
        build_show_histogram(dataset, queries)
    # print(errors_table)
    # plt.plot(errors_table)
    return dataset, queries, queries_error_fraction, queries_error_nb

# def put_elements_map(element, output):  # puts elements in hash_to_iris
#     for index, h in enumerate(output):
#         h.sort(key = lambda x: x[0])
#         if str(h) in hash_to_iris:
#             hash_to_iris[str(h)] += [element]
#         else:
#             hash_to_iris[str(h)] = [element]


def build_ND_dataset(show_hist = False):
    cwd = os.getcwd()
    dir_list = glob.glob(cwd + "//datasets//1024_folders//*")
    nd_dataset = {}
    class_labels = {}
    i = 0
    for dir in dir_list:
        feat_list = glob.glob(dir + "//*")
        nd_dataset[i] = [read_fvector(x) for x in feat_list]
        class_labels[i] = dir
        i = i + 1


    # nd_templates = ['0' for i in range(len(nd_dataset))]
    # nd_queries = ['0' for i in range(len(nd_dataset))]
    # for x in range(len(nd_dataset)):
    #     print(x)
    #     nd_templates[x] = nd_dataset[x][0]
    #     nd_queries[x] = nd_dataset[x][1]
    nd_templates = [nd_dataset[x][0] for x in range(len(nd_dataset))]
    nd_queries = [nd_dataset[x][1] for x in range(len(nd_dataset))]


    if show_hist == 1:
        nd_queries = [nd_dataset[x][1:] for x in nd_dataset]
        build_show_histogram(nd_templates, nd_queries)
    return nd_templates, nd_queries

def build_synthetic_dataset(l, n, t, show_hist= False, error_file = None):
    dataset = []
    queries = []
    labels = []
    errors_table = []
    ctr = 0
    queries_error_fraction = []
    queries_error_nb = []

    cwd = os.getcwd()
    file_list = glob.glob(cwd + "/datasets/synthetic_dataset/*")
    print("file size", len(file_list))

    for x in file_list:
        dataset.append(read_fvector(x))
        labels.append(x[len(x) - 9:])


        # create query with 30% errors
        query = read_fvector(x)

        nb_errors, fraction = sample_errors(vec_size, error_file)
        queries_error_fraction.append(fraction)
        queries_error_nb.append(nb_errors)
        # print("Errors from normal distribution : " + str(nb_errors))
        errors_table.append(nb_errors)

        # randomly sample error bits to be inverted
        error_bits = random.sample(range(vec_size), nb_errors)
        for b in error_bits:
            query[b] = (query[b] + 1) % 2

        queries.append(query)

        ctr = ctr + 1
        if ctr == l:
            break

    if show_hist == 1:
        for j in range(0, len(dataset)):
            queries[j] = []
            for i in range(0, 100):
                nb_errors, fraction = sample_errors(n)
                temp_query = dataset[j].copy()
                error_bits = random.sample(range(n), nb_errors)
                for b in error_bits:
                    temp_query[b] = (temp_query[b] + 1) % 2
                queries[j].append(temp_query)
        build_show_histogram(dataset, queries)

    return dataset, queries, queries_error_fraction, queries_error_nb



# compute eLSH and returns the list of length l
def compute_eLSH_one(eLSH, element):
    output = eLSH.hash(element.vector)  # length of l
    #put_elements_map(element, output)
    return output

# computes eLSH output of multiple elements F
def compute_eLSH_one_wrapper(args):
    eLSH, element = args
    return compute_eLSH_one(eLSH, element)

def compute_eLSH(eLSH, elements):
    with ProcessPoolExecutor() as executor:
        output = list(executor.map(compute_eLSH_one_wrapper, [(eLSH, element) for element in elements]))
    return output

def hamming_dist(sample1, sample2):
    dist = 0
    if len(sample1) != len(sample2):
        raise ValueError

    for i in range(0, len(sample1)):
        if sample1[i] != sample2[i]:
            dist+=1/len(sample1)
    return dist

def build_show_histogram(data, queries):
    redDist=[]
    blueDist=[]
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            if i != j:
                redDist.append(hamming_dist(data[i], data[j]))
        if len(queries) > i and len(queries[i]) > 0:
            if type(queries[i][0]) is int:
                diff_query = queries[i]
                blueDist.append(hamming_dist(data[i], diff_query))
            else:
                for diff_query in queries[i]:
                    blueDist.append(hamming_dist(data[i], diff_query))
        #     print(type(diff_queries))

    blueW = [1/len(blueDist) for i in range(0, len(blueDist))]
    redW = [1/len(redDist) for i in range(0, len(redDist))]
    if len(blueDist) > 0:
        plt.hist(blueDist, density=True, bins=41, histtype='stepfilled', weights= blueW,
                 color='b', alpha=0.7, label='Comparisons readings same iris')

    if len(redDist) > 0:
        plt.hist(redDist, density=True, bins=41, histtype='stepfilled', weights = redW,
                 color='r', label='Comparisons different irises')

    if len(redDist) >0 or len(blueDist) > 0:
        plt.legend()
        plt.xlabel("Hamming Distance")
        plt.ylabel("Frequency")
        plt.title("Random Data")
        plt.show()

#    plt.hist(redComparisons, normed=True, bins=120, histtype='stepfilled', color='r', label='Different')
# plt.show()


def show_plot(x_axis, y_axis, x_label, y_label, title):
    plt.hist(y_axis, max(y_axis)-min(y_axis)+1)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title("")
    plt.show()
    # plt.savefig(title)


def build_binom_error_dist(vector_size):
    n=23
    p=.1
    binom_pmf = {}
    for k in range(n+1):
        binom_pmf[round(vector_size*k/n)] = scipy.stats.binom.pmf(k, n, p)
    print(binom_pmf)
    import json

    with open('error_dist.json', 'w') as fp:
        json.dump(binom_pmf, fp)
    exit(1)

def process_single_query(args):
    j, query, n, k, dict = args
    t_start = time.time()
    res, erasure, errors, p_time = search_query_dict(query, n, k, dict)
    t_end = time.time()
    correct = n - erasure - errors
    required_search_items = 3 * errors if 2 * correct >= errors else 0
    true_accept = 1 if res == j else 0
    return t_end - t_start, p_time, required_search_items, true_accept

def process_queries(queries_lsh_list, q, n, k):

    query_time = []
    parallel_time = []
    required_search_items = 0
    true_accept_rate = 0

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_single_query, [(j, queries_lsh_list[j], n, k, dict) for j in range(q)])

    for t_time, p_time, search_items, accept in results:
        query_time.append(t_time)
        parallel_time.append(p_time)
        required_search_items = max(required_search_items, search_items)
        true_accept_rate += accept

    return query_time, parallel_time, required_search_items, true_accept_rate

def process_dataset(dataset_type, M, vec_size, t, show_hist, error_file, map_type, q, k, s, n, LSH, r, c):
    if dataset_type == "rand":
        random_data, queries, queries_error_fraction, queries_error_nb = build_rand_dataset(M, vec_size, t, show_hist, error_file)
    elif dataset_type == "synth":
        random_data, queries, queries_error_fraction, queries_error_nb = build_synthetic_dataset(M, vec_size, t, show_hist, error_file)
    elif dataset_type == "nd":
        random_data, queries = build_ND_dataset(show_hist)
        queries_error_fraction, queries_error_nb = None, None
    else:
        raise ValueError("Invalid dataset type")

    print(f"Successfully generated {dataset_type} data")
    data = to_iris(random_data)
    queries = to_iris(queries)

    success = 1
    counter = 0

    while success != 0:
        if just_eq_matrix:
            eq, num_match = gen_eq_matrix_parallel(M, n, data, s, vec_size)
        else:
            t_start = time.time()
            eLSH = eLSH_import.eLSH(LSH, vec_size, r, c, s, n)
            lsh = eLSH.hashes
            lsh_list = compute_eLSH(eLSH, data)
            queries_lsh_list = compute_eLSH(eLSH, queries)
            t_end = time.time()
            t_lsh = t_end - t_start
            print("Successfully generated LSH evaluations in "+str(t_lsh)+" seconds", flush=True)
            t_start = time.time()
            eq, num_match = gen_eq_matrix(len(lsh_list), len(lsh_list[0]), lsh_list)
            t_end = time.time()
            t_eq = t_end - t_start
            print("Successfully generated equality matrix in " + str(t_eq) + " seconds")

        t_start = time.time()
        success = is_valid_eq(eq, k)
        t_success = time.time() - t_start
        print(f"Success is {success}, checked in {t_success:.2f} seconds")
        print(f"Number of matches per LSH: {round(numpy.average(num_match))}")
        counter += 1
        print(f"Iteration: {counter}")

    codes = sample_codes(n, k, M, eq)
    global dict
    t_start = time.time()
    dict = gen_dict(codes, M, n, lsh_list, map_type)
    t_dict = time.time() - t_start
    print(f"Successfully generated {map_type} in {t_dict:.2f} seconds")

    query_time = []
    parallel_time = []
    required_search_items = 0
    true_accept_rate = 0

    q = min(q, len(queries_lsh_list))
    (query_time, parallel_time, required_search_items, true_accept_rate) = process_queries(queries_lsh_list, q, n, k)

    print(f"Avg Query Time: {numpy.average(query_time):.2f}, STDev: {numpy.std(query_time):.2f}")
    print(f"Avg Parallel Time: {numpy.average(parallel_time):.6f}, STDev: {numpy.std(parallel_time):.6f}")
    print(f"True Accept Rate: {true_accept_rate / q:.2f}")
    print(f"Number of search items needed: {required_search_items}")


if __name__ == '__main__':
    print(sys.version)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset to test.", type=str, default='rand')
    parser.add_argument('--dataset_size', help="Size of dataset to test.", type=int, default=1000)
    parser.add_argument('--lsh_size', help="LSH output size.", type=int, default=20)
    parser.add_argument('--internal_bf_fp', help="LSH output size.", type=float, default=.1)
    parser.add_argument('--root_bf_fp', help="LSH output size.", type=float, default=.0001)
    parser.add_argument('--nb_eLSHes', help="Number of eLSHes.", type=int, default=25)
    parser.add_argument('--show_histogram', help="Show histogram for tested dataset.", type=int, default=0)
    parser.add_argument('--same_t', help="Avg distance between vectors from same class.", type=float, default=0.3)
    parser.add_argument('--diff_t', help="Avg distance between vectors from different class.", type=float, default=0.4)
    parser.add_argument('--nb_queries', help="Number of queries.", type=int, default=208)
    parser.add_argument('--nb_matches_needed', help="Number of needed matches.", type=int, default=20)
    parser.add_argument('--eps_t', help="TPR of each eLSH", type=int, default=85)
    parser.add_argument('--eps_f', help="FPR of each eLSH", type=int, default=50)
    parser.add_argument('--error_rate_percent', help="mean error rate", type=float, default=15)
    parser.add_argument('--map', help="Map to use.", type=str, default='omap')
    parser.add_argument('--error_dist_file', help="Distribution to use for errors", type=str, default='error_dist.json')
    args = parser.parse_args()
    show_hist = bool(args.show_histogram)
    M = args.dataset_size  # dataset size
    n = args.nb_eLSHes  # number of eLSHes to calculate
    k = args.nb_matches_needed  # number of needed matches
    just_eq_matrix = 0
    vec_size = 1024  # vector size
    t = args.same_t
    q = args.nb_queries
    map_type = args.map
    error_file = args.error_dist_file

    r = args.eps_t/100#0.85#math.floor(t * n)
    c = args.eps_f/args.eps_t#50/85#args.diff_t * (n / r)
    s = args.lsh_size

    # error_rate = args.error_rate_percent/100

    print("alpha = ", s, " n = ", n, " k = ", k)

    # Process dataset
    if args.dataset in ["rand", "synth", "nd", "all"]:
        process_dataset(args.dataset, M, vec_size, t, show_hist, error_file, map_type, q, k, s, n, LSH, r, c)







