import sys
import scipy
import argparse
import math
import csv
import random
import time
import matplotlib.pyplot as plt

import eLSH as eLSH_import
from LSH import LSH

from os import path
import sys
sys.path.append(path.abspath('./UniReedSolomonm'))
from UniReedSolomonm import rs


from b4_objs import node_data, Iris, to_iris
from setup import gen_eq_matrix, is_valid_eq, sample_codes, gen_dict, gen_eq_matrix_parallel
from search import search_query_dict

def sample_errors(vector_size):
    mean_same = 0.21
    stdev_same = 0.056

    # compute n using degrees of freedom formula
    n = (mean_same * (1 - mean_same)) / (stdev_same ** 2)
    p = mean_same
    # print("p = " + str(p) + " and n = " + str(math.ceil(n)))

    error_fraction = scipy.stats.binom.rvs(math.ceil(n), p) / n
    # print(error_fraction)
    nb_errors = round(vector_size * error_fraction)
    return nb_errors, round(error_fraction, 3)

def build_rand_dataset(M, vec_size, t, show_hist = False):
    dataset = []
    queries = []
    queries_error_fraction = []
    queries_error_nb = []
    errors_table = []
    for i in range(M):
        feature = [random.getrandbits(1) for i in range(vec_size)]
        dataset.append(feature)

    for i in range(50):
        query = dataset[i][:]  # need to be careful to copy value ! (keep the [:] !!)

        # sample errors from distribution
        nb_errors, fraction = sample_errors(vec_size)
        queries_error_fraction.append(fraction)
        queries_error_nb.append(nb_errors)
        # print("Errors from normal distribution : " + str(nb_errors))
        errors_table.append(nb_errors)

        # randomly sample error bits to be inverted
        error_bits = random.sample(range(vec_size), nb_errors)
        for b in error_bits:
            query[b] = (query[b] + 1) % 2

        queries.append(query)

        # if show_hist == 1:
        #     for j in range(0, len(dataset)):
        #         queries[j] = []
        #         for i in range(0, 100):
        #             nb_errors, fraction = sample_errors(n)
        #             temp_query = dataset[j].copy()
        #             error_bits = random.sample(range(n), nb_errors)
        #             for b in error_bits:
        #                 temp_query[b] = (temp_query[b] + 1) % 2
        #             queries[j].append(temp_query)
        #     build_show_histogram(dataset, queries)
        # # print(errors_table)
        # plt.plot(errors_table)
    return dataset, queries, queries_error_fraction, queries_error_nb

# def put_elements_map(element, output):  # puts elements in hash_to_iris
#     for index, h in enumerate(output):
#         h.sort(key = lambda x: x[0])
#         if str(h) in hash_to_iris:
#             hash_to_iris[str(h)] += [element]
#         else:
#             hash_to_iris[str(h)] = [element]


# compute eLSH and returns the list of length l
def compute_eLSH_one(element):
    output = eLSH.hash(element.vector)  # length of l
    #put_elements_map(element, output)
    return output

# computes eLSH output of multiple elements F
def compute_eLSH(elements):
    output = []
    for i in elements:
        output.append(compute_eLSH_one(i))
    return output


def show_plot(x_axis, y_axis, x_label, y_label, title):
    plt.hist(y_axis, max(y_axis)-min(y_axis)+1)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title("")
    plt.show()
    plt.savefig(title)

if __name__ == '__main__':
    # coder = rs.RSCoder(631, 32)
    # c = coder.encode("5")
    # r = "\0" + "\uf562"*299 + c[300:630] + "\0"
    # # print(r)
    # # coder = rs.RSCoder(20, 13)
    # d = coder.decode(r)
    # print(d)

    print(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset to test.", type=str, default='rand')
    parser.add_argument('--dataset_size', help="Size of dataset to test.", type=int, default=1000)
    parser.add_argument('--lsh_size', help="LSH output size.", type=int, default=17)
    parser.add_argument('--internal_bf_fp', help="LSH output size.", type=float, default=.1)
    parser.add_argument('--root_bf_fp', help="LSH output size.", type=float, default=.0001)
    parser.add_argument('--nb_eLSHes', help="Number of eLSHes.", type=int, default=631)
    parser.add_argument('--show_histogram', help="Show histogram for tested dataset.", type=int, default=0)
    parser.add_argument('--same_t', help="Avg distance between vectors from same class.", type=float, default=0.3)
    parser.add_argument('--diff_t', help="Avg distance between vectors from different class.", type=float, default=0.4)
    parser.add_argument('--nb_queries', help="Number of queries.", type=int, default=356)
    parser.add_argument('--nb_matches_needed', help="Number of needed matches.", type=int, default=32)
    args = parser.parse_args()

    M = args.dataset_size  # dataset size
    n = args.nb_eLSHes  # number of eLSHes to calculate
    k = args.nb_matches_needed  # number of needed matches
    just_eq_matrix = 0
    vec_size = 1024  # vector size
    t = args.same_t
    q = args.nb_queries
    r = 0.9#math.floor(t * n)
    c = 50/90#args.diff_t * (n / r)
    s = args.lsh_size


    branching_factor = 2
    root_bf_fpr = args.root_bf_fp
    internal_bf_fpr = args.internal_bf_fp
    lsh_size = args.lsh_size  # LSH output size

    query = None
    # build & search using random dataset
    if args.dataset == "rand" or args.dataset == "all":

        t_start = time.time()
        random_data, queries, queries_error_fraction, queries_error_nb = build_rand_dataset(M, vec_size, t)
        t_end = time.time()
        t_dataset = t_end - t_start
        print("Successfully generated random data in "+str(t_dataset)+" seconds")
        # print("random data : " + str(random_data))
        # print("******************")
        # print("random data size : ", len(random_data))

        data = to_iris(random_data)
        queries = to_iris(queries)
        # query = to_iris(random_data[0])

        success = 1
        counter = 0

        while success != 0:
            if just_eq_matrix:
                t_start = time.time()
                eq = gen_eq_matrix_parallel(M, n, data, s, vec_size)
                t_end = time.time()
                t_eq = t_end - t_start
                print("Successfully generated equality matrix in " + str(t_eq) + " seconds")
                # print("eq: ", eq)

                t_start = time.time()
            else:
                t_start = time.time()
                eLSH = eLSH_import.eLSH(LSH, vec_size, r, c, s, n)
                lsh = eLSH.hashes
                lsh_list = compute_eLSH(data)
                queries_lsh_list = compute_eLSH(queries)
                t_end = time.time()
                t_lsh = t_end - t_start
                print("Successfully generated LSH evaluations in "+str(t_lsh)+" seconds")
                #print("elsh", lsh_list)
                # print("******************")
                #print(len(lsh_list[0]), len(lsh_list))
                t_start = time.time()
                eq = gen_eq_matrix(len(lsh_list), len(lsh_list[0]), lsh_list)
                t_end = time.time()
                t_eq = t_end - t_start
                print("Successfully generated equality matrix in " + str(t_eq) + " seconds")
            #print("eq: ", eq)

            t_start = time.time()
            success = is_valid_eq(eq, k)
            t_end = time.time()
            t_success = t_end - t_start
            print("Success is "+str(success)+", checked in "+str(t_success)+" seconds")
            counter += 1
            print("iteration: "+str(counter))

        if just_eq_matrix:
            exit(1)
        t_start = time.time()
        codes = sample_codes(n, k, M, eq)
        t_end = time.time()
        t_code_sampling = t_end - t_start
        print("Successfully sampled codes in " + str(t_code_sampling) + " seconds")
        # print(codes)
        dict = gen_dict(codes, M, n, lsh_list)

        #l_query = compute_eLSH(query)
        #l_query = lsh_list[1]

        for j in range(len(queries_lsh_list)):
            print(j, search_query_dict(queries_lsh_list[j], lsh_list, k, dict), queries_error_nb[j], queries_error_fraction[j])



        # x_axis = [i+1 for i in range(10)]
        # y_axis = [12, 12, 11, 12, 13, 10, 13, 12, 13, 12, 13, 11, 13, 11, 13, 14, 11, 11, 13, 10, 15, 12, 13, 13, 14]
        # show_plot(x_axis, y_axis,  "Frequency", "Max. number of eLSH matches",
        #           "Histogram of Maximum number of matches, M=10^3")
        #
        # x_axis = [i + 1 for i in range(10)]
        # y_axis = [14, 14, 15, 14, 15, 14, 13, 15, 14, 14, 17, 16, 16, 15, 14, 14, 15, 17, 14, 13]
        # show_plot(x_axis, y_axis, "Frequency", "Max. number of eLSH matches",
        #           "Histogram of Maximum number of matches, M=10^4")










