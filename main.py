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

from b4_objs import node_data, Iris, to_iris
from setup import gen_eq_matrix, is_valid_eq


def build_rand_dataset(M, vec_size, t, show_hist = False):
    dataset = []
    for i in range(M):
        feature = [random.getrandbits(1) for i in range(vec_size)]
        dataset.append(feature)
    return dataset

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
    plt.hist(y_axis, 5)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    print(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset to test.", type=str, default='rand')
    parser.add_argument('--dataset_size', help="Size of dataset to test.", type=int, default=1000)
    parser.add_argument('--lsh_size', help="LSH output size.", type=int, default=18)
    parser.add_argument('--internal_bf_fp', help="LSH output size.", type=float, default=.1)
    parser.add_argument('--root_bf_fp', help="LSH output size.", type=float, default=.0001)
    parser.add_argument('--nb_eLSHes', help="Number of eLSHes.", type=int, default=1259)
    parser.add_argument('--show_histogram', help="Show histogram for tested dataset.", type=int, default=0)
    parser.add_argument('--same_t', help="Avg distance between vectors from same class.", type=float, default=0.3)
    parser.add_argument('--diff_t', help="Avg distance between vectors from different class.", type=float, default=0.4)
    parser.add_argument('--nb_queries', help="Number of queries.", type=int, default=356)
    parser.add_argument('--nb_matches_needed', help="Number of needed matches.", type=int, default=21)
    args = parser.parse_args()

    M = args.dataset_size  # dataset size
    n = args.nb_eLSHes  # number of eLSHes to calculate
    k = args.nb_matches_needed  # number of needed matches
    vec_size = 1024  # vector size
    t = args.same_t
    q = args.nb_queries
    r = 0.85#math.floor(t * n)
    c = 50/85#args.diff_t * (n / r)
    s = args.lsh_size


    branching_factor = 2
    root_bf_fpr = args.root_bf_fp
    internal_bf_fpr = args.internal_bf_fp
    lsh_size = args.lsh_size  # LSH output size


    # build & search using random dataset
    if args.dataset == "rand" or args.dataset == "all":

        t_start = time.time()
        random_data = build_rand_dataset(M, vec_size, t)
        t_end = time.time()
        t_dataset = t_end - t_start
        print("Successfully generated random data in "+str(t_dataset)+" seconds")
        # print("random data : " + str(random_data))
        # print("******************")
        # print("random data size : ", len(random_data))

        data = to_iris(random_data)

        success = 1
        counter = 0
        while success != 0:
            t_start = time.time()
            eLSH = eLSH_import.eLSH(LSH, vec_size, r, c, s, n)
            lsh = eLSH.hashes
            lsh_list = compute_eLSH(data)
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

        x_axis = [i+1 for i in range(10)]
        y_axis = [12, 12, 11, 12, 13, 10, 13, 12, 13, 12]
        show_plot(x_axis, y_axis,  "Frequency", "Max. number of bad matches", "")










