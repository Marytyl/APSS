import sys
import scipy
import argparse
import math
import csv
import random
import time
import matplotlib.pyplot as plt
import os, glob, numpy


import eLSH as eLSH_import
from LSH import LSH

from os import path
import sys
sys.path.append(path.abspath('./UniReedSolomonm'))
from UniReedSolomonm import rs


from b4_objs import node_data, Iris, to_iris
from setup import gen_eq_matrix, is_valid_eq, sample_codes, gen_dict, gen_eq_matrix_parallel
from search import search_query_dict


def read_fvector(filePath):
    with open(filePath) as f:
        for line in f.readlines():
            temp_str = numpy.fromstring(line, sep=",")
            return [int(x) for x in temp_str]

def sample_errors(vector_size, error_rate=0.1):

    mean_same = error_rate
    stdev_same = 0.056

    # compute n using degrees of freedom formula
    n = (mean_same * (1 - mean_same)) / (stdev_same ** 2)
    p = mean_same
    # print("p = " + str(p) + " and n = " + str(math.ceil(n)))

    error_fraction = scipy.stats.binom.rvs(math.ceil(n), p) / n
    # print(error_fraction)
    nb_errors = round(vector_size * error_fraction)
    return nb_errors, round(error_fraction, 3)

def build_rand_dataset(M, vec_size, t, show_hist = False, error_rate=0.1):
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
        nb_errors, fraction = sample_errors(vec_size, error_rate)
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


def build_synthetic_dataset(l, n, t, show_hist= False):
    dataset = []
    queries = []
    labels = []
    errors_table = []
    ctr = 0
    queries_error_fraction = []
    queries_error_nb = []

    cwd = os.getcwd()
    file_list = glob.glob(cwd + "//datasets//synthetic_dataset//*")
    print("file size", len(file_list))

    for x in file_list:
        dataset.append(read_fvector(x))
        labels.append(x[len(x) - 9:])

        for i in range(100):
            # create query with 30% errors
            query = read_fvector(x)

            # sample errors from distribution
            nb_errors, fraction = sample_errors(n)
            # print("Errors from normal distribution : " + str(nb_errors))
            queries_error_fraction.append(fraction)
            queries_error_nb.append(nb_errors)

            errors_table.append(nb_errors)
            # randomly sample error bits to be inverted
            error_bits = random.sample(range(n), nb_errors)
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

if __name__ == '__main__':

    print(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset to test.", type=str, default='nd')
    parser.add_argument('--dataset_size', help="Size of dataset to test.", type=int, default=208)
    parser.add_argument('--lsh_size', help="LSH output size.", type=int, default=20)
    parser.add_argument('--internal_bf_fp', help="LSH output size.", type=float, default=.1)
    parser.add_argument('--root_bf_fp', help="LSH output size.", type=float, default=.0001)
    parser.add_argument('--nb_eLSHes', help="Number of eLSHes.", type=int, default=500)
    parser.add_argument('--show_histogram', help="Show histogram for tested dataset.", type=int, default=0)
    parser.add_argument('--same_t', help="Avg distance between vectors from same class.", type=float, default=0.3)
    parser.add_argument('--diff_t', help="Avg distance between vectors from different class.", type=float, default=0.4)
    parser.add_argument('--nb_queries', help="Number of queries.", type=int, default=208)
    parser.add_argument('--nb_matches_needed', help="Number of needed matches.", type=int, default=30)
    parser.add_argument('--eps_t', help="TPR of each eLSH", type=int, default=85)
    parser.add_argument('--eps_f', help="FPR of each eLSH", type=int, default=50)
    parser.add_argument('--error_rate_percent', help="mean error rate", type=float, default=15)
    args = parser.parse_args()

    show_hist = bool(args.show_histogram)
    M = args.dataset_size  # dataset size
    n = args.nb_eLSHes  # number of eLSHes to calculate
    k = args.nb_matches_needed  # number of needed matches
    just_eq_matrix = 0
    vec_size = 1024  # vector size
    t = args.same_t
    q = args.nb_queries

    r = args.eps_t/100#0.85#math.floor(t * n)
    c = args.eps_f/args.eps_t#50/85#args.diff_t * (n / r)
    s = args.lsh_size

    error_rate = args.error_rate_percent/100

    print("alpha = ", s, " n = ", n, " k = ", k)
    branching_factor = 2
    root_bf_fpr = args.root_bf_fp
    internal_bf_fpr = args.internal_bf_fp
    lsh_size = args.lsh_size  # LSH output size
    # max_nonzero_count = [0 for i in range(100)]
    # max_nonzero_rows = [0 for i in range(100)]
    # failed_rows = [0 for i in range(100)]

    query = None
    # build & search using random dataset
    if args.dataset == "rand" or args.dataset == "all":

        t_start = time.time()
        random_data, queries, queries_error_fraction, queries_error_nb = build_rand_dataset(M, vec_size, t, show_hist, error_rate)
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
                eq, num_match = gen_eq_matrix_parallel(M, n, data, s, vec_size)
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
                eq, num_match = gen_eq_matrix(len(lsh_list), len(lsh_list[0]), lsh_list)
                t_end = time.time()
                t_eq = t_end - t_start
                print("Successfully generated equality matrix in " + str(t_eq) + " seconds")
            #print("eq: ", eq)

            t_start = time.time()
            success = is_valid_eq(eq, k)
            t_end = time.time()
            t_success = t_end - t_start
            print("Success is "+str(success)+", checked in "+str(t_success)+" seconds")
            print("Number of matches per LSH: ", num_match)
            # plt.hist(num_match, n)
            # plt.show()
            counter += 1
            print("iteration: "+str(counter))


        t_start = time.time()
        codes = sample_codes(n, k, M, eq)
        t_end = time.time()
        t_code_sampling = t_end - t_start
        print("Successfully sampled codes in " + str(t_code_sampling) + " seconds")
        # print(codes)

        t_start = time.time()
        dict = gen_dict(codes, M, n, lsh_list)
        t_end = time.time()
        t_dict = t_end - t_start
        print("Successfully generated dictionary in " + str(t_dict) + " seconds")

        #l_query = compute_eLSH(query)
        #l_query = lsh_list[1]

        for j in range(len(queries_lsh_list)):
            print(j, search_query_dict(queries_lsh_list[j], lsh_list, k, dict), queries_error_nb[j], queries_error_fraction[j])



    # x_axis = [i+1 for i in range(100)]
    # y_axis = max_nonzero_count
    # show_plot(x_axis, y_axis,  "Frequency", "Max. number of eLSH matches",
    #           "Histogram of Maximum number of matches, M=10^3")
    # print("******")
    # print(max_nonzero_count)
    # print("******")
    # print(max_nonzero_rows)
    # print("******")
    # print(failed_rows)
    # print("******")
    # x_axis = [i + 1 for i in range(100)]
    # y_axis = max_nonzero_count
    # show_plot(x_axis, y_axis, "Frequency", "Max. number of eLSH matches",
    #           "Histogram of Maximum number of matches, M=10^4, c_1=5")


    if args.dataset == "synth" or args.dataset == "all":

        for i in range(10):
            print("run: ", i)
            t_start = time.time()
            random_data, queries, queries_error_fraction, queries_error_nb = build_synthetic_dataset(M, vec_size, t, show_hist)
            t_end = time.time()
            t_dataset = t_end - t_start
            print("Successfully generated synthetic data in "+str(t_dataset)+" seconds")
            # print("random data : " + str(random_data))
            # print("******************")
            # print("random data size : ", len(random_data))
            # print(len(random_data), len(queries))
            data = to_iris(random_data)
            queries = to_iris(queries)
            # query = to_iris(random_data[0])

            success = 1
            counter = 0


            while success != 0:
                if just_eq_matrix:
                    t_start = time.time()
                    eq, num_match = gen_eq_matrix_parallel(M, n, data, s, vec_size)
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
                    # print(len(lsh_list[0]), len(lsh_list))
                    t_start = time.time()
                    eq, num_match = gen_eq_matrix(len(lsh_list), len(lsh_list[0]), lsh_list)
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


        t_start = time.time()
        codes = sample_codes(n, k, M, eq)
        t_end = time.time()
        t_code_sampling = t_end - t_start
        print("Successfully sampled codes in " + str(t_code_sampling) + " seconds")
        # print(codes)

        t_start = time.time()
        dict = gen_dict(codes, M, n, lsh_list)
        t_end = time.time()
        t_dict = t_end - t_start
        print("Successfully generated dictionary in " + str(t_dict) + " seconds")

        #l_query = compute_eLSH(query)
        #l_query = lsh_list[1]

        for j in range(len(queries_lsh_list)):
            print(j, search_query_dict(queries_lsh_list[j], lsh_list, k, dict), queries_error_nb[j], queries_error_fraction[j])



    # x_axis = [i+1 for i in range(100)]
    # y_axis = max_nonzero_count
    # show_plot(x_axis, y_axis,  "Frequency", "Max. number of eLSH matches",
    #           "Histogram of Maximum number of matches, M=10^3")
    # print("******")
    # print(max_nonzero_count)
    # print("******")
    # print(max_nonzero_rows)
    # print("******")
    # print(failed_rows)
    # print("******")
    # x_axis = [i + 1 for i in range(100)]
    # y_axis = max_nonzero_count
    # show_plot(x_axis, y_axis, "Frequency", "Max. number of eLSH matches",
    #           "Histogram of Maximum number of matches, M=10^4, c_1=5")

    if args.dataset == "nd" or args.dataset == "all":
        t_start = time.time()
        random_data, queries = build_ND_dataset(show_hist)
        t_end = time.time()
        t_dataset = t_end - t_start
        print("Successfully generated real data in "+str(t_dataset)+" seconds")
        # print("random data : " + str(random_data))
        # print("******************")
        # print("random data size : ", len(random_data))
        # print(len(random_data), len(queries))
        data = to_iris(random_data)
        queries = to_iris(queries)
        # query = to_iris(random_data[0])

        success = 1
        counter = 0


        while success != 0:
            if just_eq_matrix:
                t_start = time.time()
                eq, num_match = gen_eq_matrix_parallel(M, n, data, s, vec_size)
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
                # print(lsh_list[0][0])
                # print(lsh_list[1][0])

                queries_lsh_list = compute_eLSH(queries)
                t_end = time.time()
                t_lsh = t_end - t_start
                print("Successfully generated LSH evaluations in "+str(t_lsh)+" seconds")
                # print("elsh", lsh_list)
                # print("******************")
                # print(len(lsh_list[0]), len(lsh_list))
                t_start = time.time()
                eq, num_match = gen_eq_matrix(len(lsh_list), len(lsh_list[0]), lsh_list)
                t_end = time.time()
                t_eq = t_end - t_start
                print("Successfully generated equality matrix in " + str(t_eq) + " seconds")
            # print("eq: ", eq)

            t_start = time.time()
            success = is_valid_eq(eq, k)
            t_end = time.time()
            t_success = t_end - t_start
            print("Success is "+str(success)+", checked in "+str(t_success)+" seconds")
            print("Number of matches per LSH: ", num_match)
            # plt.hist(num_match, n)
            # plt.show()
            counter += 1
            print("iteration: "+str(counter))


        t_start = time.time()
        codes = sample_codes(n, k, M, eq)
        t_end = time.time()
        t_code_sampling = t_end - t_start
        print("Successfully sampled codes in " + str(t_code_sampling) + " seconds")
        # print(codes)

        t_start = time.time()
        dict = gen_dict(codes, M, n, lsh_list)
        t_end = time.time()
        t_dict = t_end - t_start
        print("Successfully generated dictionary in " + str(t_dict) + " seconds")

        #l_query = compute_eLSH(query)
        #l_query = lsh_list[1]
        for j in range(len(queries_lsh_list)):
            print(j, search_query_dict(queries_lsh_list[j], lsh_list, k, dict))

    # x_axis = [i+1 for i in range(100)]
    # y_axis = max_nonzero_count
    # show_plot(x_axis, y_axis,  "Frequency", "Max. number of eLSH matches",
    #           "Histogram of Maximum number of matches, M=10^3")
    # print("******")
    # print(max_nonzero_count)
    # print("******")
    # print(max_nonzero_rows)
    # print("******")
    # print(failed_rows)
    # print("******")
    # x_axis = [i + 1 for i in range(100)]
    # y_axis = max_nonzero_count
    # show_plot(x_axis, y_axis, "Frequency", "Max. number of eLSH matches",
    #           "Histogram of Maximum number of matches, M=10^4, c_1=5")








