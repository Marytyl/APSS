import numpy as np
# import unireedsolomon as rs
import map as m
import multiprocessing as mp
import eLSH as eLSH_import
from joblib import Parallel, delayed
from LSH import LSH
import secrets
from functools import partial
from os import path
import sys
sys.path.append(path.abspath('./UniReedSolomonm'))
from UniReedSolomonm import rs
from collections import Counter


#num_cores = mp.cpu_count()
num_cores = 32

def column(matrix, i):
   return [row[i] for row in matrix]

def eq_matrix_one_col(dataset, s, vec_size):
    r = .85
    c = 50 / 85
    eLSH = eLSH_import.eLSH(LSH, vec_size, r, c, s, 1)
    lsh_list = []
    for data_element in dataset:
        lsh_list.append(eLSH.hash(data_element.vector))
    row = len(dataset)
    eq_col = [0 for i in range(row)]
    lsh_dict = {}
    for i in range(1, row):
        if str(lsh_list[i]) in lsh_dict:
            eq_col[i] = lsh_dict[str(lsh_list[i])]
        else:
            lsh_dict[str(lsh_list[i])] = i

    return eq_col


def gen_eq_matrix_parallel(M, n, dataset, s, vec_size):
    eq_col = Parallel(n_jobs=num_cores)(delayed(eq_matrix_one_col)
                                                (dataset, s, vec_size)
                                                for i in range(n))
    eq_matrix = [[0 for i in range(n)] for j in range(M)]
    for j in range(len(eq_col)):
        col = eq_col[j]
        for i in range(len(col)):
            eq_matrix[i][j] = col[i]
    return eq_matrix

def gen_eq_matrix(M, n, lsh_list):
    parallel = 0
    row, col = M, n
    eq_mat = [[0 for i in range(col)] for j in range(row)]
    num_lsh_matches = [1 for i in range(col)]
    arr = []

    lsh_dict = {}
    for j in range(col):
        # if j%100 == 0:
        #     print("Current column is "+str(j))
        for i in range(1, row):
            key = str(j)+", "+str(lsh_list[i][j])
            if key in lsh_dict:
                eq_mat[i][j] = lsh_dict[key]
                num_lsh_matches[j] += 1
            else:
                lsh_dict[key] = i

    return eq_mat, num_lsh_matches

def is_valid_eq(eq_mat, k):
    flag = 0
    max_nonzero_count = []
    nonzero_count_list = []
    nonzero_rows = 0
    failed_rows = 0
    max_nonzero_count.append(0)
    for i in range(len(eq_mat)):
        flag_row = 0
        count_nonzero = 0
        for j in range(len(eq_mat[0])):
            if eq_mat[i][j] != 0:
                count_nonzero += 1

                if(count_nonzero > max_nonzero_count[nonzero_rows]):
                    max_nonzero_count[nonzero_rows] = count_nonzero
            if count_nonzero == k:
                flag = 1
                flag_row = 1
                #break
        if count_nonzero != 0:
            nonzero_rows += 1
            nonzero_count_list.append(count_nonzero)
            max_nonzero_count.append(0)
        if flag_row == 1:
            #max_nonzero_count.append(k)
            failed_rows += 1
            #break
    # print("Max nonzero count per row: " + str(max_nonzero_count))
    print("nonzero count list: ", nonzero_count_list)
    print("Max nonzero count per matrix: " + str(max(max_nonzero_count)))
    print("Number of nonzero rows: " + str(nonzero_rows))
    print("Number of failed rows: " + str(failed_rows))

    nonzero_count_list_col = []
    # for j in range(len(eq_mat[0])):
    #     # test_list = [0, 1, 1, 1, 0, 4, 4]
    #     res = [counter for counter in Counter(column(eq_mat, j)).items()]
        # print("Grouped and counted list is : ", res)
        # print("Grouped and counted list is : ", [x[1] + 1 for x in res])

    #     count_nonzero_col = 0
    #     for i in range(len(eq_mat)):
    #         if eq_mat[i][j] != 0:
    #             count_nonzero_col += 1
    #     if count_nonzero_col != 0:
    #         nonzero_count_list_col.append(count_nonzero_col)
    # print("nonzero count list per col: ", nonzero_count_list_col)

    return flag

def sample_codes(n, k, M, eq):
    coder = rs.RSCoder(n+1, k)
    # rsc = RSCodec(n)
    codes = []
    for i in range(M):

        c = coder.encode_fast(str(i))
        # c = rsc.encode(str(i))

        c = list(c)
        c[0] = chr(i)
        for j in range(len(c)-1):
            index = eq[i][j]
            if index != 0:
                c[j+1] = codes[index][j]
        codes.append(c[0])
        for j in range(1, len(c)):
            if c[j] != None:
                codes[i] += c[j]
    return codes

def gen_dict(codes, M, n, lsh_list):
    dict = {}
    for i in range(M):
        for j in range(n):
            key = str(j) + ", " + str(lsh_list[i][j])
            dict[key] = codes[i][j+1]
    return dict

def gen_map(codes, M, n, lsh_list):
    for i in range(M):
        for j in range(n):
            m.insert(chr(j)+lsh_list[i](j), codes[i](j))






