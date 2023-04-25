import numpy as np
import unireedsolomon as rs
import map as m
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import partial


num_cores = mp.cpu_count()


def eq_matrix_one_col(j, eq_mat, lsh_list, row):
    lsh_dict = {}
    for i in range(1, row):
        key = str(j) + ", " + str(lsh_list[i][j])
        if key in lsh_dict:
            eq_mat[i][j] = lsh_dict[key]
        else:
            lsh_dict[key] = i
    return np.array(eq_mat)[:, j]

def gen_eq_matrix(M, n, lsh_list, parallel):
    #parallel = 0
    row, col = M, n
    eq_mat = [[0 for i in range(col)] for j in range(row)]

    arr = []

    if parallel:
        eq_matrix_one_col_ = partial(eq_matrix_one_col, eq_mat=eq_mat, lsh_list=lsh_list, row=row)
        arr.append(Parallel(n_jobs=num_cores)(delayed(eq_matrix_one_col_)(j) for j in range(col)))

        arr = np.array(arr)
        eq_mat = arr.T


    else:
        lsh_dict = {}
        for j in range(col):
            if j%100 == 0:
                print("Current column is "+str(j))
            for i in range(1, row):
                key = str(j)+", "+str(lsh_list[i][j])
                if key in lsh_dict:
                    eq_mat[i][j] = lsh_dict[key]
                else:
                    lsh_dict[key] = i


    # if parallel:
    #     self.subtrees = Parallel(n_jobs=16)(
    #         delayed(eq_matrix_one_col)(eq_mat, j, lsh_list, row)
    #         for j in range(col))
    #     # for subtree_iter in self.subtrees:
    #     #     print("total depth "+str(subtree_iter.depth))
    #     self.total_nodes = sum([st.num_nodes for st in self.subtrees])
    # else:
    #     for j in range(col):
    #         for i in range(1, row):
    #             arg_min = 0
    #             for k in range(i):
    #                 if lsh_list[k][j] == lsh_list[i][j]:
    #                     arg_min = k+1
    #                     break
    #             eq_mat[i][j] = arg_min
    # pool = multiprocessing.Pool(processes=16)
    # pool.map(eq_matrix_one_col, args = (range(10), lsh_list, row))
    # pool.close()
    # pool.join()
    #
    return eq_mat

def is_valid_eq(eq_mat, k):
    flag = 0
    max_nonzero_count = []
    nonzero_rows = 0
    failed_rows = 0
    max_nonzero_count.append(0)
    for i in range(len(eq_mat)):
        count_nonzero = 0
        for j in range(len(eq_mat[0])):
            if eq_mat[i][j] != 0:
                count_nonzero += 1
                if(count_nonzero > max_nonzero_count[nonzero_rows]):
                    max_nonzero_count[nonzero_rows] = count_nonzero
            if count_nonzero == k:
                flag = 1
                #break
        if count_nonzero != 0:
            nonzero_rows += 1
            max_nonzero_count.append(0)
        if flag == 1:
            #max_nonzero_count.append(k)
            failed_rows += 1
            #break
    print("Max nonzero count per row: " + str(max_nonzero_count))
    print("Max nonzero count per matrix: " + str(max(max_nonzero_count)))
    print("Number of nonzero rows: " + str(nonzero_rows))
    print("Number of failed rows: " + str(failed_rows))
    return flag

def sample_codes(n, k, M, eq):
    coder = rs.RSCoder(n, k)
    codes = []
    for i in range(M):
        c = coder.encode(str(i))
        c = list(c)
        for j in range(len(c)):
            index = eq[i][j]
            if index != 0:
                c[j] = codes[index][j+1]
        codes.append(chr(i))
        for j in range(len(c)):
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






