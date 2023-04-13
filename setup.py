import numpy as np
import unireedsolomon as rs
import map as m

# def build_index(m, lsh_list):
#     index = m, lsh_list
#     return index

def gen_eq_matrix(M, n, lsh_list):
    row, col = M, n
    eq_mat = [[0] * col] * row
    for i in range(row):
        for j in range(col):
            arg_min = 0
            for k in range(i):
                if lsh_list[i][j] == lsh_list[k][j]:
                    arg_min = k
                    break
            eq_mat[i][j] = arg_min
    return eq_mat

def is_valid_eq(eq_mat, k):
    flag = 0
    max_nonzero_count = 0
    for i in range(len(eq_mat)):
        count_nonzero = 0
        for j in range(len(eq_mat[0])):
            if eq_mat[i][j] != 0:
                count_nonzero += 1
                if(count_nonzero > max_nonzero_count):
                    max_nonzero_count = count_nonzero
            if count_nonzero == k:
                flag = 1
                break
        if flag == 1:
            max_nonzero_count = k
            break
    print("Max nonzero count "+str(max_nonzero_count))
    return flag

def sample_codes(n, k, M, eq):
    coder = rs.RSCoder(n, k)
    codes = []
    for i in range(M):
        c = coder.encode(i)
        c = chr(i) + c
        for j in range(n):
            index = eq[i][j]
            if index != 0:
                c[i][j] = c[index][j]
        codes[i] = c
    return codes

def gen_map(codes, M, n, lsh_list):
    for i in range(M):
        for j in range(n):
            m.insert(chr(j)+lsh_list[i](j), codes[i](j))






