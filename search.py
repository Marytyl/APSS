# import unireedsolomon as rs
from os import path
import sys
sys.path.append(path.abspath('./ReedSolomon'))
from ReedSolomon import rs

#from reedsolo import RSCodec, ReedSolomonError

from map import map as m

# def search_query(l_query, lsh_list, k):
#     n = len(lsh_list[0])
#
#     c = []
#     erase = n
#     for j in range(n):
#         c[j] = m.retrieve(j, l_query(j))
#         if c[j] is not None:
#             erase -= 1
#     if n-erase > k:
#         return None
#     codeword = ""
#     for j in range(1, n):
#         codeword += c[j]
#     coder = rs.RSCoder(n, k)
#     dec = coder.decode(codeword)
#     print(dec[0])

def search_query_dict(l_q, lsh_list, k, dict):
    n = len(lsh_list[0])
    c = [None for i in range(n+1)]
    erase = n
    for j in range(n):
        if str(j) + ", " + str(l_q[j]) in dict:
            c[j+1] = dict[str(j) + ", " + str(l_q[j])]
        if c[j+1] is not None:
            erase -= 1
    if n - erase < k:
        return ("No match")
    codeword = ""
    print(c)
    for char in c:
        if char is not None:
            codeword += char
        else:
            codeword += ""
    coder = rs.RSCoder(n, k)
    # coder = RSCodec(n)
    dec= coder.decode(codeword)

    return(dec[0])


