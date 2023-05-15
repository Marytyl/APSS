# import unireedsolomon as rs
from os import path
import sys
sys.path.append(path.abspath('./UniReedSolomonm'))
from UniReedSolomonm import rs

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
    erasure_pos = []
    erasure_pos.append(0)
    for j in range(n):
        if str(j) + ", " + str(l_q[j]) in dict:
            c[j+1] = dict[str(j) + ", " + str(l_q[j])]
        if c[j+1] is not None:
            erase -= 1
        else:
            erasure_pos.append(j+1)
    if n - erase < k:
        return "No match", erase
    codeword = ""
    # print("c:", c)
    for char in c:
        if char is not None:
            codeword += char
        else:
            codeword += "\0"
    # print("codeword: ", codeword)

    coder = rs.RSCoder(n+1, k)
    # coder = RSCodec(n)
    dec = coder.decode_fast(codeword, erasures_pos=erasure_pos, only_erasures=False)
    # print("dec", dec)
    if dec != None:
        return dec[0], erase
    else:
        return None, erase


