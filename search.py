# import unireedsolomon as rs
from os import path
import sys
import OMapE
sys.path.append(path.abspath('./UniReedSolomonm'))
from UniReedSolomonm import rs

#from reedsolo import RSCodec, ReedSolomonError

from map import map as m


def search_query_dict(l_q, n, k, dict):
    c = [None for i in range(n+1)]
    erase = n
    erasure_pos = []
    erasure_pos.append(0)
    for j in range(n):
        try:
            res = dict[str(j) + ", " + str(l_q[j])]
            if  res is not None:
                c[j+1] = res
                erase -= 1
            else:
                erasure_pos.append(j+1)
        except KeyError:
            erasure_pos.append(j + 1)
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
    dec, ecc = coder.decode_fast(codeword, erasures_pos=erasure_pos, only_erasures=False,nostrip=True)
    full_corrected = dec + ecc
    errors=0
    for i in range(n+1):
        if i in erasure_pos:
            continue
        elif full_corrected[i] != codeword[i]:
            errors = errors+1
    dec, ecc = coder.decode_fast(codeword, erasures_pos=erasure_pos, only_erasures=False)
    # print("dec", dec)
    if dec != None:
        return dec[0], erase, errors
    else:
        return None, erase, errors


