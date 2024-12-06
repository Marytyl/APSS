# import unireedsolomon as rs
from os import path
import sys
import OMapE
sys.path.append(path.abspath('./UniReedSolomonm'))
from UniReedSolomonm import rs

#from reedsolo import RSCodec, ReedSolomonError

from map import map as m

oram = 1
def search_query_dict(l_q, n, k, dict):
    c = [None for i in range(n+1)]
    erase = n
    erasure_pos = []
    erasure_pos.append(0)
    parallel_time = None
    for j in range(n):
        try:
            if oram == 1:
                (res, time_per_depth) = dict[str(j) + ", " + str(l_q[j])]
                if parallel_time is None:
                    parallel_time = time_per_depth
                else:
                    for depth in parallel_time:
                        if time_per_depth[depth]>parallel_time[depth]:
                            parallel_time[depth] = time_per_depth[depth]
            else:
                res = dict[str(j) + ", " + str(l_q[j])]
            if  res is not None:
                c[j+1] = res
                erase -= 1
            else:
                erasure_pos.append(j+1)
        except KeyError:
            erasure_pos.append(j + 1)
    if n - erase < k:
        return "No match", erase, 0
    codeword = ""
    # print("c:", c)
    for char in c:
        if char is not None:
            codeword += char
        else:
            codeword += "\0"
    print("Parallel time to process ",sum(parallel_time.values()))
    # print("codeword: ", codeword)

    coder = rs.RSCoder(n+1, k)
    # coder = RSCodec(n)
    try:
        dec, ecc = coder.decode_fast(codeword, erasures_pos=erasure_pos, only_erasures=False,nostrip=True)
        full_corrected = dec + ecc
        errors=0
        for i in range(n+1):
            if i in erasure_pos:
                continue
            elif full_corrected[i] != codeword[i]:
                errors = errors+1
        dec, ecc = coder.decode_fast(codeword, erasures_pos=erasure_pos, only_erasures=False)
        if dec != None:
            return dec[0], erase, errors, sum(parallel_time.values())
        else:
            return None, erase, errors, sum(parallel_time.values())
    except TypeError:
        return None, 0, 0, sum(parallel_time.values())


