from map import map as m
import unireedsolomon as rs
from main import compute_eLSH


def search_query(q, lsh_list, k):
    n = len(lsh_list)
    l = compute_eLSH(q)
    c = []
    erase = n
    for j in range(n):
        c[j] = m.retrieve(j, l(j))
        if c[j] is not None:
            erase -= 1
    if n-erase > k:
        return None
    codeword = ""
    for j in range(1, n):
        codeword += c[j]
    coder = rs.RSCoder(n, k)
    dec = coder.decode(codeword)
    print(dec[0])

