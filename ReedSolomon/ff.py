# Copyright (c) 2010 Andrew Brown <brownan@cs.duke.edu, brownan@gmail.com>
# See LICENSE.txt for license terms
import pandas as pd

class GF256int(int):
    """Instances of this object are elements of the field GF(2^8)
    Instances are integers in the range 0 to 255
    This field is defined using the irreducable polynomial
    x^8 + x^4 + x^3 + x + 1
    and using 3 as the generator for the exponent table and log table.
    """
    # Maps integers to GF256int instances
    cache = {}
    # Exponent table for 3, a generator for GF(256)
    exptable = []
    with open('ReedSolomon/gf216exp.txt') as f:
        for line in f:
            for s in line.split(', '):
                exptable.append(int(s))


    # Logarithm table, base 3
    # logtable = list(open('ReedSolomon/gf216log.txt').read())
    logtable = []
    logtable.append(None)
    with open('ReedSolomon/gf216log.txt') as g:
        for line in g:
            for r in line.split(', '):
                logtable.append(int(r))


    def __new__(cls, value):
        # Check cache
        # Caching sacrifices a bit of speed for less memory usage. This way,
        # there are only a max of 256 instances of this class at any time.
        try:
            return GF256int.cache[value]
        except KeyError:
            if value > 65535 or value < 0:
                raise ValueError("Field elements of GF(2^16) are between 0 and 65535. Cannot be %s" % value)

            newval = int.__new__(cls, value)
            GF256int.cache[int(value)] = newval
            return newval

    def __add__(a, b):
        "Addition in GF(2^8) is the xor of the two"
        a = GF256int(a)
        b = GF256int(b)
        return GF256int(a ^ b)
    __sub__ = __add__
    __radd__ = __add__
    __rsub__ = __add__
    def __neg__(self):
        return self
    
    def __mul__(a, b):
        "Multiplication in GF(2^8)"
        if a == 0 or b == 0:
            return GF256int(0)
        x = GF256int.logtable[a]
        y = GF256int.logtable[b]
        z = (x + y) % 65535
        return GF256int(GF256int.exptable[z])
    __rmul__ = __mul__

    def __pow__(self, power):
        if isinstance(power, GF256int):
            raise TypeError("Raising a Field element to another Field element is not defined. power must be a regular integer")
        x = GF256int.logtable[self]
        z = (x * power) % 65535
        return GF256int(GF256int.exptable[z])

    def inverse(self):
        e = int(GF256int.logtable[self])
        return GF256int(GF256int.exptable[65535 - e])

    def __div__(self, other):
        return self * GF256int(other).inverse()
    def __rdiv__(self, other):
        return self.inverse() * other

    def __repr__(self):
        n = self.__class__.__name__
        return "%s(%r)" % (n, int(self))

    def multiply(self, other):
        """A slow multiply method. This method gives the same results as the
        other multiply method, but is implemented to illustrate how it works
        and how the above tables were generated.

        This procedure is called Peasant's Algorithm (I believe)
        """
        a = int(self)
        b = int(other)

        p = a
        r = 0
        while b:
            if b & 1: r = r ^ p
            b = b >> 1
            p = p << 1
            if p & 0x10000: p = p ^ 0x1100b

        return GF256int(r)
