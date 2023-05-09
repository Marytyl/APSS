TODO
====

See also the [TODO.md on pyFileFixity](https://github.com/lrq3000/pyFileFixity/blob/master/TODO.md).

Essentially, the priority is to implement a faster encoder, using [FFT](https://github.com/lrq3000/unireedsolomon/issues/2) or [NTT](https://github.com/Bulat-Ziganshin/FastECC/issues/5).

* paper on easy fft implementation (remember I want fast encoding, not necessarily decoding)
Simple algorithms for decoding systematic Reed–Solomon codes https://link.springer.com/article/10.1007/s10623-012-9626-1
Fast transform decoding of nonsystematic Reed-Solomon codes https://ieeexplore.ieee.org/document/48982
A fully programmable reed-solomon decoder on a multi-core processor platform https://www.jstage.jst.go.jp/article/transinf/E95.D/12/E95.D_2939/_pdf/-char/ja
Fast Erasure Coding for Data Storage: A Comprehensive Study of the Acceleration Techniques https://www.usenix.org/system/files/fast19-zhou.pdf
FFT-based fast Reed-Solomon codes with arbitrary block lengths and rates https://www.researchgate.net/publication/3350462_FFT-based_fast_Reed-Solomon_codes_with_arbitrary_block_lengths_and_rates
Fast Reed-Solomon Interactive Oracle Proofs of Proximity https://drops.dagstuhl.de/opus/volltexte/2018/9018/pdf/LIPIcs-ICALP-2018-14.pdf
https://www.backblaze.com/blog/reed-solomon/
AFF3CT toolbox: https://www.sciencedirect.com/science/article/pii/S2352711019300457 and https://github.com/aff3ct/aff3ct
* the reply I got about fft on stackoverflow
* pyldpc (and the other error codes library implementing everything including polar codes) https://github.com/hichamjanati/pyldpc
Self-Learning Tool for LDPC Codes using Python https://www.researchgate.net/publication/319395601_Self-Learning_Tool_for_LDPC_Codes_using_Python
LDPCin-SSD: Making advanced error correction codes work effectively in solid state drives LDPCin-SSD: Making advanced error correction codes work effectively in solid state drives
lpdec python library: https://pythonhosted.org/lpdec/apidoc/codes/linear.html
https://github.com/AhmedElkelesh/Genetic-Algorithm-based-LDPC-Code-Design
https://www.biorxiv.org/content/10.1101/770032v1.full
tuto using pyldpc: http://mdelrosario.com/2019/06/27/ldpc-example.html
https://github.com/radfordneal/LDPC-codes
* parallelization:
    * https://github.com/luispedro/jug
    * Numba and Pythran can work with SIMD! https://wiki.python.org/moin/ParallelProcessing and Dispy!
    * Deap, nested parallel maps: http://code.google.com/p/deap/
    * joblib https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
    * VecPy SIMD: https://github.com/undefx/vecpy
    * Dask https://dask.org/
    * https://wwoods.github.io/job_stream/
    * https://www.parallelpython.com/
    * https://github.com/daleroberts/pypar
