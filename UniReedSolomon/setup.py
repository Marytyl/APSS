# See:
# https://docs.python.org/2/distutils/setupscript.html
# http://docs.cython.org/src/reference/compilation.html
# https://docs.python.org/2/extending/building.html
# http://docs.cython.org/src/userguide/source_files_and_compilation.html
# http://www.ewencp.org/blog/a-brief-introduction-to-packaging-python/
# http://stackoverflow.com/questions/12966216/make-distutils-in-python-automatically-find-packages
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
# http://blog.ionelmc.ro/2014/06/25/python-packaging-pitfalls/
# To add a commandline entry point: http://www.scotttorborg.com/python-packaging/command-line-scripts.html
# Also see pyroma and checkmanifest tools.
#
# This is only kept to be able to cythonize and compile the extensions, all config parameters are declared in pyproject.toml for Py3 or setup.cfg for Py2 according to PEP517 standard.
# Note: Even if setuptools implement in the future a declarative form for cythonized and C extensions, an empty setup.py is required for retrocompatibility with older versions of pip that do not support pyproject.toml-only projects, especially to install in editable mode, see: https://github.com/pypa/setuptools/issues/2816


try:
    from setuptools import setup, find_packages
    from setuptools import Extension
except ImportError:
    from distutils.core import setup, find_packages
    from distutils.extension import Extension

import os, sys

if '--cythonize' in sys.argv or os.getenv('UNIREED_CYTHONIZE'):  # --cythonize is usable through PEP517 by supplying --config-setting="--build-option=--cythonize" in PEP517-compliant tools (pip, build, setuptools, pytest, cibuildwheel). REEDSOLO_CYTHONIZE env variable was necessary before for cibuildwheel but not anymore, it is only kept as an alternative option, but is not used.
    # Remove the special argument, otherwise setuptools will raise an exception
    if '--cythonize' in sys.argv:
        sys.argv.remove('--cythonize')
    # If Cython is installed, transpile the optimized Cython module to C and compile as a .pyd to be distributed
    try:
        from Cython.Build import cythonize
        print("Cython is installed, building creedsolo module")
        extensions = cythonize([
                    Extension('unireedsolomon.cff', [os.path.join('src', 'unireedsolomon', 'cff.pyx')]),
                    Extension('unireedsolomon.cpolynomial', [os.path.join('src', 'unireedsolomon', 'cpolynomial.pyx')]),
                 ], annotate=True, force=True, compiler_directives={'embedsignature': True, 'binding': False, 'initializedcheck': True})
    except ImportError:
        # Else Cython is not installed (or user explicitly wanted to skip)
        # Else run in pure python mode (no compilation)
        print("WARNING: Cython is not installed despite specifying --cythonize, the compiled modules will NOT be built.")
        extensions = None
elif '--native-compile' in sys.argv:
    sys.argv.remove('--native-compile')
    # Compile pyd from pre-transpiled creedsolo.c
    # Here we use an explicit flag to compile, whereas implicit fallback is recommended by Cython, but in practice it's too difficult to maintain, because some people on Windows have Cython installed but no C compiler https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
    print("Notice: Compiling the C modules from the pre-transpiled cff.c and cpolynomial.c files using the locally installed C compiler...")
    extensions = [
                Extension('unireedsolomon.cff', [os.path.join('src', 'unireedsolomon', 'cff.c')]),
                Extension('unireedsolomon.cpolynomial', [os.path.join('src', 'unireedsolomon', 'cpolynomial.c')]),
             ]
else:
    extensions = None


setup(
    ext_modules = extensions,
)
