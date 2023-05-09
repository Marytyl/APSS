# This Makefile runs tests and builds the package to upload to pypi
# To use this Makefile, pip install py-make
# You also need to pip install also other required modules: `pip install flake8 coverage twine validate-pyproject[all] rstcheck` , or simply `pip install --editable .[test,testmeta]`
# Then, cd to this folder, and type `pymake -p` to list all commands, then `pymake <command>` to run the related entry.
# To test on multiple Python versions, install them, install also the C++ redistributables for each (so that Cython works), and then type `pymake testtox`.
# To pymake buildupload (deploy on pypi), you need to `pip install cython` and install a C++ compiler, on Windows and with Python 3.7 you need Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools": https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/

.PHONY:
	alltests
	all
	flake8
	test
	testsetup
    testsetuppost
	testcoverage
    testtox
	distclean
	coverclean
	prebuildclean
	clean
    toxclean
	installdev
	install
	build
	buildupload
	pypi
	help

help:
	@+make -p

alltests:
	@+make test
	@+make flake8
	@+make testsetup

all:
	@+make alltests
	@+make build

flake8:
	@+flake8 -j 8 --count --statistics --exit-zero .

test:
     # Build the Cython extension
	python setup.py build_ext --inplace
     # Run the tests
	python -m unittest discover tests

testnose:
    nosetests -vv --with-coverage

testtox:
    # Test for multiple Python versions
	tox --skip-missing-interpreters -p all

testsetup:
	python setup.py check --metadata --restructuredtext --strict

testsetuppost:
	twine check "dist/*"

testcoverage:
     # This is the preferred way to run the tests since Python 3.10
	@+make coverclean
     # Build the Cython extension and install module in editable mode
	#python setup.py build_ext --inplace --cythonize  # unnecessary to call build_ext --inplace now
     #python -m pip install -e . --config-setting="--build-option=--cythonize"
	python setup.py develop --cythonize  # unfortunately much faster than current pep517 options which do not allow to only build the extension
     # Run the tests
	# nosetests reedsolo --with-coverage --cover-package=reedsolo --cover-erase --cover-min-percentage=80 -d -v
     # With PyTest, it is now necessary to first install the python module so that it is found (--cov=<module>)
     #python setup.py develop
     #pytest --cov-report term-missing --cov-config=.coveragerc --cov=. tests/ --cov-branch
     #python setup.py develop --uninstall
	coverage run --branch -m pytest -v
	coverage report -m

testrst:
	rstcheck README.rst

testpyproject:
	validate-pyproject pyproject.toml -v

distclean:
	@+make coverclean
	@+make prebuildclean
	@+make clean
    @+make toxclean
prebuildclean:
	@+python -c "import shutil; shutil.rmtree('build', True)"
	@+python -c "import shutil; shutil.rmtree('dist', True)"
	@+python -c "import shutil; shutil.rmtree('src/unireedsolomon.egg-info', True)"
coverclean:
	@+python -c "import os; os.remove('.coverage') if os.path.exists('.coverage') else None"
	@+python -c "import shutil; shutil.rmtree('__pycache__', True)"
	@+python -c "import shutil; shutil.rmtree('tests/__pycache__', True)"
clean:
	@+python -c "import os, glob; [os.remove(i) for i in glob.glob('*.py[co]')]"
	@+python -c "import os, glob; [os.remove(i) for i in glob.glob('tests/*.py[co]')]"
toxclean:
	@+python -c "import shutil; shutil.rmtree('.tox', True)"

installdev:
	@+make prebuildclean
	@+python -m pip install --upgrade --editable .[test,testmeta] --config-setting="--build-option=--cythonize" --verbose --use-pep517

install:
	@+make prebuildclean
	@+python -m pip install --upgrade . --config-setting="--build-option=--cythonize" --verbose --use-pep517

build:
	@+make testrst
	@+make prebuildclean
	@+make testpyproject
	# For build, do NOT use the -w flag, otherwise only the wheel will be built, but we need sdist for source distros such as Debian and Gentoo!
	@+python -sBm build --config-setting="--build-option=--cythonize"
	@+make testsetuppost

pypi:
	twine upload dist/*

buildupload:
	@+make build
	@+make pypi
