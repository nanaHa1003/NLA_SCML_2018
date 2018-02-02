CXX   = /usr/local/bin/g++-7
FLAGS = --std=c++11 -O3

all: dylib omp_dylib

dylib: symrcm_kernels.cpp
	$(CXX) -shared $(FLAGS) -fPIC -o ../../lib/libsymrcm.dylib -I../../include $^

omp_dylib: symrcm_kernels.cpp
	$(CXX) -shared $(FLAGS) -fPIC -D_GLIBCXX_PARALLEL -fopenmp -o ../../lib/libsymrcm_omp.dylib -I../../include $^
