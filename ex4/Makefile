CXX = g++
FLAGS = --std=c++11 -O3

CUDADIR = /usr/local/cuda-9.1
CUDAINC = $(CUDADIR)/include
CUDALIB = $(CUDADIR)/lib64
CUDALNK = -L$(CUDALIB) -lcudart

NVCC = $(CUDADIR)/bin/nvcc

all: ex4.out

ex4.out: ex4.o ex4.cuo
	$(CXX) $(FLAGS) $^ $(CUDALNK) -o $@

ex4.o: ex4.cpp
	$(CXX) $(FLAGS) -isystem $(CUDAINC) -c $^ -o $@

ex4.cuo: ex4.cu
	$(NVCC) $(FLAGS) -c $^ -o $@

.PHONY: clean

clean:
	$(RM) *.o *.cuo
