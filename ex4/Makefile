CXX = g++
NVCC = nvcc
FLAGS = --std=c++11 -O3

all: ex4

ex4: ex4.o ex4.cuo
	$(NVCC) $(FLAGS) $^ -o $@

ex4.o: ex4.cpp
	$(CXX) $(FLAGS) -c $^ -o $@

ex4.cuo: ex4.cu
	$(NVCC) $(FLAGS) -c $^ -o $@
