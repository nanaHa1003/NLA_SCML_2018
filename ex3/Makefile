CXX = g++
FLAGS = --std=c++11 -march=native -O3 -DNDEBUG -fopenmp -fPIC -pedantic -Wall -Wshadow

all: ex3.out

ex3.out: ex3.cpp
	$(CXX) $(FLAGS) $^ -o $@

.PHONY: clean

clean:
	$(RM) ex3.out
