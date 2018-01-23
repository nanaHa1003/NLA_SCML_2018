CXX = /usr/local/bin/g++-7
FLAGS = --std=c++11 -O3 -fopenmp -fPIC -pedantic -Wall -Wshadow

all: ex3.out

ex3.out: ex3.cpp
	$(CXX) $(FLAGS) $^ -o $@

.PHONY: clean

clean:
	$(RM) ex3.out
