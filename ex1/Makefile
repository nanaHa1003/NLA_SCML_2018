CXX = g++
CXXFLAGS ?= --std=c++11 -O3 -march=native

all: ex1

ex1: ex1.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean

clean:
	$(RM) ex1
