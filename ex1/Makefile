FLAGS = --std=c++11 -O3

all: ex1

ex1: ex1.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $^ -o $@

plot: ex1
	./ex1 > ex1.csv
	./plot.rb

.PHONY: clean

clean:
	$(RM) ex1
