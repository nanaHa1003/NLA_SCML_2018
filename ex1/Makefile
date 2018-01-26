FLAGS = --std=c++11 -march=native -O3

all: ex1.out

ex1.out: ex1.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $^ -o $@

plot: ex1.out
	./ex1.out > ex1.csv
	./plot.rb

.PHONY: clean

clean:
	$(RM) ex1.out
