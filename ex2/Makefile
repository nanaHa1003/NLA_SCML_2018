FLAGS = --std=c++11 -march=native -O3 -pedantic

all: ex2.out

ex2.out: ex2.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $^ -o $@

plot: ex2.out
	./ex2.out > ex2.csv
	./plot.rb

.PHONY: clean

clean:
	$(RM) ex2.out
