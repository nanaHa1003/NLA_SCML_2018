FLAGS = --std=c++11 -O3 -pedantic

all: ex2

ex2: ex2.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $^ -o $@

plot: ex2
	./ex2 > ex2.csv
	./plot.rb

.PHONY: clean

clean:
	$(RM) ex2
