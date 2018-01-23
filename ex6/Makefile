CXX   = g++
FLAGS = --std=c++11 -O3

MPI_PATH = /home/u/nanaha/local
MPI_INC  = -I$(MPI_PATH)/include
MPI_LIB  = $(MPI_PATH)/lib
MPI_LNK  = -L$(MPI_LIB) -lmpi

all: ex6.out

ex6.out: ex6.o
	$(CXX) $(FLAGS) -o $@ $^ $(MPI_LNK)

ex6.o: ex6.cpp
	$(CXX) $(FLAGS) $(MPI_INC) -c $^

.PHONY: clean

clean:
	$(RM) *.o
