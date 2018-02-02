CXX   = /usr/local/bin/g++-7
FLAGS = --std=c++11 -O3

all: demo.out

demo.out: demo.cpp
	$(CXX) $(FLAGS) -o $@ -I../include $^ -Wl,-rpath,../lib -L../lib -lsymrcm
