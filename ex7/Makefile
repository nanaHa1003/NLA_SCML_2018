CXX   = g++
FLAGS = --std=c++11 -O3

CUDADIR = /usr/local/cuda-9.1
CUDAINC = -I$(CUDADIR)/include
CUDALIB = $(CUDADIR)/lib64
CUDALNK = -L$(CUDALIB) -lcudart

CUDNNDIR = /home/u/nanaha/local/modules/Cellar/CuDNN/7.0
CUDNNINC = -I$(CUDNNDIR)/include
CUDNNLIB = $(CUDNNDIR)/lib64
CUDNNLNK = -L$(CUDNNLIB) -lcudnn

OPENCVDIR = /home/u/nanaha/local/modules/Cellar/3.4
OPENCVINC = -I$(OPENCVDIR)/include
OPENCVLIB = $(OPENCVDIR)/lib
OPENCVLNK = -L$(OPENCVLIB) -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

all: ex7.out

ex7.out: ex7.cpp
	$(CXX) $(FLAGS) $(CUDAINC) $(CUDNNINC) $(OPENCVINC) -o $@ $^ $(CUDNNLNK) $(OPENCVLNK) $(CUDALNK)

