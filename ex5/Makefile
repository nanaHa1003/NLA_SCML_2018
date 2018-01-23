NVCC = nvcc
FLAGS = --std=c++11 -O3

all: ex5.out

ex5.out: spmv_framework.o mmio.o
	$(NVCC) $(FLAGS) $^ -o $@

mmio.o: mmio.cu
	$(NVCC) $(FLAGS) -c $^

spmv_framework.o: spmv_framework.cu
	$(NVCC) $(FLAGS) -I. -c $^

.PHONY: clean

clean:
	$(RM) *.o
