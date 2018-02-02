CXX   = /usr/local/bin/g++-7
FLAGS = --std=c++11 -O3

all: demo.out

demo.out: lib/libsymrcm.dylib
	make -C demo

lib/libsymrcm.dylib:
	make -C src

.PHONY: clean

clean:
	$(RM) demo/demo.out lib/libsymrcm.dylib
