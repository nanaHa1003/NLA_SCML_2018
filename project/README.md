Project Topic
====================

To implement parallel reverse Cuthill-McKee algorithm for symmertic matrices
using OpenMP.

Status
====================

Originally I plan to implement reverse Cuthill-McKee in gingko, but I
encountered some issues such like

  - What should a reordering algorithm be?

  - Should a generated permutation be a LinOp itself?

, and so on. So I decide to make it standalone.

Currently my code can be compiled, but there are still some bugs in it.

Algorithm & Data Structure
==========================

The algorithm I implemented is from the paper
[The Reverse Cuthill-McKee Algorithm in Distributed-Memory](#).

The data format I used for adjacency matrix is CSR, more precisely, the
`rowptr` and `colidx` of the input matrix.

The data format for sparse vector I used is a list of tuples. I use
`std::vector<std::tuple<int, int>>` in my implementation. And in order to
reduce memory allocation during the calculation, I have preallocated each
vector to its maximum possible size. Using a memory pool with tree based data
structure may have less memory usage and better performance.

Since this code is my very fisrt version, I only use `#pragma omp parallel for`
to parallelize some data independent loops. For algorithms like `sort`, `fill`
or `reduce`, I just simply use STL's implementation and enable parallel version
of STL by adding `-D_GLIBCXX_PARALLEL`.

Compile and Run
====================

Although my program is still buggy, it can be compiled use `make`. But the
script may not run correctly on other machines.

The compiler I used is `g++` 7.2.0, the generated shared library
`libsymrcm.dylib` and `libsymrcm_omp.dylib` will be placed in `lib`.

To run the the demo program, go to `demo` and run `demo.out`. You may have to
set `DYLD_LIBRARY_PATH` to `lib`, this is a little bit wired since I have pass
the rpath to the linker.
