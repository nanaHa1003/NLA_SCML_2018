#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>

template <typename T>
T dot(int n, T* x, int incx, T* y, int incy) noexcept {
    T sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += x[i * incx] * y[i * incy];
    }
    return sum;
}

int main(int argc, char **argv) {
    std::vector<int>    test_sizes = {{ 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 }};
    std::vector<double> elap_times;

    elap_times.reserve(test_sizes.size());

    std::cout << std::setprecision(8);
    std::cout.setf(std::ios_base::fixed);

    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << test_sizes[i] << ",";
    }
    std::cout << test_sizes.back() << std::endl;

    for (int &size: test_sizes) {
        std::vector<double> x(size), y(size);

        auto start = std::chrono::high_resolution_clock::now();
        double retval = dot(size, x.data(), 1, y.data(), 1);
        static_cast<void>(retval);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        elap_times.push_back(diff.count());
    }

    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << elap_times[i] << ",";
    }
    std::cout << elap_times.back() << std::endl;

    return 0;
}
