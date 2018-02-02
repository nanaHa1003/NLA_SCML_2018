#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <symrcm.hpp>

symrcm::matrix::Coo<double, int> read_mm_spd(const char *filename) {
    std::ifstream mmfile(filename);

    std::vector<char> buffer(4096); // max line size of mm format
    do {
        mmfile.getline(buffer.data(), 4096);
    } while(buffer[0] == '%');

    int m, n, nnz;
    sscanf(buffer.data(), "%d%d%d", &m, &n, &nnz);

    nnz = 2 * nnz + m; // Since matrix is SPD and only lower half is stored

    symrcm::matrix::Coo<double, int> ret(m, n, nnz);
    for (int i = 0; i < nnz; ++i) {
        mmfile.getline(buffer.data(), 4096);

        int    row, col;
        double val;
        sscanf(buffer.data(), "%d%d%lf", &row, &col, &val);

        // Convert to zero based index
        ret.get_rowidx()[i] = row - 1;
        ret.get_colidx()[i] = col - 1;
        ret.get_values()[i] = val;
    }
    return ret;
}

void write_mm_spd(
    const char                             *filename,
    const symrcm::matrix::Coo<double, int> &A)
{
    std::ofstream mmfile(filename);

    mmfile << "%%MatrixMarket matrix coordinate real symmetric\n";

    int rows = A.get_rows();
    int cols = A.get_cols();
    int nnz  = (A.get_nnz() - rows) / 2 + rows;
    mmfile << rows << " " << cols << " " << nnz << "\n";

    for (int i = 0; i < A.get_nnz(); ++i) {
        if (A.get_const_colidx()[i] <= A.get_const_rowidx()[i]) {
            mmfile << A.get_const_colidx()[i] << " "
                   << A.get_const_rowidx()[i] << " "
                   << A.get_const_values()[i] << "\n";
        }
    }
}

int main() {
    auto coo = read_mm_spd("data/1138_bus.mtx");
    auto csr = symrcm::util::CooToCsr(coo);

    std::vector<int> degrees(csr.get_rows());
    symrcm::cpu::count_degree(csr, degrees.data());

    int r0 = symrcm::cpu::find_pseudo_peripheral_vertex(csr, degrees.data());

    std::vector<int> perm(csr.get_rows());
    symrcm::cpu::reverse_cuthill_mckee(csr, degrees.data(), r0, perm.data());

    // Apply perm on matrix on COO matrix

    write_mm_spd("data/1138_bus_p.mtx", coo);

    return 0;
}
