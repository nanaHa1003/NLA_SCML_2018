#ifndef SYMRCM_UTIL_CONVERT_HPP_
#define SYMRCM_UTIL_CONVERT_HPP_

#include <vector>
#include <tuple>
#include <algorithm>

#include <symrcm/matrix/coo.hpp>
#include <symrcm/matrix/csr.hpp>
#include <symrcm/matrix/dense.hpp>

namespace symrcm {
namespace util {

using namespace symrcm::matrix;

template <typename ScalarType, typename SizeType>
Csr<ScalarType, SizeType> CooToCsr(const Coo<ScalarType, SizeType> &A) {
    SizeType row, col, nnz;
    row = A.get_rows();
    col = A.get_cols();
    nnz = A.get_nnz();

    Csr<ScalarType, SizeType> Ret(row, col, nnz);
    SizeType   *rowptr = Ret.get_rowptr();
    SizeType   *colidx = Ret.get_colidx();
    ScalarType *values = Ret.get_values();
    if (row > 0 && col > 0 && nnz > 0) {
        typedef std::tuple<SizeType, SizeType, ScalarType> CooValue;
        std::vector<CooValue> temp(nnz);
        for (typename std::vector<CooValue>::size_type i = 0; i < nnz; ++i) {
            temp[i] = std::tuple<SizeType, SizeType, ScalarType>(
                A.get_const_rowidx()[i],
                A.get_const_colidx()[i],
                A.get_const_values()[i]);
        }

        std::sort(temp.begin(), temp.end(), [&](CooValue &a, CooValue &b) {
            if (std::get<0>(a) < std::get<0>(b)) return true;
            if (std::get<1>(a) < std::get<1>(b)) return true;
            return false;
        });

        std::fill(rowptr, rowptr + row + 1, 0);
        for (SizeType i = 0; i < nnz; ++i) {
            SizeType ridx = std::get<0>(temp[i]);
            rowptr[ridx + 1] += 1;
            colidx[i] = std::get<1>(temp[i]);
            values[i] = std::get<2>(temp[i]);
        }
    }
    return Ret;
}

template <typename ScalarType, typename SizeType>
Coo<ScalarType, SizeType> CsrToCoo(const Csr<ScalarType, SizeType> &A) {
    SizeType row, col, nnz;
    row = A.get_rows();
    col = A.get_cols();
    nnz = A.get_nnz();

    Coo<ScalarType, SizeType> Ret(row, col, nnz);
    SizeType   *rowidx = Ret.get_rowidx();
    SizeType   *colidx = Ret.get_colidx();
    ScalarType *values = Ret.get_values();
    if (row > 0 && col > 0 && nnz > 0) {
        int ptr = 0;
        for (SizeType i = 0; i < row; ++i) {
            const SizeType *rowptr = A.get_const_rowptr();
            for (SizeType j = rowptr[i]; j < rowptr[i + 1]; ++j) {
                rowidx[ptr] = i;
                colidx[ptr] = A.get_const_colidx()[j];
                values[ptr] = A.get_const_values()[j];
                ++ptr;
            }
        }
    }
    return Ret;
}

template <typename ScalarType, typename SizeType>
Dense<ScalarType, SizeType> CooToDense(const Coo<ScalarType, SizeType> &A) {
    SizeType row, col, nnz;
    row = A.get_rows();
    col = A.get_cols();
    nnz = A.get_nnz();

    Dense<ScalarType, SizeType> Ret(row, col);
    if (row > 0 && col > 0) {
        ScalarType *data = Ret.get_data();
        for (SizeType i = 0; i < nnz; ++i) {
            const SizeType &ridx = A.get_const_rowidx()[i];
            const SizeType &cidx = A.get_const_colidx()[i];
            data[ridx * Ret.get_cols() + cidx] = A.get_const_values()[i];
        }
    }
    return Ret;
}

template <typename ScalarType, typename SizeType>
Dense<ScalarType, SizeType> CsrToDense(const Csr<ScalarType, SizeType> &A) {
    SizeType row, col, nnz;
    row = A.get_rows();
    col = A.get_cols();
    nnz = A.get_nnz();

    const SizeType   *rowptr = A.get_const_rowptr();
    const SizeType   *colidx = A.get_const_colidx();
    const ScalarType *values = A.get_const_values();

    Dense<ScalarType, SizeType> Ret(row, col);
    if (row > 0 && col > 0) {
        ScalarType *data = Ret.get_data();
        for (SizeType i = 0; i < row; ++i) {
            for (SizeType j = rowptr[i]; j < rowptr[i + 1]; ++j) {
                data[colidx[j] * Ret.get_cols() + i] = values[j];
            }
        }
    }
    return Ret;
}

}  // namespace util
}  // namespace symrcm

#endif  // SYMRCM_UTIL_CONVERT_HPP_
