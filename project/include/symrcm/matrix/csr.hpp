#ifndef SYMRCM_MATRIX_CSR_HPP_
#define SYMRCM_MATRIX_CSR_HPP_

#include <algorithm>
#include <symrcm/matrix/base.hpp>

namespace symrcm {
namespace matrix {

template <typename ScalarType, typename SizeType = int>
class Csr : public MatrixBase<ScalarType, SizeType> {
public:
    Csr(SizeType rows, SizeType cols, SizeType nz)
        : MatrixBase(rows, cols)
        , nnz(nz)
    {
        if (rows > 0 && cols > 0 && nnz > 0) {
            rowptr = new SizeType[rows + 1];
            colidx = new SizeType[nz];
            values = new ScalarType[nz];
        } else {
            rowptr = nullptr;
            colidx = nullptr;
            values = nullptr;
        }
    }

    Csr(SizeType rows, SizeType cols, SizeType nz,
        const SizeType *rptr, const SizeType *cidx, const ScalarType *vals)
        : Csr(rows, cols, nz)
    {
        if (rowptr != nullptr) {
            std::copy(rptr, rptr + this->get_rows() + 1, rowptr);
        }

        if (colidx != nullptr) {
            std::copy(cidx, cidx + nnz, colidx);
        }

        if (values != nullptr) {
            std::copy(vals, vals + nnz, values);
        }
    }

    Csr(Csr<ScalarType, SizeType> &other)
        : Csr(other.get_rows(), other.get_cols(), other.get_nnz(),
              other.get_rowptr(), other.get_colidx(), other.get_values())
    {}

    Csr(Csr<ScalarType, SizeType> &&other)
        : MatrixBase(other.get_rows(), other.get_cols())
    {
        this->nnz    = other.nnz;
        this->rowptr = other.rowptr;
        this->colidx = other.colidx;
        this->values = other.values;

        other.rowptr = nullptr;
        other.colidx = nullptr;
        other.values = nullptr;
    }

    ~Csr() {
        delete[] this->rowptr;
        delete[] this->colidx;
        delete[] this->values;
    }

    Csr<ScalarType, SizeType>& operator= (Csr<ScalarType, SizeType> &other) {
        this->_rows = other._rows;
        this->_cols = other._cols;

        this->nnz = other.nnz;
        this->rowptr = new SizeType[this->_rows + 1];
        this->coludx = new SizeType[this->nnz];
        this->values = new SizeType[this->nnz];

        if (this->rowptr != nullptr) {
            std::copy(other.rowptr, other.rowptr + other._rows + 1, this->rowptr);
        }

        if (this->colidx != nullptr) {
            std::copy(other.colidx, other.colidx + other.nnz, this->colidx);
        }

        if (this->values != nullptr) {
            std::copy(other.values, other.values + other.nnz, this->values);
        }

        return *this;
    }

    Csr<ScalarType, SizeType>& operator= (Csr<ScalarType, SizeType> &&other) {
        std::swap(this->_rows , other._rows);
        std::swap(this->_cols , other._cols);
        std::swap(this->nnz   , other.nnz);
        std::swap(this->rowptr, other.rowptr);
        std::swap(this->colidx, other.colidx);
        std::swap(this->values, other.values);

        return *this;
    }

    SizeType get_nnz() const noexcept { return nnz; }

    SizeType *get_rowptr() const noexcept { return rowptr; }

    SizeType *get_colidx() const noexcept { return colidx; }

    ScalarType *get_values() const noexcept { return values; }

    const SizeType *get_const_rowptr() const noexcept { return rowptr; }

    const SizeType *get_const_colidx() const noexcept { return colidx; }

    const ScalarType *get_const_values() const noexcept { return values; }

private:
    SizeType    nnz;
    SizeType   *rowptr;
    SizeType   *colidx;
    ScalarType *values;
};

}  // namespace matrix
}  // namespace symrcm

#endif
