#ifndef SYMRCM_MATRIX_COO_HPP_
#define SYMRCM_MATRIX_COO_HPP_

#include <algorithm>
#include <symrcm/matrix/base.hpp>

namespace symrcm {
namespace matrix {

template <typename ScalarType, typename SizeType = int>
class Coo : public MatrixBase<ScalsrType, SizeType> {
public:
    Coo(SizeType rows, SizeType cols, SizeType nz)
        : MatrixBase(rows, cols)
        , nnz(nz)
    {
        if (rows > 0 && cols > 0 && nnz > 0) {
            rowidx = new SizeType[nnz];
            colidx = new SizeType[nnz];
            values = new ScalarType[nnz];
        } else {
            rowptr = nullptr;
            colidx = nullptr;
            values = nullptr;
        }
    }

    Coo(SizeType rows, SizeType cols, SizeType nz,
        const SizeType *ridx, const SizeType *cidx, const ScalarType *vals)
        : Coo(rows, cols, nz)
    {
        if (rowidx != nullptr) {
            std::copy(ridx, ridx + nz, rowidx);
        }

        if (colidx != nullptr) {
            std::copy(cidx, cidx + nz, colidx);
        }

        if (values != nullptr) {
            std::copy(vals, vals + nz, values);
        }
    }

    Coo(Coo<ScalarType, SizeType> &other)
        : Coo(other._rows, other._cols, other.nnz,
              other.rowidx, other.colidx, other.nnz)
    {}

    Coo(Coo<ScalarType, SizeType> &&other)
        : MatrixBase(other._rows, other._cols)
    {
        this->nnz    = other.nnz;
        this->rowidx = other.rowidx;
        this->colidx = other.colidx;
        this->values = other.values;

        other.rowidx = nullptr;
        other.colidx = nullptr;
        other.values = nullptr;
    }

    ~Coo() {
        delete[] this->rowidx;
        delete[] this->colidx;
        delete[] this->values;
    }

    Csr<ScalarType, SizeType>& operator= (Csr<ScalarType, SizeType> &other) {
        this->_rows = other._rows;
        this->_cols = other._cols;

        this->nnz    = other.nnz;
        this->rowidx = new SizeType[this->nnz];
        this->colidx = new SizeType[this->nnz];
        this->values = new ScalarType[this->nnz];

        if (this->rowidx != nullptr) {
            std::copy(other.rowidx, other.rowidx + other.nnz, this->rowidx);
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
        std::swap(this->rowidx, other.rowidx);
        std::swap(this->colidx, other.colidx);
        std::swap(this->values, other.values);

        return *this;
    }

    SizeType get_nnz() const noexcept { return nnz; }

    SizeType *get_rowidx() const noexcept { return rowidx; }

    SizeType *get_colidx() const noexcept { return colidx; }

    SizeType *get_values() const noexcept { return values; }

    const SizeType *get_const_rowidx() const noexcept { return rowidx; }

    const SizeType *get_const_colidx() const noexcept { return colidx; }

    const SizeType *get_const_values() const noexcept { return values; }

private:
    SizeType    nnz;
    SizeType   *rowidx;
    SizeType   *colidx;
    ScalarType *values;
};

}  // namespace matrix
}  // namespace symrcm

#endif  // SYMRCM_MATRIX_COO_HPP_
