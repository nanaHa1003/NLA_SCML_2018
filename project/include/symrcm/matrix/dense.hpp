#ifndef SYMRCM_MATRIX_DENSE_HPP_
#define SYMRCM_MATRIX_DENSE_HPP_

#include <algorithm>
#include <symrcm/matrix/base.hpp>

namespace symrcm {
namespace matrix {

template <typename ScalarType, typename SizeType>
class Dense : public MatrixBase<ScalarType, SizeType> {
public:
    Dense(SizeType rows, SizeType cols)
        : MatrixBase(rows, cols)
    {
        if (rows > 0 && cols > 0) {
            data = new ScalarType[rows * cols];
        } else {
            data = nullptr;
        }
    }

    Dense(SizeType rows, SizeType cols, const ScalarType *vals)
        : MatrixBase(rows, cols)
    {
        if (rows > 0 && cols > 0) {
            data = new ScalarType[rows * cols];
            std::copy(vals, vals + rows * cols, data);
        } else {
            data = nullptr;
        }
    }

    Dense(SizeType rows, SizeType cols, const ScalarType val)
        : MatrixBase(rows, cols)
    {
        if (rows > 0 && cols > 0) {
            data = new ScalarType[rows * cols];
            std::fill(data, data + rows * cols, val);
        } else {
            data = nullptr;
        }
    }

    Dense(Dense<ScalarType, SizeType> &other)
        : MatrixBase(other._rows, other._cols)
    {
        if (this->_rows > 0 && this->_cols > 0) {
            data = new ScalarType[this->_rows * this->_cols];
            std::copy(other.data, other.data + other._rows * other._cols, data);
        } else {
            data = nullptr;
        }
    }

    Dense(Dense<ScalarType, SizeType> &&other)
        : Dense(0, 0)
    {
        std::swap(this->_rows, other._rows);
        std::swap(this->_cols, other._cols);
        std::swap(this->data, other.data);
    }

    ~Dense() {
        delete[] data;
    }

    Dense<ScalarType, SizeType>& operator= (Dense<ScalarType, SizeType> &other)
    {
        this->_rows = other._rows;
        this->_cols = other._cols;

        if (this->_rows > 0 && this->_cols > 0) {
            data = new ScalarType[this->_rows * this_cols];
            std::copy(other.data, other.data + other._rows * other._cols, data);
        } else {
            data = nullptr;
        }

        return *this;
    }

    Dense<ScalarType, SizeType>& operator= (Dense<ScalarType, SizeType> &&other)
    {
        std::swap(this->_rows, other._rows);
        std::swap(this->_cols, other._cols);
        std::swap(this->data, other.data);
        return *this;
    }

    ScalarType *get_data() const noexcept { return data; }

    const ScalarType *get_const_data() const noexcept { return data; }

private:
    ScalarType *data;
};

}  // namespace matrix
}  // namespace symrcm

#endif  // SYMRCM_MATRIX_DENSE_HPP_
