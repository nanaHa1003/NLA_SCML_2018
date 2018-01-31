#ifndef SYMRCM_MATRIX_BASE_HPP_
#define SYMRCM_MATRIX_BASE_HPP_

namespace symrcm {
namespace matrix {

template <typename ScalarType, typename SizeType = int>
class MatrixBase {
public:
    MatrixBase(SizeType rows, SizeType cols)
        : _rows(rows)
        , _cols(cols)
    {}

    virtual ~MatrixBase() {}

    SizeType get_rows() const noexcept { return _rows; }

    SizeType get_cols() const noexcept { return _cols; }

protected:
    // Number of rows of matrix
    SizeType _rows;
    // Number of cols of matrix
    SizeType _cols;
};

}  // namespace matrix
}  // namespace symrcm

#endif  // SYMRCM_MATRIX_BASE_HPP_
