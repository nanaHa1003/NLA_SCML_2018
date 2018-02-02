#ifndef SYMRCM_UTIL_PERMUTATION_HPP_
#define SYMRCM_UTIL_PERMUTATION_HPP_

namespace symrcm {
namespace util {

using namespace symrcm::matrix;

template <typename ScalarType, typename SizeType>
Coo<ScalarType, SizeType> Perm(
    const Coo<ScalarType, SizeType> &a,
    const SizeType                  *perm)
{
    SizeType row = a.get_rows();
    SizeType col = a.get_cols();
    SizeType nnz = a.get_nnz();

    // Target: R = A(perm(:), perm(:));
    Coo<ScalarType, SizeType> r(a);
    if (row > 0 && col > 0 && nnz > 0) {
        for (SizeType i = 0; i < nnz; ++i) {
            // Permutation
        }
    }
    return r;
}

template <typename ScalarType, typename SizeType>
Coo<ScalarType, SizeType> iPerm(
    const Coo<ScalarType, SizeType> &a,
    const SizeType                  *perm)
{
    SizeType row = a.get_rows();
    SizeType col = a.get_cols();
    SizeType nnz = a.get_nnz();

    Coo<ScalarType, SizeType> r(a);
    if (row > 0 && col > 0 && nnz > 0) {
        // inverse permutation
    }
    return r;
}

}  // util
}  // namespace symrcm

#endif
