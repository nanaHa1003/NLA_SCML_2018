#ifndef SYMRCM_CPU_SYMRCM_KERNELS_HPP_
#define SYMRCM_CPU_SYMRCM_KERNELS_HPP_

#include <symrcm/matrix/csr.hpp>

namespace symrcm {
namespace cpu {

template <typename ScalarType, typename SizeType = int>
void count_degree(matrix::Csr<ScalarType, SizeType> &A, SizeType *degrees);

template <typename ScalarType, typename SizeType = int>
SizeType find_pseudo_peripheral_vertex(
    matrix::Csr<ScalarType, SizeType> &A,
    const SizeType                    *degrees);

template <typename ScalarType, typename SizeType = int>
void reverse_cuthill_mckee(
    matrix::Csr<ScalarType, SizeType> &A,
    const SizeType                    *degrees,
    SizeType                          *perm);

}  // namespace cpu
}  // namespace symrcm

#endif  // SYMRCM_CPU_SYMRCM_KERNELS_HPP_
