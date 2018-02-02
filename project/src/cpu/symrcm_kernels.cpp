#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <symrcm/cpu/symrcm_kernels.hpp>

namespace symrcm {
namespace cpu {

using namespace symrcm::matrix;

template <typename ScalarType, typename SizeType>
void count_degree(const Csr<ScalarType, SizeType> &A, SizeType *degrees) {
    #pragma omp parallel for
    for (SizeType i = 0; i < A.get_rows(); ++i) {
        degrees[i] = A.get_const_rowptr()[i + 1] - A.get_const_rowptr()[i];
    }
}

template void count_degree(const Csr<float , int>&, int *);
template void count_degree(const Csr<double, int>&, int *);

template <typename ScalarType, typename SizeType>
void spmspv_select2nd_min(
    const Csr<ScalarType, SizeType>                   &A,
    const std::vector<std::tuple<SizeType, SizeType>> &v,
    std::vector<std::tuple<SizeType, SizeType>>       &y)
{
    typedef std::vector<std::tuple<SizeType, SizeType>> SpVec;
    std::vector<SpVec> work(v.size());

    #pragma omp parallel for
    for (typename SpVec::size_type i = 0; i < v.size(); ++i) {
        const SizeType *rowptr = A.get_const_rowptr();
        const SizeType *colidx = A.get_const_colidx();
        work[i].resize(rowptr[i + 1] - rowptr[i]);

        for (SizeType j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            work[i][j - rowptr[i]] = std::tie(colidx[j], std::get<1>(v[j - rowptr[i]]));
        }
    }

    std::unordered_map<SizeType, SizeType> y_list;
    for (SpVec &Ax_i: work) {
        for (std::tuple<SizeType, SizeType> &ax_i: Ax_i) {
            auto it = y_list.find(std::get<0>(ax_i));
            if (it == y_list.end()) {
                y_list[std::get<0>(ax_i)] = std::get<1>(ax_i);
            } else {
                auto &key = std::get<0>(ax_i);
                y_list[key] = std::min(std::get<1>(ax_i), y_list[key]);
            }
        }
    }

    y.resize(0);
    for (auto &y_i: y_list) {
        y.push_back(y_i);
    }
}

template <typename ScalarType, typename SizeType>
SizeType find_pseudo_peripheral_vertex(
    Csr<ScalarType, SizeType> &A,
    const SizeType            *degrees)
{
    typedef std::tuple<SizeType, SizeType> SpVecVal;
    typedef std::vector<SpVecVal>          SpVec;
    typedef typename SpVec::size_type      size_type;

    SizeType n = A.get_rows();
    std::vector<SizeType> L(n);

    SpVec L_cur, L_nxt, L_tmp;
    L_cur.reserve(n);
    L_nxt.reserve(n);
    L_tmp.reserve(n);

    SizeType r = 0;

    int l = 0, nlvl = -1;
    while (l > nlvl) {
        std::fill(L.begin(), L.end(), n + 1);
        L_cur.emplace_back(r, 0);
        L[r] = 0;
        do {
            #pragma omp parallel for
            for (size_type i = 0; i < L_cur.size(); ++i) {
                L[std::get<0>(L_cur[i])] = std::get<1>(L_cur[i]);
            }

            spmspv_select2nd_min(A, L_cur, L_nxt);

            for (size_type i = 0; i < L_nxt.size(); ++i) {
                if (L[std::get<0>(L_nxt[i])] == n + 1) {
                    L_tmp.push_back(L_nxt[i]);
                }
            }
            L_nxt.swap(L_tmp);
            L_tmp.resize(0);

            if (L_nxt.size()) {
                #pragma omp parallel for
                for (size_type i = 0; i < L_nxt.size(); ++i) {
                    std::get<1>(L_nxt[i]) = L[std::get<0>(L_nxt[i])];
                }
                L_cur = L_nxt;
            }
            l += 1;
        } while (L_nxt.size());

        SpVecVal v;
#if __cplusplus == 201703L
        v = std::reduce(
            L_cur.begin(),
            L_cur.end(),
            [&](SpVecVal &a, SpVecVal &b) {
                if (degrees[std::get<0>(a)] < degrees[std::get<0>(b)]) return a;
                return b;
            });
#else
        v = L_cur[0];
        for (size_type i = 0; i < L_cur.size(); ++i) {
            if (degrees[std::get<0>(L_cur[i])] < degrees[std::get<0>(v)]) {
                v = L_cur[i];
            }
        }
#endif
        r = std::get<0>(v);
    }

    return r;
}

template int find_pseudo_peripheral_vertex(Csr<float , int>&, const int *);
template int find_pseudo_peripheral_vertex(Csr<double, int>&, const int *);

template <typename ScalarType, typename SizeType>
void reverse_cuthill_mckee(
    Csr<ScalarType, SizeType> &A,
    const SizeType            *degrees,
    SizeType                   r0,
    SizeType                  *perm)
{
    if (A.get_rows() != A.get_cols()) return;

    SizeType n = A.get_rows();
    std::fill(perm, perm + n, SizeType(n + 1));

    typedef std::vector<std::tuple<SizeType, SizeType>> SpVec;
    SpVec L_cur, L_nxt, L_tmp;
    L_cur.reserve(n);
    L_nxt.reserve(n);
    L_tmp.reserve(n);

    L_cur.emplace_back(r0, 0);
    perm[r0] = 0;
    SizeType nv = 1;

    while (L_cur.size()) {
        spmspv_select2nd_min(A, L_cur, L_nxt);

        typedef typename SpVec::size_type size_type;
        for (size_type i = 0; i < L_nxt.size(); ++i) {
            if (perm[std::get<0>(L_nxt[i])] == n + 1) {
                L_tmp.push_back(L_nxt[i]);
            }
        }
        L_nxt.swap(L_tmp);

        typedef std::tuple<SizeType, SizeType> SpVecVal;
        std::sort(L_nxt.begin(), L_nxt.end(), [&](SpVecVal &a, SpVecVal &b){
            if (degrees[std::get<0>(a)] < degrees[std::get<0>(b)]) return true;
            return false;
        });

        #pragma omp parallel for
        for (size_type i = 0; i < L_nxt.size(); ++i) {
            perm[std::get<0>(L_nxt[i])] = std::get<1>(L_nxt[i]) + nv;
        }

        nv += L_nxt.size();

        L_cur.swap(L_nxt);
        L_cur.resize(0);
        L_tmp.resize(0);
    }

    std::reverse(perm, perm + n);
}

template void reverse_cuthill_mckee(Csr<float , int>&, const int *, int, int *);
template void reverse_cuthill_mckee(Csr<double, int>&, const int *, int, int *);

}  // namespace cpu
}  // namespace symrcm
