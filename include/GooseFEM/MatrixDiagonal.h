/**
 * Diagonal matrix.
 *
 * @file MatrixDiagonal.h
 * @copyright Copyright 2017. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#ifndef GOOSEFEM_MATRIXDIAGONAL_H
#define GOOSEFEM_MATRIXDIAGONAL_H

#include "Element.h"
#include "Matrix.h"
#include "config.h"

namespace GooseFEM {

/**
 * CRTP base class for a partitioned matrix with tying.
 */
template <class D>
class MatrixDiagonalBase {
public:
    /**
     * Underlying type.
     */
    using derived_type = D;

private:
    auto derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    auto derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

public:
    /**
     * @brief Solve \f$ x = A^{-1} b \f$.
     * Note that this does not involve a conversion to DOFs.
     *
     * In case of #GooseFEM::MatrixDiagonalPartitioned under the hood, schematically:
     * \f$ x_u = A_{uu}^{-1} (b_u - A_{up} * x_p) \equiv A_{uu}^{-1} b_u \f$
     * (again, no conversion to DOFs is needed).
     * Use GooseFEM::MatrixDiagonalPartitioned::Reaction() to get reaction forces.
     *
     * @param b nodevec [nelem, ndim].
     * @return x nodevec [nelem, ndim].
     */
    array_type::tensor<double, 2> Solve(const array_type::tensor<double, 2>& b)
    {
        array_type::tensor<double, 2> x = xt::empty_like(b);
        derived_cast().solve_nodevec_impl(b, x);
        return x;
    }

    /**
     * Solve \f$ x = A^{-1} b \f$.
     *
     * For #GooseFEM::MatrixDiagonalPartitioned under the hood solved
     * \f$ x_u = A_{uu}^{-1} (b_u - A_{up} * x_p) \equiv A_{uu}^{-1} b_u \f$.
     * Use GooseFEM::MatrixDiagonalPartitioned::Reaction() to get reaction forces.
     *
     * @param b dofval [ndof].
     * @return x dofval [ndof].
     */
    array_type::tensor<double, 1> Solve(const array_type::tensor<double, 1>& b)
    {
        array_type::tensor<double, 1> x = xt::empty_like(b);
        derived_cast().solve_dofval_impl(b, x);
        return x;
    }

    /**
     * Solve \f$ x = A^{-1} b \f$.
     *
     * For #GooseFEM::MatrixDiagonalPartitioned under the hood solved
     * \f$ x_u = A_{uu}^{-1} (b_u - A_{up} * x_p) \equiv A_{uu}^{-1} b_u \f$.
     * Use GooseFEM::MatrixDiagonalPartitioned::Reaction() to get reaction forces.
     *
     * @param b nodevec [nelem, ndim].
     * @param x (overwritten) nodevec [nelem, ndim].
     */
    void solve(const array_type::tensor<double, 2>& b, array_type::tensor<double, 2>& x)
    {
        derived_cast().solve_nodevec_impl(b, x);
    }

    /**
     * Solve \f$ x = A^{-1} b \f$.
     *
     * For #GooseFEM::MatrixDiagonalPartitioned under the hood solved
     * \f$ x_u = A_{uu}^{-1} (b_u - A_{up} * x_p) \equiv A_{uu}^{-1} b_u \f$.
     * Use GooseFEM::MatrixDiagonalPartitioned::Reaction() to get reaction forces.
     *
     * @param b nodevec [nelem, ndim].
     * @param x (overwritten) nodevec [nelem, ndim].
     */
    void solve(const array_type::tensor<double, 1>& b, array_type::tensor<double, 1>& x)
    {
        derived_cast().solve_dofval_impl(b, x);
    }
};

/**
 * Diagonal matrix.
 *
 * Warning: assemble() ignores all off-diagonal terms.
 *
 * See Vector() for bookkeeping definitions.
 */
class MatrixDiagonal : public MatrixBase<MatrixDiagonal>,
                       public MatrixDiagonalBase<MatrixDiagonal> {
private:
    friend MatrixBase<MatrixDiagonal>;
    friend MatrixDiagonalBase<MatrixDiagonal>;

public:
    MatrixDiagonal() = default;

    /**
     * Constructor.
     *
     * @tparam C e.g. `array_type::tensor<size_t, 2>`
     * @tparam D e.g. `array_type::tensor<size_t, 2>`
     * @param dofs DOFs per node [#nnode, #ndim].
     */
    template <class D>
    MatrixDiagonal(const D& dofs)
    {
        m_dofs = dofs;
        m_nnode = m_dofs.shape(0);
        m_ndim = m_dofs.shape(1);
        m_ndof = xt::amax(m_dofs)() + 1;
        m_A = xt::empty<double>({m_ndof});
        m_inv = xt::empty<double>({m_ndof});

        GOOSEFEM_ASSERT(m_ndof <= m_nnode * m_ndim);
    }

private:
    void clear_impl()
    {
        m_A.fill(0.0);
    }
    
    template <class T_ElemMat, class T_Conn>
    void assemble_impl(
        const T_ElemMat& elemmat,
        const T_Conn& conn_elem 
    )
    {
        size_t nelem = elemmat.shape()[0];
        size_t nnodes_per_elem = conn_elem.shape()[1]; 

        GOOSEFEM_ASSERT(elemmat.shape()[1] == nnodes_per_elem * m_ndim && "elemmat: Row dimension mismatch (expected num_nodes * ndim).");
        GOOSEFEM_ASSERT(elemmat.shape()[2] == nnodes_per_elem * m_ndim && "elemmat: Column dimension mismatch (expected num_nodes * ndim).");
        GOOSEFEM_ASSERT(conn_elem.shape()[0] == nelem && "conn_elem: Number of elements mismatch with elemmat.");
        GOOSEFEM_ASSERT(conn_elem.dimension() == 2 && "Connectivity 'conn_elem' must be a 2D array.");

        GOOSEFEM_ASSERT(Element::isDiagonal(elemmat)); 

        for (size_t e = 0; e < nelem; ++e) { 
            for (size_t m = 0; m < nnodes_per_elem; ++m) { 
                for (size_t i = 0; i < m_ndim; ++i) {
                    size_t global_node_id = conn_elem(e, m);
                    size_t global_dof_index = m_dofs(global_node_id, i); 

                    m_A(global_dof_index) += elemmat(e, m * m_ndim + i, m * m_ndim + i); 
                }
            }
        }
    }

    void finalize_impl()
    {
        m_changed = true;
    }  

    template <class T>
    void todense_impl(T& ret) const
    {
        ret.fill(0.0);

#pragma omp parallel for
        for (size_t d = 0; d < m_ndof; ++d) {
            ret(d, d) = m_A(d);
        }
    }

public:
    /**
     * Set all (diagonal) matrix components.
     * @param A The matrix [#ndof].
     */
    void set(const array_type::tensor<double, 1>& A)
    {
        GOOSEFEM_ASSERT(A.size() == m_ndof);
        std::copy(A.begin(), A.end(), m_A.begin());
        m_changed = true;
    }

    /**
     * Copy as diagonal matrix.
     * @return [#ndof].
     */
    [[deprecated]]
    const array_type::tensor<double, 1>& Todiagonal() const
    {
        return m_A;
    }

    /**
     * Underlying matrix
     * @return [#ndof].
     */
    const array_type::tensor<double, 1>& data() const
    {
        return m_A;
    }

private:
    template <class T>
    void dot_nodevec_impl(const T& x, T& b) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(x, {m_nnode, m_ndim}));
        GOOSEFEM_ASSERT(xt::has_shape(b, {m_nnode, m_ndim}));

#pragma omp parallel for
        for (size_t m = 0; m < m_nnode; ++m) {
            for (size_t i = 0; i < m_ndim; ++i) {
                b(m, i) = m_A(m_dofs(m, i)) * x(m, i);
            }
        }
    }

    template <class T>
    void dot_dofval_impl(const T& x, T& b) const
    {
        GOOSEFEM_ASSERT(x.size() == m_ndof);
        GOOSEFEM_ASSERT(b.size() == m_ndof);

        xt::noalias(b) = m_A * x;
    }

private:
    template <class T>
    void solve_nodevec_impl(const T& b, T& x)
    {
        GOOSEFEM_ASSERT(xt::has_shape(b, {m_nnode, m_ndim}));
        GOOSEFEM_ASSERT(xt::has_shape(x, {m_nnode, m_ndim}));

        this->factorize();

#pragma omp parallel for
        for (size_t m = 0; m < m_nnode; ++m) {
            for (size_t i = 0; i < m_ndim; ++i) {
                x(m, i) = m_inv(m_dofs(m, i)) * b(m, i);
            }
        }
    }

    template <class T>
    void solve_dofval_impl(const T& b, T& x)
    {
        GOOSEFEM_ASSERT(b.size() == m_ndof);
        GOOSEFEM_ASSERT(x.size() == m_ndof);
        this->factorize();
        xt::noalias(x) = m_inv * b;
    }

private:
    array_type::tensor<double, 1> m_A; ///< The matrix.
    array_type::tensor<double, 1> m_inv; /// Inverse of the matrix.

    /**
     * Compute inverse (automatically evaluated by "solve").
     */
    void factorize()
    {
        if (!m_changed) {
            return;
        }

#pragma omp parallel for
        for (size_t d = 0; d < m_ndof; ++d) {
            m_inv(d) = 1.0 / m_A(d);
        }

        m_changed = false;
    }
};

} // namespace GooseFEM

#endif
