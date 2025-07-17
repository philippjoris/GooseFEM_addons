/**
 * Methods to switch between storage types based on a mesh and DOFs that are partitioned in:
 * -   unknown DOFs
 * -   prescribed DOFs
 * -   dependent DOFs
 *
 * @file VectorPartitionedTyings.h
 * @copyright Copyright 2017. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#pragma once

#include "Mesh.h"
#include "Vector.h"
#include "assertions.h"
#include "config.h"

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace GooseFEM {

/**
 * @brief Class to switch between storage types for DOFs partitioned into
 * unknown, prescribed, and dependent.
 *
 * This class extends the `Vector` class to handle systems with linear
 * tyings between DOFs, typically arising from periodic boundary conditions
 * or multi-point constraints. It manages the mapping between:
 * - Unknown DOFs (u)
 * - Prescribed DOFs (p)
 * - Independent DOFs (i = u + p)
 * - Dependent DOFs (d)
 *
 * For reference:
 * -   "nodevec": nodal vectors [#nnode, #ndim].
 * -   "elemvec": nodal vectors stored per element [nelem, #nne, #ndim].
 * -   "dofval": DOF values [#ndof].
 * -   "dofval_u": DOF values (Unknown), `== dofval[iiu]`, [#nnu].
 * -   "dofval_p": DOF values (Prescribed), `== dofval[iip]`, [#nnp].
 * -   "dofval_i": DOF values (Independent), `== dofval[iii]`, [#nni].
 * -   "dofval_d": DOF values (Dependent), `== dofval[iid]`, [#nnd].
 */
class VectorPartitionedTyings : public Vector {
private:
    array_type::tensor<size_t, 1> m_iiu; ///< See iiu().
    array_type::tensor<size_t, 1> m_iip; ///< See iip().
    array_type::tensor<size_t, 1> m_iii; ///< See iii().
    array_type::tensor<size_t, 1> m_iid; ///< See iid().
    size_t m_nnu; ///< See nnu().
    size_t m_nnp; ///< See nnp().
    size_t m_nni; ///< See nni().
    size_t m_nnd; ///< See nnd().
    Eigen::SparseMatrix<double> m_Cdu; ///< Tying matrix, see Tyings::Periodic::Cdu().
    Eigen::SparseMatrix<double> m_Cdp; ///< Tying matrix, see Tyings::Periodic::Cdp().
    Eigen::SparseMatrix<double> m_Cdi; ///< Tying matrix, see Tyings::Periodic::Cdi().
    Eigen::SparseMatrix<double> m_Cud; ///< Transpose of "m_Cdu".
    Eigen::SparseMatrix<double> m_Cpd; ///< Transpose of "m_Cdp".
    Eigen::SparseMatrix<double> m_Cid; ///< Transpose of "m_Cdi".

private:
    /**
     * @brief Convert to "dofval" (overwrite entries that occur more than once).
     * Only the dependent DOFs are retained.
     *
     * @param nodevec nodal vectors [#nnode, #ndim].
     * @return dofval[iid()] [#nnd].
     */
    template <class T>
    Eigen::VectorXd Eigen_asDofs_d(const T& nodevec) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(nodevec, {this->nnode(), this->ndim()})); // Use base class nnode, ndim

        Eigen::VectorXd dofval_d(m_nnd); // Correct size for Eigen::VectorXd

        for (size_t m = 0; m < this->nnode(); ++m) {
            for (size_t i = 0; i < this->ndim(); ++i) {
                size_t global_dof_id = this->dofs()(m, i);
                if (global_dof_id >= m_nni && global_dof_id < (m_nni + m_nnd)) {
                    dofval_d(global_dof_id - m_nni) = nodevec(m, i);
                }
            }
        }
        return dofval_d;
    }

public:
    VectorPartitionedTyings() = default;

    /**
     * @brief Constructor.
     *
     * @tparam E e.g. `array_type::tensor<size_t, 2>` (for dofs)
     * @tparam M e.g. `Eigen::SparseMatrix<double>` (for tying matrices)
     * @param dofs DOFs per node [#nnode, #ndim].
     * @param Cdu Tying matrix, see Tyings::Periodic::Cdu().
     * @param Cdp Tying matrix, see Tyings::Periodic::Cdp().
     * @param Cdi Tying matrix, see Tyings::Periodic::Cdi().
     */
    template <class E, class M>
    VectorPartitionedTyings(const E& dofs, const M& Cdu, const M& Cdp, const M& Cdi)
        : Vector(dofs), m_Cdu(Cdu), m_Cdp(Cdp), m_Cdi(Cdi) 
    {
        GOOSEFEM_ASSERT(Cdu.rows() == Cdp.rows());
        GOOSEFEM_ASSERT(Cdi.rows() == Cdp.rows());

        m_nnu = static_cast<size_t>(m_Cdu.cols());
        m_nnp = static_cast<size_t>(m_Cdp.cols());
        m_nnd = static_cast<size_t>(m_Cdp.rows()); 
        m_nni = m_nnu + m_nnp; 

        m_iiu = xt::arange<size_t>(m_nnu);
        m_iip = xt::arange<size_t>(m_nnu, m_nnu + m_nnp);
        m_iii = xt::arange<size_t>(m_nni); 
        m_iid = xt::arange<size_t>(m_nni, m_nni + m_nnd); 

        m_Cud = m_Cdu.transpose();
        m_Cpd = m_Cdp.transpose();
        m_Cid = m_Cdi.transpose();

        GOOSEFEM_ASSERT(static_cast<size_t>(m_Cdi.cols()) == m_nni);
    }

    /** @return Number of dependent DOFs. */
    size_t nnd() const
    {
        return m_nnd;
    }

    /** @return Number of independent DOFs. */
    size_t nni() const
    {
        return m_nni;
    }

    /** @return Number of independent unknown DOFs. */
    size_t nnu() const
    {
        return m_nnu;
    }

    /** @return Number of independent prescribed DOFs. */
    size_t nnp() const
    {
        return m_nnp;
    }

    /** @return Dependent DOFs (list of global DOF numbers) [#nnd]. */
    const array_type::tensor<size_t, 1>& iid() const
    {
        return m_iid;
    }

    /** @return Independent DOFs (list of global DOF numbers) [#nni]. */
    const array_type::tensor<size_t, 1>& iii() const
    {
        return m_iii;
    }

    /** @return Independent unknown DOFs (list of global DOF numbers) [#nnu]. */
    const array_type::tensor<size_t, 1>& iiu() const
    {
        return m_iiu;
    }

    /** @return Independent prescribed DOFs (list of global DOF numbers) [#nnp]. */
    const array_type::tensor<size_t, 1>& iip() const
    {
        return m_iip;
    }

    /**
     * @brief Copy (part of) "dofval" to another "dofval": dofval_dest[iip()] = dofval_src[iip()].
     *
     * @param dofval_src DOF values, iip() updated, [#ndof].
     * @param dofval_dest DOF values, iip() updated, [#ndof].
     */
    template <class T>
    void copy_p(const T& dofval_src, T& dofval_dest) const
    {
        GOOSEFEM_ASSERT(dofval_src.dimension() == 1);
        GOOSEFEM_ASSERT(dofval_dest.dimension() == 1);
        GOOSEFEM_ASSERT(dofval_src.size() == this->ndof()); // Use base class ndof
        GOOSEFEM_ASSERT(dofval_dest.size() == this->ndof()); // Use base class ndof

#pragma omp parallel for
        for (size_t i = m_nnu; i < m_nni; ++i) { // Loop over independent prescribed DOFs
            dofval_dest(m_iip(i - m_nnu)) = dofval_src(m_iip(i - m_nnu)); // Access using original iip indices
        }
    }

    template <class T>
    array_type::tensor<double, 1> AsDofs_i(const T& nodevec) const
    {
        array_type::tensor<double, 1> dofval = xt::empty<double>({m_nni});
        this->asDofs_i(nodevec, dofval);
        return dofval;
    }

    template <class T, class R>
    void asDofs_i(const T& nodevec, R& dofval_i, bool apply_tyings = true) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(nodevec, {this->nnode(), this->ndim()}));
        GOOSEFEM_ASSERT(dofval_i.size() == m_nni);

        dofval_i.fill(0.0);

#pragma omp parallel for
        for (size_t m = 0; m < this->nnode(); ++m) {
            for (size_t i = 0; i < this->ndim(); ++i) {
                size_t global_dof_id = this->dofs()(m, i);
                if (global_dof_id < m_nni) { 
                    dofval_i(global_dof_id) = nodevec(m, i); 
                }
            }
        }

        if (!apply_tyings) {
            return;
        }

        Eigen::VectorXd Dofval_d = this->Eigen_asDofs_d(nodevec); 
        Eigen::VectorXd Dofval_i_from_d = m_Cid * Dofval_d; 

#pragma omp parallel for
        for (size_t i = 0; i < m_nni; ++i) {
            dofval_i(i) += Dofval_i_from_d(i);
        }
    }
};

} // namespace GooseFEM