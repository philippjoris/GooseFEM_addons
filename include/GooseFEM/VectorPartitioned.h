/**
 * Methods to switch between storage types based on a mesh and DOFs that are partitioned in:
 * -   unknown DOFs
 * -   prescribed DOFs
 *
 * @file VectorPartitioned.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#pragma once

#include "Mesh.h"
#include "Vector.h"
#include "assertions.h"
#include "config.h"

namespace GooseFEM {

/**
 * @brief Class to switch between storage types, specifically for partitioned DOF systems.
 *
 * It extends the base `Vector` class by providing functionality to handle
 * 'unconstrained' (u) and 'prescribed' (p) DOFs. Connectivity is now passed
 * explicitly to methods that require it for element-based operations.
 *
 * For reference:
 * -   "dofval": DOF values [#ndof].
 * -   "dofval_u": unknown DOF values, `== dofval[iiu()]`, [#nnu].
 * -   "dofval_p": prescribed DOF values, `== dofval[iip()]`, [#nnp].
 * -   "nodevec": nodal vectors [#nnode, #ndim].
 * -   "elemvec": nodal vectors stored per element [#nelem, #nne, #ndim].
 */
class VectorPartitioned : public Vector {
protected:
    array_type::tensor<size_t, 1> m_iiu; ///< See iiu()
    array_type::tensor<size_t, 1> m_iip; ///< See iip()
    size_t m_nnu; ///< See #nnu
    size_t m_nnp; ///< See #nnp

    /**
     * @brief Boolean mask: `True` if DOF (global ID) is unconstrained.
     * This is a 1D mask over global DOFs.
     */
    array_type::tensor<bool, 1> m_dofs_is_u_mask;

    /**
     * @brief Boolean mask: `True` if DOF (global ID) is prescribed.
     * This is a 1D mask over global DOFs.
     */
    array_type::tensor<bool, 1> m_dofs_is_p_mask;

    /**
     * @brief Renumbered DOFs per node, such that:
     * iiu = arange(nnu)
     * iip = nnu + arange(nnp)
     * This maps a global DOF ID to its partitioned ID.
     */
    array_type::tensor<size_t, 1> m_part1d;

    /**
     * @brief Renumbered DOFs per node (node-based view), such that:
     * iiu = arange(nnu)
     * iip = nnu + arange(nnp)
     * This maps `(node_id, dim_id)` to its partitioned ID.
     */
    array_type::tensor<size_t, 2> m_part_node_view;

public:
    VectorPartitioned() = default;

    /**
     * @brief Constructor.
     *
     * @param dofs DOFs per node [#nnode, #ndim].
     * @param iip prescribed DOFs [#nnp].
     */
    template <class T_Dofs, class T_Iip>
    VectorPartitioned(const T_Dofs& dofs, const T_Iip& iip)
        : Vector(dofs), m_iip(iip) // Call base class constructor with dofs
    {
        GOOSEFEM_ASSERT(is_unique(iip));
        GOOSEFEM_ASSERT(xt::amax(iip)() < this->ndof()); // iip values must be less than total DOFs

        // Calculate unconstrained DOFs
        m_iiu = xt::setdiff1d(xt::eval(xt::arange<size_t>(this->ndof())), m_iip);
        m_nnp = m_iip.size();
        m_nnu = m_iiu.size();

        // Create 1D masks for u/p DOFs based on global DOF IDs
        m_dofs_is_u_mask = xt::zeros<bool>({this->ndof()});
        m_dofs_is_p_mask = xt::zeros<bool>({this->ndof()});
        xt::view(m_dofs_is_u_mask, xt::keep(m_iiu)) = true;
        xt::view(m_dofs_is_p_mask, xt::keep(m_iip)) = true;

        // Create the 1D partitioned mapping (global DOF ID -> partitioned ID)
        m_part1d = xt::empty<size_t>({this->ndof()});
        for (size_t i = 0; i < m_iiu.size(); ++i) {
            m_part1d(m_iiu(i)) = i;
        }
        for (size_t i = 0; i < m_iip.size(); ++i) {
            m_part1d(m_iip(i)) = m_nnu + i;
        }

        // Create the 2D partitioned mapping (node_id, dim_id -> partitioned ID)
        // This relies on the global dofs() mapping from the base class
        m_part_node_view = Mesh::Reorder({m_iiu, m_iip}).apply(this->dofs());
    }

    /** @return Number of unknown DOFs. */
    size_t nnu() const { return m_nnu; }

    /** @return Number of prescribed DOFs. */
    size_t nnp() const { return m_nnp; }

    /** @return Unknown DOFs [#nnu]. */
    const array_type::tensor<size_t, 1>& iiu() const { return m_iiu; }

    /** @return Prescribed DOFs [#nnp]. */
    const array_type::tensor<size_t, 1>& iip() const { return m_iip; }

    /**
     * @brief Per DOF (global ID) list if unknown ("u").
     * @return Boolean array [#ndof].
     */
    const array_type::tensor<bool, 1>& dofs_is_u() const { return m_dofs_is_u_mask; }

    /**
     * @brief Per DOF (global ID) list if prescribed ("p").
     * @return Boolean array [#ndof].
     */
    const array_type::tensor<bool, 1>& dofs_is_p() const { return m_dofs_is_p_mask; }

    /**
     * @brief Copy unknown DOFs from "nodevec" to another "nodevec".
     *
     * `nodevec_dest[vector.dofs_is_u()] = nodevec_src`
     * The other DOFs are taken from `nodevec_dest`.
     *
     * @param nodevec_src input [#nnode, #ndim]
     * @param nodevec_dest input [#nnode, #ndim]
     * @return nodevec output [#nnode, #ndim]
     */
    array_type::tensor<double, 2> Copy_u(
        const array_type::tensor<double, 2>& nodevec_src,
        const array_type::tensor<double, 2>& nodevec_dest
    ) const
    {
        array_type::tensor<double, 2> ret = nodevec_dest;
        this->copy_u(nodevec_src, ret);
        return ret;
    }

    /**
     * @brief Copy unknown DOFs from "nodevec" to another "nodevec" (in-place).
     *
     * `nodevec_dest[vector.dofs_is_u()] = nodevec_src`
     * The other DOFs are taken from `nodevec_dest`.
     *
     * @param nodevec_src input [#nnode, #ndim]
     * @param nodevec_dest input/output [#nnode, #ndim]
     */
    void copy_u(
        const array_type::tensor<double, 2>& nodevec_src,
        array_type::tensor<double, 2>& nodevec_dest
    ) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(nodevec_src, {this->nnode(), this->ndim()}));
        GOOSEFEM_ASSERT(xt::has_shape(nodevec_dest, {this->nnode(), this->ndim()}));

        // Access using the 2D partitioned map (node, dim) -> partitioned_id
#pragma omp parallel for
        for (size_t m = 0; m < this->nnode(); ++m) {
            for (size_t i = 0; i < this->ndim(); ++i) {
                if (m_part_node_view(m, i) < m_nnu) { // If this DOF is unconstrained
                    nodevec_dest(m, i) = nodevec_src(m, i);
                }
            }
        }
    }

    /**
     * @brief Copy prescribed DOFs from "nodevec" to another "nodevec".
     *
     * `nodevec_dest[vector.dofs_is_p()] = nodevec_src`
     * The other DOFs are taken from `nodevec_dest`.
     *
     * @param nodevec_src input [#nnode, #ndim]
     * @param nodevec_dest input [#nnode, #ndim]
     * @return nodevec output [#nnode, #ndim]
     */
    array_type::tensor<double, 2> Copy_p(
        const array_type::tensor<double, 2>& nodevec_src,
        const array_type::tensor<double, 2>& nodevec_dest
    ) const
    {
        array_type::tensor<double, 2> ret = nodevec_dest;
        this->copy_p(nodevec_src, ret);
        return ret;
    }

    /**
     * @brief Copy prescribed DOFs from "nodevec" to another "nodevec" (in-place).
     *
     * `nodevec_dest[vector.dofs_is_p()] = nodevec_src`
     * The other DOFs are taken from `nodevec_dest`.
     *
     * @param nodevec_src input [#nnode, #ndim]
     * @param nodevec_dest input/output [#nnode, #ndim]
     */
    void copy_p(
        const array_type::tensor<double, 2>& nodevec_src,
        array_type::tensor<double, 2>& nodevec_dest
    ) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(nodevec_src, {this->nnode(), this->ndim()}));
        GOOSEFEM_ASSERT(xt::has_shape(nodevec_dest, {this->nnode(), this->ndim()}));

        // Access using the 2D partitioned map (node, dim) -> partitioned_id
#pragma omp parallel for
        for (size_t m = 0; m < this->nnode(); ++m) {
            for (size_t i = 0; i < this->ndim(); ++i) {
                if (m_part_node_view(m, i) >= m_nnu) { // If this DOF is prescribed
                    nodevec_dest(m, i) = nodevec_src(m, i);
                }
            }
        }
    }

    /**
     * @brief Combine unknown and prescribed "dofval" into a single "dofval" list.
     *
     * @param dofval_u input [#nnu]
     * @param dofval_p input [#nnp]
     * @return dofval output [#ndof]
     */
    array_type::tensor<double, 1> DofsFromPartitioned(
        const array_type::tensor<double, 1>& dofval_u,
        const array_type::tensor<double, 1>& dofval_p
    ) const
    {
        array_type::tensor<double, 1> dofval = xt::empty<double>({this->ndof()});
        this->dofsFromPartitioned(dofval_u, dofval_p, dofval);
        return dofval;
    }

    /**
     * @brief Combine unknown and prescribed "dofval" into a single "dofval" list (in-place).
     *
     * @param dofval_u input [#nnu]
     * @param dofval_p input [#nnp]
     * @param dofval output [#ndof]
     */
    void dofsFromPartitioned(
        const array_type::tensor<double, 1>& dofval_u,
        const array_type::tensor<double, 1>& dofval_p,
        array_type::tensor<double, 1>& dofval
    ) const
    {
        GOOSEFEM_ASSERT(dofval_u.size() == m_nnu);
        GOOSEFEM_ASSERT(dofval_p.size() == m_nnp);
        GOOSEFEM_ASSERT(dofval.size() == this->ndof());

        dofval.fill(0.0); // Initialize to zero before filling

#pragma omp parallel for
        for (size_t d = 0; d < m_nnu; ++d) {
            dofval(m_iiu(d)) = dofval_u(d);
        }

#pragma omp parallel for
        for (size_t d = 0; d < m_nnp; ++d) {
            dofval(m_iip(d)) = dofval_p(d);
        }
    }

    /**
     * @brief Combine unknown and prescribed "dofval" into a single "dofval" list
     * and directly convert to "nodevec" without a temporary
     * (overwrite entries that occur more than once).
     *
     * @param dofval_u input [#nnu]
     * @param dofval_p input [#nnp]
     * @return nodevec output [#nnode, #ndim]
     */
    array_type::tensor<double, 2> NodeFromPartitioned(
        const array_type::tensor<double, 1>& dofval_u,
        const array_type::tensor<double, 1>& dofval_p
    ) const
    {
        array_type::tensor<double, 2> nodevec = xt::empty<double>({this->nnode(), this->ndim()});
        this->nodeFromPartitioned(dofval_u, dofval_p, nodevec);
        return nodevec;
    }

    /**
     * @brief Combine unknown and prescribed "dofval" into a single "dofval" list
     * and directly convert to "nodevec" without a temporary (in-place).
     * (overwrite entries that occur more than once).
     *
     * @param dofval_u input [#nnu]
     * @param dofval_p input [#nnp]
     * @param nodevec output [#nnode, #ndim]
     */
    void nodeFromPartitioned(
        const array_type::tensor<double, 1>& dofval_u,
        const array_type::tensor<double, 1>& dofval_p,
        array_type::tensor<double, 2>& nodevec
    ) const
    {
        GOOSEFEM_ASSERT(dofval_u.size() == m_nnu);
        GOOSEFEM_ASSERT(dofval_p.size() == m_nnp);
        GOOSEFEM_ASSERT(xt::has_shape(nodevec, {this->nnode(), this->ndim()}));

#pragma omp parallel for
        for (size_t m = 0; m < this->nnode(); ++m) {
            for (size_t i = 0; i < this->ndim(); ++i) {
                size_t partitioned_id = m_part_node_view(m, i);
                if (partitioned_id < m_nnu) {
                    nodevec(m, i) = dofval_u(partitioned_id);
                }
                else {
                    nodevec(m, i) = dofval_p(partitioned_id - m_nnu);
                }
            }
        }
    }

    /**
     * @brief Combine unknown and prescribed "dofval" into a single "dofval" list
     * and directly convert to "elemvec" without a temporary.
     * (overwrite entries that occur more than once).
     *
     * @param dofval_u input [#nnu]
     * @param dofval_p input [#nnp]
     * @param conn connectivity [#nelem, #nne].
     * @return elemvec output [#nelem, #nne, #ndim]
     */
    array_type::tensor<double, 3> ElementFromPartitioned(
        const array_type::tensor<double, 1>& dofval_u,
        const array_type::tensor<double, 1>& dofval_p,
        const array_type::tensor<size_t, 2>& conn // Added conn
    ) const
    {
        size_t nelem = conn.shape(0);
        size_t nne = conn.shape(1);
        array_type::tensor<double, 3> elemvec = xt::empty<double>(this->shape_elemvec(nelem, nne));
        this->elementFromPartitioned(dofval_u, dofval_p, conn, elemvec); // Pass conn
        return elemvec;
    }

    /**
     * @brief Combine unknown and prescribed "dofval" into a single "dofval" list
     * and directly convert to "elemvec" without a temporary (in-place).
     * (overwrite entries that occur more than once).
     *
     * @param dofval_u input [#nnu]
     * @param dofval_p input [#nnp]
     * @param conn connectivity [#nelem, #nne].
     * @param elemvec output [#nelem, #nne, #ndim]
     */
    void elementFromPartitioned(
        const array_type::tensor<double, 1>& dofval_u,
        const array_type::tensor<double, 1>& dofval_p,
        const array_type::tensor<size_t, 2>& conn, // Added conn
        array_type::tensor<double, 3>& elemvec
    ) const
    {
        GOOSEFEM_ASSERT(dofval_u.size() == m_nnu);
        GOOSEFEM_ASSERT(dofval_p.size() == m_nnp);
        GOOSEFEM_ASSERT(xt::has_shape(elemvec, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Use conn shapes
        GOOSEFEM_ASSERT(conn.dimension() == 2); // Ensure conn is 2D

#pragma omp parallel for
        for (size_t e = 0; e < conn.shape(0); ++e) { // Use conn.shape(0) for nelem
            for (size_t m = 0; m < conn.shape(1); ++m) { // Use conn.shape(1) for nne
                for (size_t i = 0; i < this->ndim(); ++i) {
                    size_t global_dof_id = this->dofs()(conn(e, m), i); // Get global DOF ID
                    size_t partitioned_id = m_part1d(global_dof_id); // Map to partitioned ID

                    if (partitioned_id < m_nnu) {
                        elemvec(e, m, i) = dofval_u(partitioned_id);
                    }
                    else {
                        elemvec(e, m, i) = dofval_p(partitioned_id - m_nnu);
                    }
                }
            }
        }
    }

    /**
     * @brief Extract the unknown "dofval": `dofval[iiu()]`.
     * @param dofval input [#ndof]
     * @return dofval_u output [#nnu]
     */
    array_type::tensor<double, 1> AsDofs_u(const array_type::tensor<double, 1>& dofval) const
    {
        array_type::tensor<double, 1> dofval_u = xt::empty<double>({m_nnu});
        this->asDofs_u(dofval, dofval_u);
        return dofval_u;
    }

    /**
     * @brief Extract the unknown "dofval" (in-place).
     * @param dofval input [#ndof]
     * @param dofval_u output [#nnu]
     */
    void asDofs_u(
        const array_type::tensor<double, 1>& dofval,
        array_type::tensor<double, 1>& dofval_u
    ) const
    {
        GOOSEFEM_ASSERT(dofval.size() == this->ndof());
        GOOSEFEM_ASSERT(dofval_u.size() == m_nnu);

#pragma omp parallel for
        for (size_t d = 0; d < m_nnu; ++d) {
            dofval_u(d) = dofval(m_iiu(d));
        }
    }

    /**
     * @brief Convert "nodevec" to "dofval" and extract the unknown "dofval" without a temporary.
     * @param nodevec input [#nnode, #ndim]
     * @return dofval_u output [#nnu]
     */
    array_type::tensor<double, 1> AsDofs_u(const array_type::tensor<double, 2>& nodevec) const
    {
        array_type::tensor<double, 1> dofval_u = xt::empty<double>({m_nnu});
        this->asDofs_u(nodevec, dofval_u);
        return dofval_u;
    }

    /**
     * @brief Convert "nodevec" to "dofval" and extract the unknown "dofval" without a temporary (in-place).
     * @param nodevec input [#nnode, #ndim]
     * @param dofval_u output [#nnu]
     */
    void asDofs_u(
        const array_type::tensor<double, 2>& nodevec,
        array_type::tensor<double, 1>& dofval_u
    ) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(nodevec, {this->nnode(), this->ndim()}));
        GOOSEFEM_ASSERT(dofval_u.size() == m_nnu);

        dofval_u.fill(0.0); // Initialize before filling

#pragma omp parallel for
        for (size_t m = 0; m < this->nnode(); ++m) {
            for (size_t i = 0; i < this->ndim(); ++i) {
                if (m_part_node_view(m, i) < m_nnu) {
                    dofval_u(m_part_node_view(m, i)) = nodevec(m, i);
                }
            }
        }
    }

    /**
     * @brief Convert "elemvec" to "dofval" and extract the unknown "dofval" without a temporary.
     * @param elemvec input [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne].
     * @return dofval_u output [#nnu]
     */
    array_type::tensor<double, 1> AsDofs_u(const array_type::tensor<double, 3>& elemvec, const array_type::tensor<size_t, 2>& conn) const
    {
        array_type::tensor<double, 1> dofval_u = xt::empty<double>({m_nnu});
        this->asDofs_u(elemvec, conn, dofval_u);
        return dofval_u;
    }

    /**
     * @brief Convert "elemvec" to "dofval" and extract the unknown "dofval" without a temporary (in-place).
     * @param elemvec input [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne].
     * @param dofval_u output [#nnu]
     */
    void asDofs_u(
        const array_type::tensor<double, 3>& elemvec,
        const array_type::tensor<size_t, 2>& conn, // Added conn
        array_type::tensor<double, 1>& dofval_u
    ) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(elemvec, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Use conn shapes
        GOOSEFEM_ASSERT(conn.dimension() == 2); // Ensure conn is 2D
        GOOSEFEM_ASSERT(dofval_u.size() == m_nnu);

        dofval_u.fill(0.0); // Initialize before filling

        // This loop cannot be parallelized with OpenMP directly on 'e'
        // due to potential race conditions on 'dofval_u(m_part1d(global_dof_id))'.
        // Multiple elements can share nodes, and thus global DOFs, leading to
        // multiple threads writing to the same partitioned DOF.
        for (size_t e = 0; e < conn.shape(0); ++e) { // Use conn.shape(0) for nelem
            for (size_t m = 0; m < conn.shape(1); ++m) { // Use conn.shape(1) for nne
                for (size_t i = 0; i < this->ndim(); ++i) {
                    size_t global_dof_id = this->dofs()(conn(e, m), i);
                    size_t partitioned_id = m_part1d(global_dof_id);
                    if (partitioned_id < m_nnu) {
                        dofval_u(partitioned_id) = elemvec(e, m, i);
                    }
                }
            }
        }
    }

    /**
     * @brief Extract the prescribed "dofval": `dofval[iip()]`.
     * @param dofval input [#ndof]
     * @return dofval_p output [#nnp]
     */
    array_type::tensor<double, 1> AsDofs_p(const array_type::tensor<double, 1>& dofval) const
    {
        array_type::tensor<double, 1> dofval_p = xt::empty<double>({m_nnp});
        this->asDofs_p(dofval, dofval_p);
        return dofval_p;
    }

    /**
     * @brief Extract the prescribed "dofval" (in-place).
     * @param dofval input [#ndof]
     * @param dofval_p output [#nnp]
     */
    void asDofs_p(
        const array_type::tensor<double, 1>& dofval,
        array_type::tensor<double, 1>& dofval_p
    ) const
    {
        GOOSEFEM_ASSERT(dofval.size() == this->ndof());
        GOOSEFEM_ASSERT(dofval_p.size() == m_nnp);

#pragma omp parallel for
        for (size_t d = 0; d < m_nnp; ++d) {
            dofval_p(d) = dofval(m_iip(d));
        }
    }

    /**
     * @brief Convert "nodevec" to "dofval" and extract the prescribed "dofval" without a temporary.
     * @param nodevec input [#nnode, #ndim]
     * @return dofval_p output [#nnp]
     */
    array_type::tensor<double, 1> AsDofs_p(const array_type::tensor<double, 2>& nodevec) const
    {
        array_type::tensor<double, 1> dofval_p = xt::empty<double>({m_nnp});
        this->asDofs_p(nodevec, dofval_p);
        return dofval_p;
    }

    /**
     * @brief Convert "nodevec" to "dofval" and extract the prescribed "dofval" without a temporary (in-place).
     * @param nodevec input [#nnode, #ndim]
     * @param dofval_p output [#nnp]
     */
    void asDofs_p(
        const array_type::tensor<double, 2>& nodevec,
        array_type::tensor<double, 1>& dofval_p
    ) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(nodevec, {this->nnode(), this->ndim()}));
        GOOSEFEM_ASSERT(dofval_p.size() == m_nnp);

        dofval_p.fill(0.0); // Initialize before filling

#pragma omp parallel for
        for (size_t m = 0; m < this->nnode(); ++m) {
            for (size_t i = 0; i < this->ndim(); ++i) {
                if (m_part_node_view(m, i) >= m_nnu) {
                    dofval_p(m_part_node_view(m, i) - m_nnu) = nodevec(m, i);
                }
            }
        }
    }

    /**
     * @brief Convert "elemvec" to "dofval" and extract the prescribed "dofval" without a temporary.
     * @param elemvec input [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne].
     * @return dofval_p output [#nnp]
     */
    array_type::tensor<double, 1> AsDofs_p(const array_type::tensor<double, 3>& elemvec, const array_type::tensor<size_t, 2>& conn) const
    {
        array_type::tensor<double, 1> dofval_p = xt::empty<double>({m_nnp});
        this->asDofs_p(elemvec, conn, dofval_p);
        return dofval_p;
    }

    /**
     * @brief Convert "elemvec" to "dofval" and extract the prescribed "dofval" without a temporary (in-place).
     * @param elemvec input [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne].
     * @param dofval_p output [#nnp]
     */
    void asDofs_p(
        const array_type::tensor<double, 3>& elemvec,
        const array_type::tensor<size_t, 2>& conn, // Added conn
        array_type::tensor<double, 1>& dofval_p
    ) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(elemvec, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Use conn shapes
        GOOSEFEM_ASSERT(conn.dimension() == 2); // Ensure conn is 2D
        GOOSEFEM_ASSERT(dofval_p.size() == m_nnp);

        dofval_p.fill(0.0); // Initialize before filling

        // This loop cannot be parallelized with OpenMP directly on 'e'
        // due to potential race conditions on 'dofval_p(m_part1d(global_dof_id) - m_nnu)'.
        // Multiple elements can share nodes, and thus global DOFs, leading to
        // multiple threads writing to the same partitioned DOF.
        for (size_t e = 0; e < conn.shape(0); ++e) { // Use conn.shape(0) for nelem
            for (size_t m = 0; m < conn.shape(1); ++m) { // Use conn.shape(1) for nne
                for (size_t i = 0; i < this->ndim(); ++i) {
                    size_t global_dof_id = this->dofs()(conn(e, m), i);
                    size_t partitioned_id = m_part1d(global_dof_id);
                    if (partitioned_id >= m_nnu) {
                        dofval_p(partitioned_id - m_nnu) = elemvec(e, m, i);
                    }
                }
            }
        }
    }
};

} // namespace GooseFEM