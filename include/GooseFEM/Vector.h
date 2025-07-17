/**
 * Class to switch between storage types: "dofval", "nodevec", "elemvec".
 *
 * @file VectorCohesive.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 *
 */

#pragma once

#include "config.h"

namespace GooseFEM {

/**
 * @brief Class to switch between storage types: "dofval", "nodevec", "elemvec".
 *
 * This class provides utilities to convert and assemble data between different
 * representations of vectors on a mesh. It now operates based on a global
 * DOF mapping provided at construction, and requires connectivity (conn)
 * to be passed explicitly for element-based operations.
 */
class Vector {
public:
    Vector() = default;

    /**
     * @brief Constructor.
     *
     * @param dofs DOFs per node [#nnode, #ndim]. This defines the global DOF mapping.
     */
    template <class T_Dofs> // Renamed template parameter for clarity
    Vector(const T_Dofs& dofs) : m_dofs(dofs)
    {
        GOOSEFEM_ASSERT(dofs.dimension() == 2);

        m_nnode = m_dofs.shape(0);
        m_ndim = m_dofs.shape(1);
        m_ndof = xt::amax(m_dofs)() + 1;

        GOOSEFEM_ASSERT(m_ndof <= m_nnode * m_ndim);
    }

    // Removed nelem(), nne(), conn() getters as they are no longer members.

    /**
     * @return Number of nodes.
     */
    size_t nnode() const
    {
        return m_nnode;
    }

    /**
     * @return Number of dimensions.
     */
    size_t ndim() const
    {
        return m_ndim;
    }

    /**
     * @return Number of DOFs.
     */
    size_t ndof() const
    {
        return m_ndof;
    }

    /**
     * @return DOFs per node [#nnode, #ndim]
     */
    const array_type::tensor<size_t, 2>& dofs() const
    {
        return m_dofs;
    }

    /**
     * Copy "nodevec" to another "nodevec".
     *
     * @param nodevec_src input [#nnode, #ndim]
     * @param nodevec_dest input [#nnode, #ndim]
     * @return nodevec output [#nnode, #ndim]
     */
    template <class T>
    T Copy(const T& nodevec_src, const T& nodevec_dest) const
    {
        T ret = T::from_shape(nodevec_dest.shape());
        this->copy(nodevec_src, ret);
        return ret;
    }

    /**
     * Copy "nodevec" to another "nodevec".
     *
     * @param nodevec_src input [#nnode, #ndim]
     * @param nodevec_dest output [#nnode, #ndim]
     */
    template <class T>
    void copy(const T& nodevec_src, T& nodevec_dest) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(nodevec_src, this->shape_nodevec()));
        GOOSEFEM_ASSERT(xt::has_shape(nodevec_dest, this->shape_nodevec()));

        xt::noalias(nodevec_dest) = nodevec_src;
    }

    /**
     * Convert "nodevec" to "dofval" (overwrite entries that occur more than once).
     *
     * @param arg nodevec [#nnode, #ndim]
     * @return dofval [#ndof]
     */
    template <class T>
    array_type::tensor<double, 1> AsDofs(const T& arg) const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>(this->shape_dofval());
        // This will call asDofs_impl_nodevec directly if T is fixed-rank 2,
        // or the dispatching asDofs_impl if T is dynamic-rank.
        this->asDofs_impl(arg, ret);
        return ret;
    }

    /**
     * Convert "nodevec" to "dofval" (overwrite entries that occur more than once).
     *
     * @param arg nodevec [#nnode, #ndim]
     * @param ret dofval (output) [#ndof]
     */
    template <class T, class R>
    void asDofs(const T& arg, R& ret) const
    {
        this->asDofs_impl(arg, ret);
    }

    /**
     * Convert "elemvec" to "dofval" (overwrite entries that occur more than once).
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @return dofval [#ndof]
     */
    template <class T_ElemVec, class T_Conn>
    array_type::tensor<double, 1> AsDofs(const T_ElemVec& arg, const T_Conn& conn) const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>(this->shape_dofval());
        this->asDofs_impl_elemvec(arg, conn, ret); // Pass conn here
        return ret;
    }

    /**
     * Convert "elemvec" to "dofval" (overwrite entries that occur more than once).
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @param ret dofval (output) [#ndof]
     */
    template <class T_ElemVec, class T_Conn, class R>
    void asDofs(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        this->asDofs_impl_elemvec(arg, conn, ret); // Pass conn here
    }

    /**
     * Convert "dofval" or "elemvec" to "nodevec" (overwrite entries that occur more than once).
     *
     * @param arg dofval [#ndof]
     * @return nodevec output [#nnode, #ndim]
     */
    template <class T>
    array_type::tensor<double, 2> AsNode(const T& arg) const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>(this->shape_nodevec());
        this->asNode_impl(arg, ret);
        return ret;
    }

    /**
     * Convert "elemvec" to "nodevec" (overwrite entries that occur more than once).
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @return nodevec output [#nnode, #ndim]
     */
    template <class T_ElemVec, class T_Conn>
    array_type::tensor<double, 2> AsNode(const T_ElemVec& arg, const T_Conn& conn) const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>(this->shape_nodevec());
        this->asNode_impl_elemvec(arg, conn, ret); // Pass conn here
        return ret;
    }

    /**
     * Convert "dofval" or "elemvec" to "nodevec" (overwrite entries that occur more than once).
     *
     * @param arg dofval [#ndof]
     * @param ret nodevec, output [#nnode, #ndim]
     */
    template <class T, class R>
    void asNode(const T& arg, R& ret) const
    {
        this->asNode_impl(arg, ret);
    }

    /**
     * Convert "elemvec" to "nodevec" (overwrite entries that occur more than once).
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @param ret nodevec, output [#nnode, #ndim]
     */
    template <class T_ElemVec, class T_Conn, class R>
    void asNode(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        this->asNode_impl_elemvec(arg, conn, ret); // Pass conn here
    }

    /**
     * Convert "dofval" or "nodevec" to "elemvec" (overwrite entries that occur more than once).
     *
     * @param arg dofval [#ndof] or nodevec [#nnode, #ndim].
     * @param conn connectivity [#nelem, #nne].
     * @return elemvec output [#nelem, #nne, #ndim].
     */
    template <class T_Arg, class T_Conn>
    array_type::tensor<double, 3> AsElement(const T_Arg& arg, const T_Conn& conn) const
    {
        array_type::tensor<double, 3> ret = xt::empty<double>(this->shape_elemvec(conn.shape(0), conn.shape(1))); // Pass nelem, nne
        this->asElement_impl(arg, conn, ret); // Pass conn here
        return ret;
    }

    /**
     * Convert "dofval" or "nodevec" to "elemvec" (overwrite entries that occur more than once).
     *
     * @param arg dofval [#ndof] or nodevec [#nnode, #ndim].
     * @param conn connectivity [#nelem, #nne].
     * @param ret elemvec, output [#nelem, #nne, #ndim].
     */
    template <class T_Arg, class T_Conn, class R>
    void asElement(const T_Arg& arg, const T_Conn& conn, R& ret) const
    {
        this->asElement_impl(arg, conn, ret); // Pass conn here
    }

    /**
     * Assemble "nodevec" to "dofval" (adds entries that occur more that once).
     *
     * @param arg nodevec [#nnode, #ndim]
     * @return dofval output [#ndof]
     */
    template <class T>
    array_type::tensor<double, 1> AssembleDofs(const T& arg) const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>(this->shape_dofval());
        this->assembleDofs_impl(arg, ret);
        return ret;
    }

    /**
     * Assemble "nodevec" to "dofval" (adds entries that occur more that once).
     *
     * @param arg nodevec [#nnode, #ndim]
     * @param ret dofval, output [#ndof]
     */
    template <class T, class R>
    void assembleDofs(const T& arg, R& ret) const
    {
        this->assembleDofs_impl(arg, ret);
    }

    /**
     * Assemble "elemvec" to "dofval" (adds entries that occur more that once).
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @return dofval output [#ndof]
     */
    template <class T_ElemVec, class T_Conn>
    array_type::tensor<double, 1> AssembleDofs(const T_ElemVec& arg, const T_Conn& conn) const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>(this->shape_dofval());
        this->assembleDofs_impl_elemvec(arg, conn, ret); // Pass conn here
        return ret;
    }

    /**
     * Assemble "elemvec" to "dofval" (adds entries that occur more that once).
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @param ret dofval, output [#ndof]
     */
    template <class T_ElemVec, class T_Conn, class R>
    void assembleDofs(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        this->assembleDofs_impl_elemvec(arg, conn, ret); // Pass conn here
    }

    /**
     * Assemble "elemvec" to "nodevec" (adds entries that occur more that once.
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @return nodevec output [#nnode, #ndim]
     */
    template <class T_ElemVec, class T_Conn>
    array_type::tensor<double, 2> AssembleNode(const T_ElemVec& arg, const T_Conn& conn) const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>(this->shape_nodevec());
        this->assembleNode_impl_elemvec(arg, conn, ret); // Pass conn here
        return ret;
    }

    /**
     * Assemble "elemvec" to "nodevec" (adds entries that occur more that once.
     *
     * @param arg elemvec [#nelem, #nne, #ndim]
     * @param conn connectivity [#nelem, #nne]
     * @param ret nodevec, output [#nnode, #ndim]
     */
    template <class T_ElemVec, class T_Conn, class R>
    void assembleNode(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        this->assembleNode_impl_elemvec(arg, conn, ret); // Pass conn here
    }

    /**
     * Shape of "dofval".
     *
     * @return [#ndof]
     */
    std::array<size_t, 1> shape_dofval() const
    {
        return std::array<size_t, 1>{m_ndof};
    }

    /**
     * Shape of "nodevec".
     *
     * @return [#nnode, #ndim]
     */
    std::array<size_t, 2> shape_nodevec() const
    {
        return std::array<size_t, 2>{m_nnode, m_ndim};
    }

    /**
     * Shape of "elemvec".
     *
     * @param nelem Number of elements.
     * @param nne Number of nodes per element.
     * @return [#nelem, #nne, #ndim]
     */
    std::array<size_t, 3> shape_elemvec(size_t nelem, size_t nne) const
    {
        return std::array<size_t, 3>{nelem, nne, m_ndim};
    }

    /**
     * Shape of "elemmat".
     *
     * @param nelem Number of elements.
     * @param nne Number of nodes per element.
     * @return [#nelem, #nne * #ndim, #nne * #ndim]
     */
    std::array<size_t, 3> shape_elemmat(size_t nelem, size_t nne) const
    {
        return std::array<size_t, 3>{nelem, nne * m_ndim, nne * m_ndim};
    }

    /**
     * Allocated "dofval".
     *
     * @return [#ndof]
     */
    array_type::tensor<double, 1> allocate_dofval() const
    {
        array_type::tensor<double, 1> dofval = xt::empty<double>(this->shape_dofval());
        return dofval;
    }

    /**
     * Allocated and initialised "dofval".
     *
     * @param val value to which to initialise.
     * @return [#ndof]
     */
    array_type::tensor<double, 1> allocate_dofval(double val) const
    {
        array_type::tensor<double, 1> dofval = xt::empty<double>(this->shape_dofval());
        dofval.fill(val);
        return dofval;
    }

    /**
     * Allocated "nodevec".
     *
     * @return [#nnode, #ndim]
     */
    array_type::tensor<double, 2> allocate_nodevec() const
    {
        array_type::tensor<double, 2> nodevec = xt::empty<double>(this->shape_nodevec());
        return nodevec;
    }

    /**
     * Allocated and initialised "nodevec".
     *
     * @param val value to which to initialise.
     * @return [#nnode, #ndim]
     */
    array_type::tensor<double, 2> allocate_nodevec(double val) const
    {
        array_type::tensor<double, 2> nodevec = xt::empty<double>(this->shape_nodevec());
        nodevec.fill(val);
        return nodevec;
    }

    /**
     * Allocated "elemvec".
     *
     * @param nelem Number of elements.
     * @param nne Number of nodes per element.
     * @return [#nelem, #nne, #ndim]
     */
    array_type::tensor<double, 3> allocate_elemvec(size_t nelem, size_t nne) const
    {
        array_type::tensor<double, 3> elemvec = xt::empty<double>(this->shape_elemvec(nelem, nne));
        return elemvec;
    }

    /**
     * Allocated and initialised "elemvec".
     *
     * @param val value to which to initialise.
     * @param nelem Number of elements.
     * @param nne Number of nodes per element.
     * @return [#nelem, #nne, #ndim]
     */
    array_type::tensor<double, 3> allocate_elemvec(double val, size_t nelem, size_t nne) const
    {
        array_type::tensor<double, 3> elemvec = xt::empty<double>(this->shape_elemvec(nelem, nne));
        elemvec.fill(val);
        return elemvec;
    }

    /**
     * Allocated "elemmat".
     *
     * @param nelem Number of elements.
     * @param nne Number of nodes per element.
     * @return [#nelem, #nne * #ndim, #nne * #ndim]
     */
    array_type::tensor<double, 3> allocate_elemmat(size_t nelem, size_t nne) const
    {
        array_type::tensor<double, 3> elemmat = xt::empty<double>(this->shape_elemmat(nelem, nne));
        return elemmat;
    }

    /**
     * Allocated and initialised "elemmat".
     *
     * @param val value to which to initialise.
     * @param nelem Number of elements.
     * @param nne Number of nodes per element.
     * @return [#nelem, #nne * #ndim, #nne * #ndim]
     */
    array_type::tensor<double, 3> allocate_elemmat(double val, size_t nelem, size_t nne) const
    {
        array_type::tensor<double, 3> elemmat = xt::empty<double>(this->shape_elemmat(nelem, nne));
        elemmat.fill(val);
        return elemmat;
    }

private:
    // --- Internal implementation methods ---

    /**
     * Dispatch to relevant implementation of \copydoc asDofs(const T&, R&) const
     * For dynamic rank inputs.
     */
    template <class T, class R, typename std::enable_if_t<!xt::has_fixed_rank_t<T>::value, int> = 0>
    void asDofs_impl(const T& arg, R& ret) const
    {
        if (arg.dimension() == 2) {
            this->asDofs_impl_nodevec(arg, ret);
        }
        // Removed the 3D case from here, as it now requires 'conn' and has its own overloads.
        else {
            throw std::runtime_error("Vector::asDofs unknown dimension for conversion (dynamic rank).");
        }
    }

    /**
     * Dispatch to relevant implementation of \copydoc asDofs(const T&, R&) const
     * For fixed rank 2 (nodevec) inputs.
     */
    template <class T, class R, typename std::enable_if_t<xt::get_rank<T>::value == 2, int> = 0>
    void asDofs_impl(const T& arg, R& ret) const
    {
        this->asDofs_impl_nodevec(arg, ret);
    }

    // Removed fixed rank 3 asDofs_impl as it now requires conn.

    /**
     * Dispatch to relevant implementation of \copydoc asNode(const T&, R&) const
     * For dynamic rank inputs.
     */
    template <class T, class R, typename std::enable_if_t<!xt::has_fixed_rank_t<T>::value, int> = 0>
    void asNode_impl(const T& arg, R& ret) const
    {
        if (arg.dimension() == 1) {
            this->asNode_impl_dofval(arg, ret);
        }
        // Removed the 3D case from here, as it now requires 'conn' and has its own overloads.
        else {
            throw std::runtime_error("Vector::asNode unknown dimension for conversion (dynamic rank).");
        }
    }

    /**
     * Dispatch to relevant implementation of \copydoc asNode(const T&, R&) const
     * For fixed rank 1 (dofval) inputs.
     */
    template <class T, class R, typename std::enable_if_t<xt::get_rank<T>::value == 1, int> = 0>
    void asNode_impl(const T& arg, R& ret) const
    {
        this->asNode_impl_dofval(arg, ret);
    }

    // Removed fixed rank 3 asNode_impl as it now requires conn.

    /**
     * Dispatch to relevant implementation of \copydoc asElement(const T&, R&) const
     * For dynamic rank inputs.
     */
    template <class T_Arg, class T_Conn, class R, typename std::enable_if_t<!xt::has_fixed_rank_t<T_Arg>::value, int> = 0>
    void asElement_impl(const T_Arg& arg, const T_Conn& conn, R& ret) const
    {
        if (arg.dimension() == 1) {
            this->asElement_impl_dofval(arg, conn, ret); // Pass conn
        }
        else if (arg.dimension() == 2) {
            this->asElement_impl_nodevec(arg, conn, ret); // Pass conn
        }
        else {
            throw std::runtime_error("Vector::asElement unknown dimension for conversion (dynamic rank).");
        }
    }

    /**
     * Dispatch to relevant implementation of \copydoc asElement(const T&, R&) const
     * For fixed rank 1 (dofval) inputs.
     */
    template <class T_Arg, class T_Conn, class R, typename std::enable_if_t<xt::get_rank<T_Arg>::value == 1, int> = 0>
    void asElement_impl(const T_Arg& arg, const T_Conn& conn, R& ret) const
    {
        this->asElement_impl_dofval(arg, conn, ret); // Pass conn
    }

    /**
     * Dispatch to relevant implementation of \copydoc asElement(const T&, R&) const
     * For fixed rank 2 (nodevec) inputs.
     */
    template <class T_Arg, class T_Conn, class R, typename std::enable_if_t<xt::get_rank<T_Arg>::value == 2, int> = 0>
    void asElement_impl(const T_Arg& arg, const T_Conn& conn, R& ret) const
    {
        this->asElement_impl_nodevec(arg, conn, ret); // Pass conn
    }

    /**
     * Dispatch to relevant implementation of \copydoc assembleDofs(const T&, R&) const
     * For dynamic rank inputs.
     */
    template <class T, class R, typename std::enable_if_t<!xt::has_fixed_rank_t<T>::value, int> = 0>
    void assembleDofs_impl(const T& arg, R& ret) const
    {
        if (arg.dimension() == 2) {
            this->assembleDofs_impl_nodevec(arg, ret);
        }
        // Removed the 3D case from here, as it now requires 'conn' and has its own overloads.
        else {
            throw std::runtime_error("Vector::assembleDofs unknown dimension for conversion (dynamic rank).");
        }
    }

    /**
     * Dispatch to relevant implementation of \copydoc assembleDofs(const T&, R&) const
     * For fixed rank 2 (nodevec) inputs.
     */
    template <class T, class R, typename std::enable_if_t<xt::get_rank<T>::value == 2, int> = 0>
    void assembleDofs_impl(const T& arg, R& ret) const
    {
        this->assembleDofs_impl_nodevec(arg, ret);
    }

    // Removed fixed rank 3 assembleDofs_impl as it now requires conn.

    /**
     * Dispatch to relevant implementation of \copydoc assembleNode(const T&, R&) const
     * For dynamic rank inputs.
     */
    template <class T, class R, typename std::enable_if_t<!xt::has_fixed_rank_t<T>::value, int> = 0>
    void assembleNode_impl(const T& arg, R& ret) const
    {
        if (arg.dimension() == 3) { // Only elemvec is converted to nodevec for dynamic rank
            throw std::runtime_error("Vector::assembleNode requires 'conn' argument for elemvec input (dynamic rank).");
        }
        else {
            throw std::runtime_error("Vector::assembleNode unknown dimension for conversion (dynamic rank).");
        }
    }

    /**
     * Dispatch to relevant implementation of \copydoc assembleNode(const T&, R&) const
     * For fixed rank 3 (elemvec) inputs.
     */
    template <class T, class R, typename std::enable_if_t<xt::get_rank<T>::value == 3, int> = 0>
    void assembleNode_impl(const T& arg, R& ret) const
    {
        throw std::runtime_error("Vector::assembleNode requires 'conn' argument for elemvec input (fixed rank).");
    }

    /**
     * Implementation for 'nodevec' input of \copydoc asDofs(const T&, R&) const
     */
    template <class T, class R>
    void asDofs_impl_nodevec(const T& arg, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 1 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_nodevec()));
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_dofval()));

        ret.fill(0.0);

#pragma omp parallel for
        for (size_t m = 0; m < m_nnode; ++m) {
            for (size_t i = 0; i < m_ndim; ++i) {
                ret(m_dofs(m, i)) = arg(m, i);
            }
        }
    }

    /**
     * Implementation for 'elemvec' input of \copydoc asDofs(const T&, R&) const
     * Now takes 'conn' as an argument.
     */
    template <class T_ElemVec, class T_Conn, class R>
    void asDofs_impl_elemvec(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 1 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Pass nelem, nne
        GOOSEFEM_ASSERT(xt::has_shape(conn, {arg.shape(0), arg.shape(1)})); // Check conn shape
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_dofval()));

        ret.fill(0.0);

        // This loop cannot be parallelized with OpenMP directly on 'e'
        // due to potential race conditions on 'ret(m_dofs(conn(e, m), i))'.
        // If multiple elements can share nodes, then multiple threads might try to write
        // to the same global DOF.
        // For safe parallelization, you'd need atomic operations or a reduction.
        for (size_t e = 0; e < arg.shape(0); ++e) { // Use arg.shape(0) for current nelem
            for (size_t m = 0; m < arg.shape(1); ++m) { // Use arg.shape(1) for current nne
                for (size_t i = 0; i < m_ndim; ++i) {
                    ret(m_dofs(conn(e, m), i)) = arg(e, m, i); // Use conn
                }
            }
        }
    }

    /**
     * Implementation for 'dofval' input of \copydoc asNode(const T&, R&) const
     */
    template <class T, class R>
    void asNode_impl_dofval(const T& arg, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 2 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_dofval()));
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_nodevec()));

#pragma omp parallel for
        for (size_t m = 0; m < m_nnode; ++m) {
            for (size_t i = 0; i < m_ndim; ++i) {
                ret(m, i) = arg(m_dofs(m, i));
            }
        }
    }

    /**
     * Implementation for 'elemvec' input of \copydoc asNode(const T&, R&) const
     * Now takes 'conn' as an argument.
     */
    template <class T_ElemVec, class T_Conn, class R>
    void asNode_impl_elemvec(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 2 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Pass nelem, nne
        GOOSEFEM_ASSERT(xt::has_shape(conn, {arg.shape(0), arg.shape(1)})); // Check conn shape
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_nodevec()));

        ret.fill(0.0); // Clear output before assembly

        // This loop cannot be parallelized with OpenMP directly on 'e'
        // due to potential race conditions on 'ret(conn(e, m), i)'.
        // Multiple elements can share nodes, leading to multiple threads writing
        // to the same global node.
        // For safe parallelization, you'd need atomic operations or a reduction.
        for (size_t e = 0; e < arg.shape(0); ++e) { // Use arg.shape(0) for current nelem
            for (size_t m = 0; m < arg.shape(1); ++m) { // Use arg.shape(1) for current nne
                for (size_t i = 0; i < m_ndim; ++i) {
                    ret(conn(e, m), i) = arg(e, m, i); // Use conn
                }
            }
        }
    }

    /**
     * Implementation for 'dofval' input of \copydoc asElement(const T&, R&) const
     * Now takes 'conn' as an argument.
     */
    template <class T_DofVal, class T_Conn, class R>
    void asElement_impl_dofval(const T_DofVal& arg, const T_Conn& conn, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 3 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(arg.size() == m_ndof);
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Pass nelem, nne
        GOOSEFEM_ASSERT(conn.dimension() == 2); // Basic check for conn

#pragma omp parallel for
        for (size_t e = 0; e < conn.shape(0); ++e) { // Use conn.shape(0) for current nelem
            for (size_t m = 0; m < conn.shape(1); ++m) { // Use conn.shape(1) for current nne
                for (size_t i = 0; i < m_ndim; ++i) {
                    ret(e, m, i) = arg(m_dofs(conn(e, m), i)); // Use conn
                }
            }
        }
    }

    /**
     * Implementation for 'nodevec' input of \copydoc asElement(const T&, R&) const
     * Now takes 'conn' as an argument.
     */
    template <class T_NodeVec, class T_Conn, class R>
    void asElement_impl_nodevec(const T_NodeVec& arg, const T_Conn& conn, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 3 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_nodevec()));
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Pass nelem, nne
        GOOSEFEM_ASSERT(conn.dimension() == 2); // Basic check for conn

#pragma omp parallel for
        for (size_t e = 0; e < conn.shape(0); ++e) { // Use conn.shape(0) for current nelem
            for (size_t m = 0; m < conn.shape(1); ++m) { // Use conn.shape(1) for current nne
                for (size_t i = 0; i < m_ndim; ++i) {
                    ret(e, m, i) = arg(conn(e, m), i); // Use conn
                }
            }
        }
    }

    /**
     * Implementation for 'nodevec' input of \copydoc assembleDofs(const T&, R&) const
     */
    template <class T, class R>
    void assembleDofs_impl_nodevec(const T& arg, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 1 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_nodevec()));
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_dofval()));

        ret.fill(0.0); // Assemble implies adding, so start from zero.

        // This loop cannot be parallelized with OpenMP directly on 'm'
        // due to potential race conditions on 'ret(m_dofs(m, i))'.
        // If m_dofs(m,i) can map to the same global DOF for different 'm',
        // then multiple threads might try to write to the same location.
        // For safe parallelization, you'd need atomic operations or a reduction.
        // For now, keep it serial or use a private sum array per thread.
        for (size_t m = 0; m < m_nnode; ++m) {
            for (size_t i = 0; i < m_ndim; ++i) {
                ret(m_dofs(m, i)) += arg(m, i);
            }
        }
    }

    /**
     * Implementation for 'elemvec' input of \copydoc assembleDofs(const T&, R&) const
     * Now takes 'conn' as an argument.
     */
    template <class T_ElemVec, class T_Conn, class R>
    void assembleDofs_impl_elemvec(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 1 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Pass nelem, nne
        GOOSEFEM_ASSERT(xt::has_shape(conn, {arg.shape(0), arg.shape(1)})); // Check conn shape
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_dofval()));

        ret.fill(0.0); // Assemble implies adding, so start from zero.

        // This loop cannot be parallelized with OpenMP directly on 'e'
        // due to potential race conditions on 'ret(m_dofs(conn(e, m), i))'.
        // Multiple elements can share nodes, leading to multiple threads writing
        // to the same global DOF.
        // For safe parallelization, you'd need atomic operations or a reduction
        // (e.g., coloring, or a private sum for each thread then a final reduction).
        // For now, keep it serial.
        for (size_t e = 0; e < arg.shape(0); ++e) { // Use arg.shape(0) for current nelem
            for (size_t m = 0; m < arg.shape(1); ++m) { // Use arg.shape(1) for current nne
                for (size_t i = 0; i < m_ndim; ++i) {
                    ret(m_dofs(conn(e, m), i)) += arg(e, m, i); // Use conn
                }
            }
        }
    }

    /**
     * Implementation for 'elemvec' input of \copydoc assembleNode(const T&, R&) const
     * Now takes 'conn' as an argument.
     */
    template <class T_ElemVec, class T_Conn, class R>
    void assembleNode_impl_elemvec(const T_ElemVec& arg, const T_Conn& conn, R& ret) const
    {
        static_assert(
            xt::get_rank<R>::value == 2 || !xt::has_fixed_rank_t<R>::value, "Unknown rank 'ret'"
        );
        GOOSEFEM_ASSERT(xt::has_shape(arg, this->shape_elemvec(conn.shape(0), conn.shape(1)))); // Pass nelem, nne
        GOOSEFEM_ASSERT(xt::has_shape(conn, {arg.shape(0), arg.shape(1)})); // Check conn shape
        GOOSEFEM_ASSERT(xt::has_shape(ret, this->shape_nodevec()));

        ret.fill(0.0); // Assemble implies adding, so start from zero.

        // This loop cannot be parallelized with OpenMP directly on 'e'
        // due to potential race conditions on 'ret(conn(e, m), i)'.
        // Multiple elements can share nodes, leading to multiple threads writing
        // to the same global node.
        // For safe parallelization, you'd need atomic operations or a reduction.
        for (size_t e = 0; e < arg.shape(0); ++e) { // Use arg.shape(0) for current nelem
            for (size_t m = 0; m < arg.shape(1); ++m) { // Use arg.shape(1) for current nne
                for (size_t i = 0; i < m_ndim; ++i) {
                    ret(conn(e, m), i) += arg(e, m, i); // Use conn
                }
            }
        }
    }

protected:
    // Removed m_conn, m_nelem, m_nne
    array_type::tensor<size_t, 2> m_dofs; ///< See dofs()
    size_t m_nnode; ///< See #nnode
    size_t m_ndim; ///< See #ndim
    size_t m_ndof; ///< See #ndof
};

} // namespace GooseFEM